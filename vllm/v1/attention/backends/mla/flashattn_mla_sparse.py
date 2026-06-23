# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_supports_mla
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


class FlashAttnMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type["FlashAttnMLASparseMetadataBuilder"]:
        return FlashAttnMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[SparseMLAAttentionImpl[Any]]:
        return FlashAttnMLASparseImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 9

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if kv_cache_dtype not in (None, "auto", "float16", "bfloat16"):
            return (
                "FlashAttention MLA Sparse currently supports only FP16/BF16 KV cache"
            )

        if not flash_attn_supports_mla():
            return "FlashAttention MLA not supported on this device"

        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            hf_config = vllm_config.model_config.hf_config
            if not hasattr(hf_config, "index_topk"):
                return "FlashAttention MLA Sparse requires model with index_topk"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


@dataclass
class FlashAttnMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048
    # DCP context, forwarded from common_attn_metadata for the per-rank
    # top-k slot filtering in forward_mqa (see triton_convert_req_index_to_
    # global_index's cp_rank/cp_size). dcp_world_size<=1 => non-DCP path.
    dcp_world_size: int = 1
    cp_rank: int = 0
    cp_interleave: int = 1


class FlashAttnMLASparseMetadataBuilder(
    AttentionMetadataBuilder[FlashAttnMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.device = device

        # Classify single-token queries (plus num_speculative_tokens via
        # supports_spec_as_decode=True) as decodes. Declare
        # supports_dcp_with_varlen (gated on interleave == 1, mirroring
        # FlashMLA/FlashAttn MLA) so reorder_batch_threshold is not forced
        # back to 1 under DCP -- otherwise MTP verify rows (q_len>1) get
        # classified as prefills and break the full-cudagraph decode capture.
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=(parallel_config.cp_kv_cache_interleave_size == 1),
        )

        self.topk_tokens = vllm_config.model_config.hf_config.index_topk
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttnMLASparseMetadata:
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )

        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        return FlashAttnMLASparseMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,
            slot_mapping=cm.slot_mapping,
            block_table=cm.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[:num_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
            dcp_world_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            cp_interleave=(
                self.vllm_config.parallel_config.cp_kv_cache_interleave_size
            ),
        )


class FlashAttnMLASparseImpl(SparseMLAAttentionImpl[FlashAttnMLASparseMetadata]):
    # DCP requires the per-rank decode LSE for the cross-rank merge; we return
    # it from forward_mqa when return_softmax_lse=True (see forward_mqa).
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl does not support alibi, sliding window, "
                "or logits soft cap."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                "FlashAttnMLASparseImpl currently supports only FP16/BF16 KV cache."
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )
        assert self.topk_indices_buffer is not None, (
            "Indexer or topk_indices_buffer required for sparse MLA"
        )
        self.supports_quant_query_input = False
        # DCP context: mla_attention.py auto-fills dcp_world_size from
        # get_dcp_group() if left at -1, and dispatches the cross-rank LSE
        # merge after forward_mqa returns (attn_out, lse). We only need to
        # return the per-rank lse here. cp_rank/cp_interleave drive the
        # per-rank top-k slot filtering in forward_mqa.
        self.dcp_world_size = -1
        self.dcp_rank = 0
        self.q_pad_num_heads = None

        vllm_config = get_current_vllm_config()
        self.cp_interleave = vllm_config.parallel_config.cp_kv_cache_interleave_size
        # Sparse MLA DCP shards the indexer's top-k selection across DCP ranks
        # and assumes token-interleaved KV layout (CP_INTERLEAVE == 1: rank r
        # owns global positions {r, r+N, ...}). Fail closed rather than
        # silently producing a wrong global top-k; the slot conversion is
        # general but the indexer merge (reused from FlashMLA sparse) is not.
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            if self.cp_interleave != 1:
                raise ValueError(
                    "FlashAttn MLA Sparse with DCP currently requires "
                    "cp_kv_cache_interleave_size == 1, but got "
                    f"{self.cp_interleave}."
                )
            self.dcp_rank = get_dcp_group().rank_in_group

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashAttnMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # q arrives as a (q_nope, q_rope) tuple on the non-DCP path, or as a
        # head-dim-concatenated single tensor (q_nope || q_rope along the last
        # dim) on the DCP path (mla_attention concatenates then head-dim
        # all-gathers before calling forward_mqa). Split the latter back.
        if isinstance(q, tuple):
            q_nope, q_rope = q
        else:
            q_nope = q[..., : self.kv_lora_rank]
            q_rope = q[..., self.kv_lora_rank :]
        num_actual_toks = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        # Under DCP each rank only owns 1/N of the KV. triton_convert_req_index_
        # to_global_index with cp_rank/cp_size marks slots owned by other ranks
        # as -1 (and returns valid_counts = per-row count of this rank's slots).
        # FA3's seqused_k then attends only to the valid (owned) slots; the
        # per-rank lse is returned for the cross-rank merge mla_attention does.
        topk_indices, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
            cp_rank=self.dcp_rank,
            cp_size=self.dcp_world_size if self.dcp_world_size > 0 else 1,
            cp_interleave=self.cp_interleave,
        )

        cu_seqlens_q = torch.arange(
            0, num_actual_toks + 1, dtype=torch.int32, device=q_rope.device
        )
        kv_cache = kv_c_and_k_pe_cache.view(
            -1, attn_metadata.block_size, self.head_size
        )
        k_cache = kv_cache[:, :, self.kv_lora_rank :].view(
            -1, 1, 1, self.qk_rope_head_dim
        )
        v_cache = kv_cache[:, :, : self.kv_lora_rank].view(-1, 1, 1, self.kv_lora_rank)

        # return_softmax_lse only under DCP: the cross-rank LSE merge (in
        # mla_attention.py) needs the per-rank lse. Non-DCP skips it (FA3
        # returns out only, matching the upstream backend).
        return_lse = self.dcp_world_size > 1
        out = flash_attn_varlen_func(
            q=q_rope,
            k=k_cache,
            v=v_cache,
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_indices,
            softmax_scale=self.scale,
            causal=True,
            fa_version=3,
            return_softmax_lse=return_lse,
        )
        if not return_lse:
            return out, None
        out, lse = out  # type: ignore[misc]
        # FA3 returns lse as (nheads, total_q); the DCP LSE reducer
        # (_cp_lse_common / dcp_a2a_lse_reduce) expects [B, H] = (total_q,
        # nheads). Transpose to match.
        lse = lse.transpose(0, 1).contiguous()
        return out, lse
