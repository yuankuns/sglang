import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
)
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class ExtendMode:
    def is_decode_or_idle(self):
        return False

    def is_extend_or_draft_extend_or_mixed(self):
        return True

    def is_target_verify(self):
        return False


class TestXPUBackendAppendKV(unittest.TestCase):
    def make_eligibility_case(
        self, head_dim, q_heads, kv_heads, max_q, batch_size=1
    ):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.use_mla = False
        backend.topk = 0
        backend.attention_chunk_size = None
        backend.kv_cache_dtype = torch.bfloat16
        backend.forward_metadata = FlashAttentionMetadata(
            page_table=torch.tensor([[0]], dtype=torch.int32),
            cache_seqlens_int32=torch.zeros(batch_size, dtype=torch.int32),
            max_seq_len_q=max_q,
        )
        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=-1,
            head_dim=head_dim,
            tp_q_head_num=q_heads,
            tp_k_head_num=kv_heads,
        )
        forward_batch = SimpleNamespace(forward_mode=ExtendMode())
        q = torch.empty(1, q_heads, head_dim, dtype=torch.bfloat16)
        k = torch.empty(1, kv_heads, head_dim, dtype=torch.bfloat16)
        return backend, layer, forward_batch, q, k

    def assert_appendkv_policy(
        self,
        expected,
        head_dim,
        q_heads,
        kv_heads,
        max_q,
        batch_size=1,
    ):
        backend, layer, forward_batch, q, k = self.make_eligibility_case(
            head_dim, q_heads, kv_heads, max_q, batch_size
        )
        self.assertEqual(
            backend._appendkv_applicable(
                layer, forward_batch, q, k, k, True
            ),
            expected,
        )

    def test_appendkv_policy_uses_gemma4_shapes(self):
        self.assert_appendkv_policy(True, 256, 8, 4, 256)
        self.assert_appendkv_policy(True, 256, 8, 4, 512)
        self.assert_appendkv_policy(True, 256, 16, 8, 256)
        self.assert_appendkv_policy(True, 256, 16, 8, 256, batch_size=8)
        self.assert_appendkv_policy(True, 512, 8, 1, 256)

    def test_appendkv_policy_keeps_other_supported_families(self):
        self.assert_appendkv_policy(True, 128, 8, 2, 512)

    def test_init_precomputes_appendkv_prefix_lengths(self):
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(32, dtype=torch.int32).view(2, 16)
        )
        backend.use_sliding_window_kv_pool = False
        backend.use_mla = False
        backend.page_size = 1

        forward_batch = SimpleNamespace(
            forward_mode=ExtendMode(),
            seq_lens=torch.tensor([11, 23]),
            seq_lens_cpu=torch.tensor([11, 23]),
            batch_size=2,
            extend_seq_lens=torch.tensor([3, 5]),
            extend_seq_lens_cpu=[3, 5],
            extend_prefix_lens_cpu=[8, 18],
            req_pool_indices=torch.tensor([0, 1]),
            encoder_lens=None,
        )

        backend.init_forward_metadata(forward_batch)

        torch.testing.assert_close(
            backend.forward_metadata.appendkv_cache_seqlens_int32,
            torch.tensor([8, 18], dtype=torch.int32),
        )

    def test_forward_extend_reuses_precomputed_appendkv_metadata(self):
        prefix_lens = torch.tensor([8], dtype=torch.int32)
        cu_seqlens_q = torch.tensor([0, 3], dtype=torch.int32)
        metadata = FlashAttentionMetadata(
            cache_seqlens_int32=torch.tensor([11], dtype=torch.int32),
            appendkv_cache_seqlens_int32=prefix_lens,
            max_seq_len_q=3,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=torch.tensor([0, 11], dtype=torch.int32),
            page_table=torch.tensor([[0]], dtype=torch.int32),
        )
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.forward_metadata = metadata
        backend.use_mla = False
        backend.topk = 0
        backend.attention_chunk_size = None
        backend.use_sliding_window_kv_pool = False
        backend.page_size = 16
        backend.token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda _layer_id: (
                torch.empty(16, 1, 64, dtype=torch.bfloat16),
                torch.empty(16, 1, 64, dtype=torch.bfloat16),
            )
        )
        backend._appendkv_applicable = lambda *_args: True

        layer = SimpleNamespace(
            is_cross_attention=False,
            sliding_window_size=-1,
            layer_id=0,
            tp_q_head_num=1,
            tp_k_head_num=1,
            tp_v_head_num=1,
            head_dim=64,
            v_head_dim=64,
            scaling=0.125,
            logit_cap=0.0,
            k_scale=None,
            v_scale=None,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ExtendMode(),
            out_cache_loc=torch.tensor([8, 9, 10]),
            encoder_out_cache_loc=None,
            _attn_output=None,
        )
        q = torch.empty(3, 1, 64, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)

        with patch(
            "sglang.srt.layers.attention.xpu_backend.flash_attn_with_kvcache",
            return_value=torch.empty_like(q),
        ) as flash_attn:
            backend.forward_extend(q, k, v, layer, forward_batch)

        kwargs = flash_attn.call_args.kwargs
        self.assertIs(kwargs["cache_seqlens"], prefix_lens)
        self.assertIs(kwargs["cu_seqlens_k_new"], cu_seqlens_q)
        self.assertEqual(kwargs["max_seqlen_k"], 3)


if __name__ == "__main__":
    unittest.main()
