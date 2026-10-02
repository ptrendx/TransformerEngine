# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""THD maxima and safely prepared mixed-policy contracts."""

from unittest.mock import patch

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention as dpa
from transformer_engine.pytorch.attention.dot_product_attention import utils


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("maxima", [(None, None), (117, None), (None, 129), (117, 129)])
@pytest.mark.parametrize("same", [False, True])
def test_independent_maxima(device, maxima, same):
    """Missing values remain independent and supplied values are unchanged."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    q = torch.tensor([0, 63, 180, 211], device=device, dtype=torch.int32)
    kv = q if same else torch.tensor([0, 95, 224, 271], device=device, dtype=torch.int32)
    result = utils.get_max_seqlens(q, kv, *maxima)
    assert result == (
        117 if maxima[0] is None else maxima[0],
        (117 if same else 129) if maxima[1] is None else maxima[1],
    )


def test_supplied_maxima_do_not_touch_tensors():
    """The fast path needs no metadata, allocation or device reads."""
    assert utils.get_max_seqlens(None, None, 17, 31) == (17, 31)


def _policies(device="cpu"):
    return [
        dict(
            sequence_ids=torch.tensor(ids, device=device), mask_type="padding", window_size=(-1, -1)
        )
        for ids in ([0, 2], [1])
    ]


def test_prepared_policy_snapshots():
    """Preparation owns snapshots, protects mutations and survives cache eviction."""
    original = _policies()
    prepared = DotProductAttention.prepare_thd_attention_policies(original, 3)
    prepared.check(prepared, 3)
    original[0]["sequence_ids"].fill_(0)
    prepared.check(prepared, 3)
    dpa._thd_policy_validation_cache.clear()
    prepared.check(prepared, 3)
    prepared[0]["sequence_ids"].fill_(0)
    with pytest.raises(RuntimeError, match="prepare_thd_attention_policies again"):
        prepared.check(prepared, 3)


def test_prepared_inference_mode():
    """Snapshots track mutations even when input IDs have no version counter."""
    with torch.inference_mode():
        original = _policies()
        prepared = DotProductAttention.prepare_thd_attention_policies(original, 3)
        assert prepared[0]["sequence_ids"]._version >= 0
        original[0]["sequence_ids"].fill_(0)
        prepared.check(prepared, 3)
    prepared[1]["sequence_ids"] = torch.tensor([1])
    with pytest.raises(RuntimeError, match="sequence_ids changed"):
        prepared.check(prepared, 3)


@pytest.mark.parametrize(
    "ids,error",
    [
        ([2, 0], "strictly ascending"),
        ([0, 0], "strictly ascending"),
        ([0, 3], "every sequence exactly once"),
    ],
)
def test_preparation_rejects_invalid_ids(ids, error):
    policies = _policies()
    policies[0]["sequence_ids"] = torch.tensor(ids)
    with pytest.raises(ValueError, match=error):
        DotProductAttention.prepare_thd_attention_policies(policies, 3)


def test_unversioned_policies_are_revalidated():
    """An inference tensor identity cannot serve as a reusable validation proof."""
    with torch.inference_mode():
        policies = _policies()
        dpa._validate_thd_policy_sequence_ids(policies, 3)
        policies[0]["sequence_ids"].fill_(0)
        with pytest.raises(ValueError, match="strictly ascending"):
            dpa._validate_thd_policy_sequence_ids(policies, 3)


def test_preparation_rejects_capture():
    """Data validation belongs before graph capture."""
    with patch.object(dpa, "is_graph_capturing", return_value=True), pytest.raises(
        RuntimeError, match="outside CUDA graph capture"
    ):
        DotProductAttention.prepare_thd_attention_policies(_policies(), 3)


def test_prepared_length_mutation():
    """Token indices are valid only for the cumulative lengths used at preparation."""
    policies = _policies()
    cu = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
    prepared = DotProductAttention.prepare_thd_attention_policies(
        policies, 3, cu_seqlens_q=cu, cu_seqlens_kv=cu
    )
    prepared.check(prepared, 3, (cu, cu, None, None))
    with pytest.raises(RuntimeError, match="cumulative lengths changed"):
        prepared.check(prepared, 3, (cu.clone(), cu, None, None))
    cu[1] = 1
    with pytest.raises(RuntimeError, match="cumulative lengths changed"):
        prepared.check(prepared, 3, (cu, cu, None, None))


@pytest.mark.parametrize("backend", ["fused", "flash"])
def test_prepared_grouped_capture(backend, monkeypatch):
    """Prepared mixed policies also capture the FA2 grouped token path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if backend == "flash" and not utils.FlashAttentionUtils.is_installed:
        pytest.skip("FlashAttention 2 is required")
    monkeypatch.setenv("NVTE_FUSED_ATTN", str(int(backend == "fused")))
    monkeypatch.setenv("NVTE_FLASH_ATTN", str(int(backend == "flash")))
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    dpa._attention_backends["backend_selection_requires_update"] = True
    cu = torch.tensor([0, 2, 5, 7], device="cuda", dtype=torch.int32)
    kwargs = dict(cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=3, max_seqlen_kv=3)
    policies = DotProductAttention.prepare_thd_attention_policies(
        _policies("cuda"), 3, cu_seqlens_q=cu, cu_seqlens_kv=cu
    )
    module = DotProductAttention(4, 64, qkv_format="thd", attention_dropout=0).cuda().eval()
    tensors = [torch.randn(7, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with torch.no_grad():
        expected = module(*tensors, thd_attention_policies=policies, **kwargs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                module(*tensors, thd_attention_policies=policies, **kwargs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = module(*tensors, thd_attention_policies=policies, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


def test_unversioned_policy_capture_requires_preparation():
    """Warmup cannot safely validate future writes to unversioned ID storage."""
    with torch.inference_mode():
        policies = _policies()
        dpa._validate_thd_policy_sequence_ids(policies, 3)
        with patch.object(dpa, "is_graph_capturing", return_value=True), pytest.raises(
            RuntimeError, match="prepare_thd_attention_policies"
        ):
            dpa._validate_thd_policy_sequence_ids(policies, 3)
