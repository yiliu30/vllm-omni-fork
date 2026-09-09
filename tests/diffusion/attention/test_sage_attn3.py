# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch
from vllm.platforms import current_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


SAGE_ATTN3_MODULE = "vllm_omni.diffusion.attention.backends.sage_attn3"


def load_sage_attn3_module(monkeypatch: pytest.MonkeyPatch, kernel_impl):
    fake_module = types.ModuleType("sageattn3")
    fake_module.sageattn3_blackwell = kernel_impl
    monkeypatch.setitem(sys.modules, "sageattn3", fake_module)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)
    return importlib.import_module(SAGE_ATTN3_MODULE)


def load_sage_attn3_module_xpu(monkeypatch: pytest.MonkeyPatch, kernel_impl):
    fake_pkg = types.ModuleType("deepklox")
    fake_iface = types.ModuleType("deepklox.sageattn_interface")
    fake_iface.sageattn_v3_hybrid = kernel_impl
    fake_pkg.sageattn_interface = fake_iface
    monkeypatch.setitem(sys.modules, "deepklox", fake_pkg)
    monkeypatch.setitem(sys.modules, "deepklox.sageattn_interface", fake_iface)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)
    return importlib.import_module(SAGE_ATTN3_MODULE)


def _xpu_impl(backend_module, **kwargs):
    params = dict(num_heads=8, head_size=128, softmax_scale=1.0 / 16.0, causal=False)
    params.update(kwargs)
    return backend_module.SageAttention3Impl(**params)


def test_sage_attn3_module_imports_without_kernels():
    # Kernel packages are acquired lazily; the module must import cleanly
    # even when neither sageattn3 (Blackwell) nor deepklox is installed.
    sys.modules.pop(SAGE_ATTN3_MODULE, None)
    module = importlib.import_module(SAGE_ATTN3_MODULE)
    assert module.SageAttention3Backend.get_name() == "SAGE_ATTN_3"


def test_sage_attn3_xpu_uses_v3_hybrid_kernel(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_op(query, key, value):
        calls["query_dtype"] = query.dtype
        calls["query_shape"] = tuple(query.shape)
        return query + key + value

    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)
    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    impl = _xpu_impl(backend_module)

    query = torch.randn(1, 16, 8, 128, dtype=torch.float16)  # NHD
    key = torch.randn(1, 16, 8, 128, dtype=torch.float16)
    value = torch.randn(1, 16, 8, 128, dtype=torch.float16)

    output = impl.forward_xpu(query, key, value)

    assert calls["query_shape"] == (1, 8, 16, 128)  # HND
    assert calls["query_dtype"] == torch.bfloat16
    expected = (
        query.bfloat16().transpose(1, 2)
        + key.bfloat16().transpose(1, 2)
        + value.bfloat16().transpose(1, 2)
    ).transpose(1, 2).half()
    assert output.dtype == torch.float16
    assert output.shape == (1, 16, 8, 128)
    assert torch.allclose(output, expected, rtol=2e-2, atol=2e-2)


def test_sage_attn3_xpu_cross_attention_falls_back_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)

    def fake_op(*args):
        raise AssertionError("kernel must not be used for cross-attention")

    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    sdpa_calls = {}

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls["key_heads"] = key.shape[1]
        return query + 1

    monkeypatch.setattr(backend_module.F, "scaled_dot_product_attention", fake_sdpa)
    impl = _xpu_impl(backend_module)

    query = torch.randn(1, 512, 8, 128)
    key = torch.randn(1, 77, 8, 128)
    value = torch.randn(1, 77, 8, 128)

    output = impl.forward_xpu(query, key, value)

    assert sdpa_calls["key_heads"] == 8
    assert torch.allclose(output, query + 1, rtol=2e-2, atol=2e-2)


def test_sage_attn3_xpu_forced_sdpa_blocks(monkeypatch: pytest.MonkeyPatch):
    calls = {"kernel": 0, "sdpa": 0}

    def fake_op(query, key, value):
        calls["kernel"] += 1
        return query

    def fake_sdpa(query, key, value, **kwargs):
        calls["sdpa"] += 1
        return query + 1

    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)
    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    monkeypatch.setattr(backend_module.F, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_FORCE_SDPA_BLOCKS", frozenset({3}))

    qkv = [torch.randn(1, 16, 8, 128) for _ in range(3)]

    forced = _xpu_impl(backend_module, prefix="blocks.3.self_attn.q_proj")
    assert forced.layer_idx == 3
    out_forced = forced.forward_xpu(*qkv)
    assert calls == {"kernel": 0, "sdpa": 1}
    assert torch.allclose(out_forced, qkv[0] + 1, rtol=2e-2, atol=2e-2)

    regular = _xpu_impl(backend_module, prefix="blocks.5.self_attn.q_proj")
    assert regular.layer_idx == 5
    out_regular = regular.forward_xpu(*qkv)
    assert calls == {"kernel": 1, "sdpa": 1}
    assert torch.allclose(out_regular, qkv[0], rtol=2e-2, atol=2e-2)


def test_sage_attn3_xpu_contract_violations_fall_back_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)

    def fake_op(*args):
        raise AssertionError("kernel must not be used outside its contract")

    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    monkeypatch.setattr(
        backend_module.F, "scaled_dot_product_attention", lambda q, k, v, **kw: q
    )

    cases = [
        # GQA: kv heads != query heads
        (dict(), dict(batch=1, seq=16, hq=8, hkv=4, head=128, causal=False)),
        # causal self-attention
        (dict(causal=True), dict(batch=1, seq=16, hq=8, hkv=8, head=128, causal=True)),
        # unsupported head size
        (dict(head_size=64, softmax_scale=1.0 / 8.0), dict(batch=1, seq=16, hq=8, hkv=8, head=64, causal=False)),
        # batch > 1
        (dict(), dict(batch=2, seq=16, hq=8, hkv=8, head=128, causal=False)),
    ]
    for init_kwargs, shape in cases:
        impl = _xpu_impl(backend_module, **init_kwargs)
        query = torch.randn(shape["batch"], shape["seq"], shape["hq"], shape["head"])
        key = torch.randn(shape["batch"], shape["seq"], shape["hkv"], shape["head"])
        value = torch.randn(shape["batch"], shape["seq"], shape["hkv"], shape["head"])
        output = impl.forward_xpu(query, key, value)
        assert torch.allclose(output, query, rtol=2e-2, atol=2e-2)


def test_sage_attn3_xpu_unsupported_device_falls_back_and_stops_retrying(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"op": 0}

    def fake_op(query, key, value):
        calls["op"] += 1
        raise RuntimeError("sageattn_v3_hybrid is supported on CRI (Xe3P) only")

    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)
    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    monkeypatch.setattr(
        backend_module.F, "scaled_dot_product_attention", lambda q, k, v, **kw: q + 2
    )
    impl = _xpu_impl(backend_module)
    qkv = [torch.randn(1, 16, 8, 128) for _ in range(3)]

    out_first = impl.forward_xpu(*qkv)
    assert calls["op"] == 1
    assert not backend_module._xpu_v3_hybrid_available

    out_second = impl.forward_xpu(*qkv)
    assert calls["op"] == 1  # state short-circuits the kernel path
    assert torch.allclose(out_first, qkv[0] + 2, rtol=2e-2, atol=2e-2)
    assert torch.allclose(out_second, qkv[0] + 2, rtol=2e-2, atol=2e-2)


def test_sage_attn3_xpu_missing_kernel_falls_back_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)

    def fake_op(*args):
        raise AssertionError("kernel must not be called when unavailable")

    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    monkeypatch.setattr(
        backend_module.F, "scaled_dot_product_attention", lambda q, k, v, **kw: q + 3
    )
    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_available", False)
    impl = _xpu_impl(backend_module)

    qkv = [torch.randn(1, 16, 8, 128) for _ in range(3)]
    assert torch.allclose(impl.forward_xpu(*qkv), qkv[0] + 3, rtol=2e-2, atol=2e-2)


def test_sage_attn3_xpu_nonfinite_output_falls_back(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)

    def fake_op(query, key, value):
        return torch.full_like(query, float("inf"))

    sdpa_calls = {"count": 0}

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls["count"] += 1
        return query + 4

    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    monkeypatch.setattr(backend_module.F, "scaled_dot_product_attention", fake_sdpa)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_DEBUG_CHECK_FINITE", True)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_REPORT_FALLBACKS", True)
    impl = _xpu_impl(backend_module)

    qkv = [torch.randn(1, 16, 8, 128) for _ in range(3)]
    output = impl.forward_xpu(*qkv)

    assert sdpa_calls["count"] == 1
    assert torch.allclose(output, qkv[0] + 4, rtol=2e-2, atol=2e-2)
    assert backend_module.get_sage_attn3_call_counts()["nonfinite_sdpa"] == 1
    assert backend_module.get_sage_attn3_call_counts()["sage"] == 0


def test_sage_attn3_call_counters(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module_xpu(monkeypatch, lambda *args: None)

    def fake_op(query, key, value):
        return query

    monkeypatch.setattr(backend_module, "_xpu_v3_hybrid_kernel_call", fake_op)
    monkeypatch.setattr(
        backend_module.F, "scaled_dot_product_attention", lambda q, k, v, **kw: q
    )
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_REPORT_FALLBACKS", True)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_FORCE_SDPA_BLOCKS", frozenset({7}))

    kernel_impl = _xpu_impl(backend_module)
    cross_impl = _xpu_impl(backend_module)
    forced_impl = _xpu_impl(backend_module, prefix="blocks.7.self_attn.q_proj")

    self_qkv = [torch.randn(1, 16, 8, 128) for _ in range(3)]
    cross_qkv = [
        torch.randn(1, 16, 8, 128),
        torch.randn(1, 4, 8, 128),
        torch.randn(1, 4, 8, 128),
    ]

    kernel_impl.forward_xpu(*self_qkv)
    cross_impl.forward_xpu(*cross_qkv)
    forced_impl.forward_xpu(*self_qkv)

    counts = backend_module.get_sage_attn3_call_counts()
    assert counts["sage"] == 1
    assert counts["cross_sdpa"] == 1
    assert counts["forced_sdpa"] == 1
    assert counts["sdpa_fallback"] == 0
    assert counts["nonfinite_sdpa"] == 0

    backend_module.reset_sage_attn3_call_counts()
    assert backend_module.get_sage_attn3_call_counts() == {
        "sage": 0,
        "forced_sdpa": 0,
        "cross_sdpa": 0,
        "sdpa_fallback": 0,
        "nonfinite_sdpa": 0,
    }


def test_sage_attn3_forward_uses_blackwell_layout(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_kernel(query, key, value, is_causal=False):
        calls["query_shape"] = query.shape
        calls["is_causal"] = is_causal
        return query + key + value

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    impl = backend_module.SageAttention3Impl(
        num_heads=4,
        head_size=64,
        softmax_scale=1.0 / 8.0,
        causal=False,
    )

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 4, 64)
    value = torch.randn(2, 8, 4, 64)

    output = impl.forward_cuda(query, key, value)

    assert calls["query_shape"] == (2, 4, 8, 64)
    assert calls["is_causal"] is False
    expected = (query.transpose(1, 2) + key.transpose(1, 2) + value.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(output, expected)


def test_sage_attn3_falls_back_to_sdpa_for_gqa(monkeypatch: pytest.MonkeyPatch):
    def fake_kernel(*args, **kwargs):
        raise AssertionError("sageattn3_blackwell should not be used for GQA")

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    sdpa_calls = {}

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls["query_shape"] = query.shape
        sdpa_calls["key_shape"] = key.shape
        sdpa_calls["enable_gqa"] = kwargs["enable_gqa"]
        return query + 1

    monkeypatch.setattr(backend_module.F, "scaled_dot_product_attention", fake_sdpa)

    impl = backend_module.SageAttention3Impl(
        num_heads=4,
        head_size=64,
        softmax_scale=1.0 / 8.0,
        causal=False,
    )

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 2, 64)
    value = torch.randn(2, 8, 2, 64)

    output = impl.forward_cuda(query, key, value)

    assert sdpa_calls["query_shape"] == (2, 4, 8, 64)
    assert sdpa_calls["key_shape"] == (2, 2, 8, 64)
    assert sdpa_calls["enable_gqa"] is True
    expected = (query.permute(0, 2, 1, 3) + 1).permute(0, 2, 1, 3)
    assert torch.allclose(output, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_selects_sage_attn3_alias(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda import platform as cuda_platform_module
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    original_import_module = importlib.import_module

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(
        cuda_platform_module.importlib,
        "import_module",
        lambda module_name: object() if module_name == "sageattn3" else original_import_module(module_name),
    )

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.SAGE_ATTN_3.get_path()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_falls_back_when_sage_attn3_gpu_is_unsupported(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
