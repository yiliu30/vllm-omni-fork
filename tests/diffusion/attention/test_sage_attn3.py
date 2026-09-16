# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib
import sys
import types
from unittest.mock import Mock

import pytest
import torch
from vllm.platforms import current_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


SAGE_ATTN3_MODULE = "vllm_omni.diffusion.attention.backends.sage_attn3"


def load_sage_attn3_module(monkeypatch: pytest.MonkeyPatch, kernel_impl):
    import vllm_omni.platforms as platforms

    fake_module = types.ModuleType("sageattn3")
    fake_module.sageattn3_blackwell = kernel_impl
    monkeypatch.setattr(platforms.current_omni_platform, "is_xpu", lambda: False)
    monkeypatch.setitem(sys.modules, "sageattn3", fake_module)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)
    return importlib.import_module(SAGE_ATTN3_MODULE)


def test_xpu_import_selects_deepklox_without_importing_cuda_sageattn3(monkeypatch: pytest.MonkeyPatch):
    import vllm.logger as vllm_logger

    import vllm_omni.platforms as platforms

    calls = {}
    logger = Mock()

    def fake_kernel(query, key, value, tensor_layout):
        calls["tensor_layout"] = tensor_layout
        return query

    deepklox_module = types.ModuleType("deepklox")
    deepklox_module.__path__ = []
    deepklox_module.__file__ = "/test/deepklox/__init__.py"
    interface_module = types.ModuleType("deepklox.sageattn_interface")
    interface_module.sageattn_v3_hybrid = fake_kernel
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sageattn3":
            raise AssertionError("XPU must not import the CUDA sageattn3 package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(platforms.current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(vllm_logger, "init_logger", lambda _: logger)
    monkeypatch.setitem(sys.modules, "deepklox", deepklox_module)
    monkeypatch.setitem(sys.modules, "deepklox.sageattn_interface", interface_module)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)

    backend_module = importlib.import_module(SAGE_ATTN3_MODULE)
    query = torch.randn(1, 2, 1, 128)

    assert backend_module._xpu_v3_hybrid_kernel_call(query, query, query) is query
    assert calls["tensor_layout"] == "NHD"
    logger.warning_once.assert_called_once_with(
        "SAGE_ATTN_3 is using DeepKloX from %s",
        "/test/deepklox/__init__.py",
    )


def test_xpu_import_warns_when_deepklox_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    import vllm.logger as vllm_logger

    import vllm_omni.platforms as platforms

    logger = Mock()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "deepklox.sageattn_interface":
            raise ImportError("deepklox is unavailable")
        if name == "sageattn3":
            raise AssertionError("XPU must not import the CUDA sageattn3 package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(platforms.current_omni_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(vllm_logger, "init_logger", lambda _: logger)
    monkeypatch.delitem(sys.modules, "deepklox.sageattn_interface", raising=False)
    monkeypatch.delitem(sys.modules, "deepklox", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)

    backend_module = importlib.import_module(SAGE_ATTN3_MODULE)

    assert backend_module._xpu_v3_hybrid_kernel_ready() is False
    logger.warning.assert_called_once_with(
        "DeepKLOX SageAttention V3 Hybrid is not available on XPU; "
        "SAGE_ATTN_3 will fall back to flash attention. Install `deepklox` "
        "with SageAttention V3 support."
    )


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


@pytest.mark.parametrize(
    ("mode", "precision"),
    [("sage_v1", "v1fp8"), ("sage_v2", "mxfp8")],
)
def test_xpu_sage_fallback_selects_deepklox_precision(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    precision: str,
):
    backend_module = load_sage_attn3_module(monkeypatch, lambda *args, **kwargs: None)
    calls = {}

    def fake_sageattn(query, key, value, **kwargs):
        calls.update(kwargs)
        return query

    monkeypatch.setattr(backend_module, "sageattn_v1_v2", fake_sageattn)
    query = torch.randn(1, 8, 4, 128)

    output = backend_module._xpu_sage_fallback_kernel_call(
        query,
        query,
        query,
        causal=False,
        softmax_scale=1.0 / 128**0.5,
        mode=mode,
    )

    assert output is query
    assert calls["tensor_layout"] == "NHD"
    assert calls["is_causal"] is False
    assert calls["precision"] == precision


@pytest.mark.parametrize(
    ("mode", "precision"),
    [("sage_v1", "v1fp8_fp8"), ("sage_v2", "fp8")],
)
def test_xpu_sage_fallback_honors_precision_env(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    precision: str,
):
    backend_module = load_sage_attn3_module(monkeypatch, lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN_PRECISION", precision)

    assert backend_module._xpu_sage_fallback_precision(mode) == precision


def test_xpu_sage_fallback_rejects_incompatible_precision(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module(monkeypatch, lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN_PRECISION", "mxfp8")

    with pytest.raises(ValueError, match="incompatible"):
        backend_module._xpu_sage_fallback_precision("sage_v1")


def test_xpu_forced_blocks_use_separate_fallback(monkeypatch: pytest.MonkeyPatch):
    backend_module = load_sage_attn3_module(monkeypatch, lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_FALLBACK", "flash")
    monkeypatch.setattr(backend_module, "_SAGE_ATTN3_FORCE_FALLBACK", "sage_v1")

    assert backend_module._xpu_fallback_mode("cross_sdpa") == "flash"
    assert backend_module._xpu_fallback_mode("sdpa_fallback") == "flash"
    assert backend_module._xpu_fallback_mode("forced_sdpa") == "sage_v1"


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
