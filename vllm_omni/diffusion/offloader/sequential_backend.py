# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig, SupportsModelCpuOffload
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion on encoder and DiT modules.

    To be used as a model-level (or "component-level") of CPU offloading method;
    When a module's forward is called, this hook offloads target modules to CPU
    and loads the current module to GPU.
    """

    _HOOK_NAME = "sequential_offload"

    def __init__(
        self,
        offload_targets: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp

    @staticmethod
    def _move_params(
        module: nn.Module,
        target_device: torch.device,
        *,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        for p in module.parameters():
            if p.data.device != target_device:
                data = p.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                p.data = data
        for b in module.buffers():
            if b.device != target_device:
                data = b.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                b.data = data

    @staticmethod
    def _tensors(module: nn.Module) -> list[torch.Tensor]:
        """Parameters and buffers as a flat list, in move order."""
        return [*module.parameters(), *module.buffers()]

    def _move_tensor(self, holder: torch.Tensor, target_device: torch.device, *, non_blocking: bool) -> None:
        """Move one parameter/buffer in place, mirroring _move_params per tensor."""
        if holder.data.device == target_device:
            return
        data = holder.data.to(target_device, non_blocking=non_blocking)
        if self.pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
            data = data.pin_memory()
        holder.data = data

    def _swap_interleaved(self, out_modules: list[nn.Module], in_module: nn.Module) -> None:
        """Evict ``out_modules`` while loading ``in_module``, keeping host use flat.

        The straightforward order -- evict everything, then load -- transiently holds
        both sides in host memory. For two same-sized DiTs that doubles the host
        footprint at the boundary crossing, which is fatal where host RAM is the
        scarce resource (61 GiB host vs 121 GiB device here: 2 x 27 GiB of bf16
        transformer does not fit, and pinned pages cannot even swap).

        So free before allocating, tensor by tensor: for each incoming tensor, evict
        at least as many bytes from the outgoing side first. Host usage then stays at
        roughly one module instead of two, and the device peak is unchanged because
        eviction still leads.
        """
        cpu = torch.device("cpu")
        # Same rule as _to_cpu: XPU's allocator does not respect stream dependencies
        # in empty_cache, so non-blocking copies can race with cache eviction.
        non_blocking = not self.use_hsdp and not current_omni_platform.is_xpu()

        out_tensors = [t for m in out_modules for t in self._tensors(m) if t.data.device != cpu]
        in_tensors = [t for t in self._tensors(in_module) if t.data.device != self.device]

        out_idx = 0
        for holder in in_tensors:
            budget = holder.data.numel() * holder.data.element_size()
            freed = 0
            while out_idx < len(out_tensors) and freed < budget:
                victim = out_tensors[out_idx]
                out_idx += 1
                freed += victim.data.numel() * victim.data.element_size()
                self._move_tensor(victim, cpu, non_blocking=non_blocking)
            self._move_tensor(holder, self.device, non_blocking=False)

        # Anything left on the outgoing side (in_module smaller than out_modules).
        while out_idx < len(out_tensors):
            self._move_tensor(out_tensors[out_idx], cpu, non_blocking=non_blocking)
            out_idx += 1

        current_omni_platform.empty_cache()

    def _to_cpu(self, module: nn.Module) -> None:
        try:
            param = next(module.parameters())
        except StopIteration:
            return

        if param.device.type == "cpu":
            return

        # XPU's allocator doesn't respect stream dependencies in empty_cache,
        # so non-blocking copies can race with cache eviction. Use blocking
        # copies on XPU to avoid NULL pointer errors during DMA.
        non_blocking = not self.use_hsdp and not current_omni_platform.is_xpu()
        self._move_params(
            module,
            torch.device("cpu"),
            non_blocking=non_blocking,
            pin_memory=self.pin_memory,
        )
        current_omni_platform.empty_cache()

    def _to_gpu(self, module: nn.Module) -> None:
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        self._move_params(module, self.device, non_blocking=False)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        # When this module has to come in AND targets have to go out, interleave the
        # two so host memory holds ~one module rather than both. Otherwise fall back
        # to the simple order (nothing to bring in, or nothing resident to evict).
        needs_load = any(t.data.device != self.device for t in self._tensors(module))
        cpu = torch.device("cpu")
        evictable = [t for m in self.offload_targets for t in self._tensors(m) if t.data.device != cpu]
        if needs_load and evictable:
            self._swap_interleaved(self.offload_targets, module)
        else:
            for target in self.offload_targets:
                self._to_cpu(target)
            self._to_gpu(module)
        current_omni_platform.synchronize()

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


def apply_sequential_offload(
    dit_modules: list[nn.Module],
    encoder_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
    offload_initial_dits: bool = False,
) -> None:
    """Apply sequential offloading hooks to DiT and encoder modules.

    Registers hooks on modules to implement mutual-exclusion GPU allocation.
        - Before DiT runs, encoders are offloaded to CPU.
        - Before encoders run, DiT is offloaded to CPU.

    Args:
        dit_modules: DiT/transformer modules to register hooks on
        encoder_modules: Encoder modules to register hooks on
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior)
        offload_initial_dits: Whether to begin with all DiT modules on CPU.

    Example:
        >>> apply_sequential_offload(
        ...     dit_modules=[pipeline.transformer],
        ...     encoder_modules=[pipeline.text_encoder, pipeline.vae],
        ...     device=torch.device("cuda:0"),
        ... )
        >>> # Modules of pipeline now automatically swap between CPU and GPU
    """
    # Register hooks on DiT modules (offload encoders AND other DiTs when a DiT runs)
    for i, dit_mod in enumerate(dit_modules):
        other_dits = [d for j, d in enumerate(dit_modules) if j != i]
        registry = HookRegistry.get_or_create(dit_mod)
        hook = SequentialOffloadHook(
            offload_targets=encoder_modules + other_dits,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

    # Register hooks on encoders (offload DiTs when encoder runs)
    for enc in encoder_modules:
        registry = HookRegistry.get_or_create(enc)
        hook = SequentialOffloadHook(
            offload_targets=dit_modules,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", enc.__class__.__name__)

    if offload_initial_dits:
        try:
            for dit_mod in dit_modules:
                _get_sequential_offload_hook(dit_mod)._to_cpu(dit_mod)
        except Exception:
            remove_sequential_offload([*dit_modules, *encoder_modules])
            raise


def apply_selective_offload(
    offload_modules: list[nn.Module],
    resident_consumers: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
    mutual_exclusion_targets: list[nn.Module] | None = None,
) -> None:
    """Offload only ``offload_modules``; keep everything else device-resident.

    Reuses :class:`SequentialOffloadHook` unchanged -- the hook moves its
    ``offload_targets`` to CPU and the hooked module to GPU on forward. Two
    registrations express "only these components leave the device":

      * on each offloaded module, targeting its offloaded siblings, so at most
        one offloaded component is on the device at a time;
      * on each resident consumer, targeting all offloaded modules, so they are
        evicted before the big consumer runs. Without this the memory is never
        actually reclaimed and the feature buys nothing.

    ``_to_gpu`` on an already-resident module is a no-op copy (``Tensor.to`` of
    the same device returns self), so hooking a resident consumer costs a walk
    over the offloaded modules' parameters, not a transfer.

    Args:
        offload_modules: Components to keep on CPU between uses.
        resident_consumers: Device-resident modules whose forward should evict
            the offloaded set -- normally the DiT(s).
        device: Target device for loading.
        pin_memory: Pin CPU memory for faster transfers.
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior).
    """
    mutual_exclusion_targets = mutual_exclusion_targets or []

    for i, mod in enumerate(offload_modules):
        siblings = [m for j, m in enumerate(offload_modules) if j != i]
        # Mutual exclusion with the resident consumers, not just the siblings: when
        # an offloaded DiT runs it must push the resident DiT out, otherwise both
        # are on the device at once and the peak is unchanged -- the offload would
        # save nothing. SequentialOffloadHook.pre_forward evicts every target before
        # loading itself, so the order is evict-then-load, which keeps the device
        # peak at one DiT rather than two.
        registry = HookRegistry.get_or_create(mod)
        registry.register_hook(
            SequentialOffloadHook._HOOK_NAME,
            SequentialOffloadHook(
                offload_targets=siblings + list(mutual_exclusion_targets),
                device=device,
                pin_memory=pin_memory,
                use_hsdp=use_hsdp,
            ),
        )
        logger.debug("Selective offload: hooked %s (offloaded)", mod.__class__.__name__)

    for consumer in resident_consumers:
        registry = HookRegistry.get_or_create(consumer)
        registry.register_hook(
            SequentialOffloadHook._HOOK_NAME,
            SequentialOffloadHook(
                offload_targets=list(offload_modules),
                device=device,
                pin_memory=pin_memory,
                use_hsdp=use_hsdp,
            ),
        )
        logger.debug("Selective offload: hooked %s (resident consumer)", consumer.__class__.__name__)


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is not None:
            registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
            logger.debug("Removed offload hook from %s", module.__class__.__name__)


def _get_sequential_offload_hook(module: nn.Module) -> SequentialOffloadHook:
    registry: HookRegistry | None = getattr(module, "_hook_registry", None)
    hook = registry.get_hook(SequentialOffloadHook._HOOK_NAME) if registry is not None else None
    if not isinstance(hook, SequentialOffloadHook):
        raise RuntimeError(f"{module.__class__.__name__} has no sequential offload hook")
    return hook


@contextmanager
def sequential_offload_component(module: nn.Module) -> Iterator[None]:
    """Activate and release a hooked component called outside ``forward``."""
    hook = _get_sequential_offload_hook(module)
    try:
        hook.pre_forward(module)
        yield
    except BaseException:
        try:
            hook._to_cpu(module)
        except Exception:
            logger.exception("Failed to release %s after component failure", module.__class__.__name__)
        raise
    else:
        hook._to_cpu(module)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Uses SequentialOffloadHook registered via HookRegistry for automatic module swapping.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # Track modules with hooks
        self._custom_pipeline: SupportsModelCpuOffload | None = None

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return

        # Pipelines with non-forward component entry points own their complete
        # mutual-exclusion lifecycle. Delegate through the explicit protocol.
        if isinstance(pipeline, SupportsModelCpuOffload):
            pipeline.enable_omni_model_cpu_offload(
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                use_hsdp=self.config.use_hsdp,
            )
            self._custom_pipeline = pipeline
            self.enabled = True
            logger.info(
                "Model-level offloading enabled through %s.enable_omni_model_cpu_offload",
                pipeline.__class__.__name__,
            )
            return

        modules = ModuleDiscovery.discover(pipeline)

        if self.config.offload_models:
            self._enable_selective(pipeline, modules)
            return

        # Move encoders to GPU
        for enc in modules.encoders:
            enc.to(self.device)

        # Move VAE(s) to GPU if available
        for vae in modules.vaes:
            try:
                vae.to(self.device, non_blocking=True)
            except Exception as exc:
                logger.debug("Failed to move VAE to GPU: %s", exc)

        # Pin resident modules on GPU (small hot submodules called inside the DiT loop).
        for res, name in zip(modules.resident_modules, modules.resident_names):
            try:
                res.to(self.device)
            except Exception as exc:
                logger.warning("Failed to move resident module '%s' to GPU: %s", name, exc)

        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping model-level offloading")
            return

        if not modules.encoders:
            # Nothing to swap against — move DiTs to GPU and skip hooks.
            for dit in modules.dits:
                dit.to(self.device)
            logger.warning("No encoder modules found, skipping model-level offloading")
            return

        # Apply sequential offloading hooks
        apply_sequential_offload(
            dit_modules=modules.dits,
            encoder_modules=modules.encoders,
            device=self.device,
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
        )

        # Track modules for cleanup
        self._offload_modules = [*modules.dits, *modules.encoders]

        self.enabled = True

        logger.info(
            "Model-level offloading enabled: %s <-> %s (mutual exclusion)%s",
            ", ".join(modules.dit_names),
            ", ".join(modules.encoder_names),
            f"; resident on GPU: {', '.join(modules.resident_names)}" if modules.resident_names else "",
        )


    def _enable_selective(self, pipeline: nn.Module, modules) -> None:
        """Offload only the components named in ``config.offload_models``.

        Everything discovered but unnamed is moved to the device and stays there.
        Unlike the category-based path, this makes no assumption about which role
        should be evicted -- a text encoder used once per request can be the only
        thing on CPU while both DiTs stay resident.
        """
        by_name: dict[str, nn.Module] = {}
        for mods, names in (
            (modules.dits, modules.dit_names),
            (modules.encoders, modules.encoder_names),
            (modules.vaes, modules.vae_names),
            (modules.resident_modules, modules.resident_names),
        ):
            for mod, name in zip(mods, names):
                by_name.setdefault(name, mod)

        requested = list(self.config.offload_models)
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            # Fail loudly: a typo would otherwise offload nothing and quietly load
            # the whole pipeline onto the device, which is the opposite of asked.
            raise ValueError(
                f"cpu_offload_models names not found on {pipeline.__class__.__name__}: "
                f"{unknown}. Discovered components: {sorted(by_name)}"
            )

        offload_names = [name for name in requested if name in by_name]
        offload_modules = [by_name[name] for name in offload_names]
        offload_ids = {id(m) for m in offload_modules}

        resident_names = [name for name in by_name if name not in offload_names]
        for name in resident_names:
            mod = by_name[name]
            try:
                mod.to(self.device)
            except Exception as exc:
                logger.warning("Failed to move resident component '%s' to %s: %s", name, self.device, exc)

        # DiTs drive the memory peak, so they are what should evict the offloaded
        # set. A DiT that is itself offloaded is handled by the sibling logic.
        resident_consumers = [d for d in modules.dits if id(d) not in offload_ids]
        if not resident_consumers:
            logger.warning(
                "Selective offload: no device-resident DiT to trigger eviction; "
                "offloaded components will only be swapped against each other."
            )

        # Only DiTs mutually exclude. A resident encoder/VAE is resident by user
        # choice and must not be dragged off the device by a DiT forward.
        apply_selective_offload(
            offload_modules=offload_modules,
            resident_consumers=resident_consumers,
            device=self.device,
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
            mutual_exclusion_targets=[m for m in resident_consumers if id(m) not in offload_ids],
        )

        # Weights were loaded on the device (see diffusion_model_runner), so the
        # named components start resident and must be evicted once here. Roll the
        # hooks back on failure rather than leaving a half-applied state.
        try:
            for name, mod in zip(offload_names, offload_modules):
                _get_sequential_offload_hook(mod)._to_cpu(mod)
                logger.debug("Selective offload: evicted '%s' to CPU", name)
        except Exception:
            remove_sequential_offload([*offload_modules, *resident_consumers])
            raise
        self._offload_modules = [*offload_modules, *resident_consumers]
        self.enabled = True
        logger.info(
            "Selective model-level offloading enabled: on CPU %s; device-resident %s",
            offload_names,
            resident_names,
        )

    def disable(self) -> None:
        if not self.enabled:
            return

        if self._custom_pipeline is not None:
            self._custom_pipeline.disable_omni_model_cpu_offload()
            self._custom_pipeline = None
            self.enabled = False
            logger.info("Model-level offloading disabled")
            return

        remove_sequential_offload(self._offload_modules)

        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")
