#!/usr/bin/env python3
"""
Quantize FLUX.1-dev model to GPTQ-Marlin w4a16 format.

This script quantizes a full-precision FLUX.1-dev model to GPTQ-Marlin w4a16 format
for memory-efficient inference with vLLM-Omni. GPTQ-Marlin provides optimized
kernels for modern CUDA GPUs (compute capability 8.0+).

Example usage:
    # Basic quantization with default settings
    python tasks/standalone_gptq_quantizer/quant_flux.py \
        --input /storage/yiliu7/black-forest-labs/FLUX.1-dev \
        --output /storage/yiliu7/black-forest-labs/FLUX.1-dev-GPTQ-Marlin-w4a16

    # Custom quantization parameters
    python tasks/standalone_gptq_quantizer/quant_flux.py \
        --input /storage/yiliu7/black-forest-labs/FLUX.1-dev \
        --output /storage/yiliu7/black-forest-labs/FLUX.1-dev-GPTQ-Marlin-w4a16 \
        --weight-bits 4 \
        --group-size 64 \
        --desc-act
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file, save_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_model_path(model_path: str) -> Path:
    """Validate that the input model path exists and contains expected files."""
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    if not path.is_dir():
        raise ValueError(f"Model path must be a directory: {model_path}")

    # Check for transformer config specifically
    transformer_config = path / "transformer" / "config.json"
    if not transformer_config.exists():
        raise ValueError(f"Missing transformer config.json at {transformer_config}")

    # Check for model weight files in transformer directory
    transformer_path = path / "transformer"
    has_safetensors = any(transformer_path.glob("*.safetensors"))
    has_bin_files = any(transformer_path.glob("*.bin"))

    if not (has_safetensors or has_bin_files):
        raise ValueError("No transformer weight files (.safetensors or .bin) found in transformer directory")

    logger.info(f"Validated model path: {model_path}")
    return path


def load_model_config(model_path: Path) -> Dict[str, Any]:
    """Load and parse transformer model configuration."""
    config_path = model_path / "transformer" / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded transformer config from {config_path}")
    return config


def update_config_for_gptq_marlin(config: Dict[str, Any], weight_bits: int, group_size: int,
                                 desc_act: bool, lm_head_quantized: bool) -> Dict[str, Any]:
    """Update model config to include GPTQ-Marlin quantization settings."""

    # Add GPTQ-Marlin quantization config
    # Note: GPTQ-Marlin requires symmetric quantization (sym=True)
    quantization_config = {
        "quant_method": "gptq_marlin",  # Use gptq_marlin for vLLM-Omni compatibility
        "weight_bits": weight_bits,
        "group_size": group_size,
        "desc_act": desc_act,
        # "sym": True,  # GPTQ-Marlin requires symmetric quantization
        "lm_head_quantized": lm_head_quantized,
        "true_sequential": True,  # Recommended for better quality
        "use_cuda_fp16": True,    # Use FP16 for CUDA operations
        "model_file_base_name": "model",
    }

    config["quantization_config"] = quantization_config

    logger.info(f"Updated config with GPTQ-Marlin quantization: {quantization_config}")
    return config



def _create_minimal_gptq_config(bits: int = 4, group_size: int = 128, backend: str = "marlin") -> dict:
    """
    Create a minimal GPTQ configuration dictionary for optimum compatibility.

    Args:
        bits: Number of quantization bits
        group_size: Quantization group size
        backend: Backend to use (marlin for Marlin kernels)

    Returns:
        Configuration dictionary compatible with optimum GPTQQuantizer
    """
    return {
        "bits": bits,
        "group_size": group_size,
        "damp_percent": 0.1,
        "desc_act": False,
        "act_group_aware": True,
        "sym": True,
        "true_sequential": True,
        "format": "gptq",
        "backend": backend.lower(),
        "model_seqlen": 2048,  # Default sequence length
        "batch_size": 1,
        "cache_block_outputs": True,
        "quant_method": "gptq"
    }



def quantize_tensor_to_gptq_marlin(weight: torch.Tensor, group_size: int = 128,
                                   bits: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to GPTQ-Marlin format using RTN symmetric quantization.

    Uses GPTQ-style symmetric quantization: values stored as unsigned integers
    [0, 2^bits - 1] with a constant zero point of (maxq + 1) / 2. Weights are
    packed into INT32 following the AutoGPTQ packing convention. The Marlin kernel
    repacks at load time.

    Args:
        weight: Original weight tensor [out_features, in_features] (float32)
        group_size: Group size for quantization (default: 128)
        bits: Number of quantization bits (default: 4, supports 4 or 8)

    Returns:
        qweight: Quantized weights packed as INT32 [in_features // pack_factor, out_features]
        qzeros: Packed zero points as INT32 [num_groups, out_features // pack_factor]
        scales: Scaling factors as FP16 [num_groups, out_features]
    """
    assert weight.dim() == 2, "Only 2D weight tensors supported"
    assert bits in (4, 8), f"Only 4-bit and 8-bit quantization supported, got {bits}"
    out_features, in_features = weight.shape

    pack_factor = 32 // bits  # number of values packed into one int32 (8 for 4-bit)

    # Pad in_features to be divisible by group_size if needed
    if in_features % group_size != 0:
        pad_len = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad_len), value=0.0)
        in_features = weight.shape[1]

    num_groups = in_features // group_size
    maxq = 2 ** bits - 1  # e.g., 15 for 4-bit
    zp = (maxq + 1) // 2  # e.g., 8 for 4-bit (constant for symmetric)

    # --- Step 1: Compute per-group scales (GPTQ symmetric formula) ---
    # Reshape to [out_features, num_groups, group_size]
    w_grouped = weight.reshape(out_features, num_groups, group_size)
    wmax = w_grouped.amax(dim=-1)  # [out_features, num_groups]
    wmin = w_grouped.amin(dim=-1)
    max_abs = torch.max(wmax.abs(), wmin.abs())
    scale = (2.0 * max_abs) / maxq  # [out_features, num_groups]
    scale = scale.clamp(min=1e-5)

    # --- Step 2: Quantize to unsigned int [0, maxq] ---
    # scale: [out_features, num_groups] -> broadcast over group_size
    scale_expanded = scale.unsqueeze(-1)  # [out_features, num_groups, 1]
    intweight = torch.round(w_grouped / scale_expanded) + zp
    intweight = intweight.clamp(0, maxq).to(torch.int32)
    intweight = intweight.reshape(out_features, in_features)

    # --- Step 3: Pack qweight [in_features // pack_factor, out_features] ---
    # Transpose to [in_features, out_features], then group and bit-shift
    intweight_t = intweight.t().contiguous()  # [in_features, out_features]
    intweight_t = intweight_t.reshape(in_features // pack_factor, pack_factor, out_features)
    shifts = (torch.arange(pack_factor, device=weight.device) * bits).unsqueeze(-1)  # [pack_factor, 1]
    qweight = (intweight_t << shifts).sum(dim=1).to(torch.int32)  # [in_features // pack_factor, out_features]

    # --- Step 4: Pack qzeros [num_groups, out_features // pack_factor] ---
    # For symmetric quantization, zp is constant → all qzeros entries are identical
    zp_stored = zp - 1  # optimum/AutoGPTQ convention: subtract 1 before packing
    packed_zp_value = 0
    for j in range(pack_factor):
        packed_zp_value |= zp_stored << (bits * j)
    qzeros = torch.full(
        (num_groups, out_features // pack_factor),
        packed_zp_value,
        dtype=torch.int32,
        device=weight.device,
    )

    # --- Step 5: Scales [num_groups, out_features] as FP16 ---
    scales = scale.t().contiguous().to(torch.float16)  # [num_groups, out_features]

    return qweight, qzeros, scales


def quantize_state_dict(state_dict: Dict[str, torch.Tensor], weight_bits: int = 4,
                       group_size: int = 128) -> Dict[str, torch.Tensor]:
    """
    Real INT4 quantization of linear layer weights to GPTQ-Marlin format.

    This performs actual quantization, converting BF16/FP32 weights to 4-bit integers
    with proper scales, zero points, and group indices for vLLM GPTQ-Marlin kernel.
    Uses symmetric quantization as required by GPTQ-Marlin.
    """
    logger.info("Performing real INT4 quantization of weights (GPTQ-Marlin format)...")

    quantized_state_dict = {}
    quantized_weights = 0
    total_weights = len(state_dict)
    original_size = 0
    quantized_size = 0

    # Define which layers should be quantized (linear layers only)
    # Qwen-Image quantizable patterns
    qwen_quantizable_patterns = [
        'to_q.weight', 'to_k.weight', 'to_v.weight',     # Self-attention
        'add_q_proj.weight', 'add_k_proj.weight', 'add_v_proj.weight',  # Cross-attention
        'to_out.0.weight', 'to_add_out.weight',         # Attention output projections
        "img_mlp", "txt_mlp",
        'ff.net.0.proj.weight', 'ff.net.2.weight',      # Feed-forward layers
        # Optionally include these (commonly excluded from quantization):
        # 'img_in.weight', 'txt_in.weight', 'proj_out.weight'
    ]

    # FLUX-specific quantizable patterns
    flux_quantizable_patterns = [
        'attn.to_qkv.weight',         # Self-attention QKV
        'attn.add_kv_proj.weight',    # Cross-attention QKV
        'attn.to_out.0.weight',       # Attention output projection
        'attn.to_add_out.weight',     # Cross-attention output projection
        'ff.net.0.proj.weight',       # Feed-forward input projection
        'ff.net.2.weight',            # Feed-forward output projection
        # 'proj_mlp.weight',            # Single block MLP projection
        # 'proj_out.weight',            # Single block output projection
        # Optionally include these (commonly excluded from quantization):
        # 'context_embedder.weight', 'x_embedder.weight', 'proj_out.weight'  # Input/output layers
    ]

    # Combine patterns
    quantizable_patterns = qwen_quantizable_patterns + flux_quantizable_patterns

    # Process packed QKV modules first to create proper packed weights
    processed_packed_weights = set()
    quantized_weights = 0
    quantized_size = 0

    # Find all transformer blocks for both Qwen and FLUX models
    transformer_blocks = {}
    single_transformer_blocks = {}
    for name in state_dict.keys():
        if 'transformer_blocks.' in name:
            parts = name.split('.')
            if len(parts) >= 2:
                block_idx = parts[1]
                if block_idx not in transformer_blocks:
                    transformer_blocks[block_idx] = {}
        elif 'single_transformer_blocks.' in name:
            parts = name.split('.')
            if len(parts) >= 2:
                block_idx = parts[1]
                if block_idx not in single_transformer_blocks:
                    single_transformer_blocks[block_idx] = {}

    # Process Qwen-Image packed modules (add_kv_proj) - group add_q_proj, add_k_proj, add_v_proj
    for block_idx in transformer_blocks.keys():
        base_prefix = f"transformer_blocks.{block_idx}.attn"

        # Check for add_q_proj, add_k_proj, add_v_proj weights
        add_q_name = f"{base_prefix}.add_q_proj.weight"
        add_k_name = f"{base_prefix}.add_k_proj.weight"
        add_v_name = f"{base_prefix}.add_v_proj.weight"

        if all(name in state_dict for name in [add_q_name, add_k_name, add_v_name]):
            logger.info(f"Processing packed add_kv_proj for block {block_idx}")

            # Get the individual weight tensors
            add_q_weight = state_dict[add_q_name]
            add_k_weight = state_dict[add_k_name]
            add_v_weight = state_dict[add_v_name]

            # Concatenate weights for packed quantization (Q, K, V along output dimension)
            packed_weight = torch.cat([add_q_weight, add_k_weight, add_v_weight], dim=0)
            logger.info(f"Packed weight shape: {packed_weight.shape}")

            # Quantize the packed weight using GPTQ-Marlin format
            weight_f32 = packed_weight.float()
            qweight, qzeros, scales = quantize_tensor_to_gptq_marlin(weight_f32)

            # Add packed quantized weights with add_kv_proj naming (vLLM expects this)
            packed_base = f"{base_prefix}.add_kv_proj"
            quantized_state_dict[f"{packed_base}.qweight"] = qweight
            quantized_state_dict[f"{packed_base}.qzeros"] = qzeros
            quantized_state_dict[f"{packed_base}.scales"] = scales.half()

            # Track these for size calculation
            quantized_size += qweight.numel() * qweight.element_size()
            quantized_size += qzeros.numel() * qzeros.element_size()
            quantized_size += scales.numel() * scales.element_size()

            quantized_weights += 1

            # Mark these individual weights as processed
            processed_packed_weights.add(add_q_name)
            processed_packed_weights.add(add_k_name)
            processed_packed_weights.add(add_v_name)

    # Process FLUX packed modules (to_qkv, add_kv_proj) - group q, k, v projections
    # Handle dual-stream transformer blocks
    for block_idx in transformer_blocks.keys():
        base_prefix = f"transformer_blocks.{block_idx}.attn"

        # Check for FLUX to_qkv weight (packed QKV for main attention) - only if already packed
        to_qkv_name = f"{base_prefix}.to_qkv.weight"
        if to_qkv_name in state_dict:
            logger.info(f"Processing FLUX to_qkv for transformer block {block_idx}")

            weight_f32 = state_dict[to_qkv_name].float()
            qweight, qzeros, scales = quantize_tensor_to_gptq_marlin(weight_f32)

            # Add packed quantized weights
            quantized_state_dict[f"{base_prefix}.to_qkv.qweight"] = qweight
            quantized_state_dict[f"{base_prefix}.to_qkv.qzeros"] = qzeros
            quantized_state_dict[f"{base_prefix}.to_qkv.scales"] = scales.half()

            # Track these for size calculation
            quantized_size += qweight.numel() * qweight.element_size()
            quantized_size += qzeros.numel() * qzeros.element_size()
            quantized_size += scales.numel() * scales.element_size()

            quantized_weights += 1

        # Check for FLUX add_kv_proj weight (packed QKV for cross-attention)
        add_kv_proj_name = f"{base_prefix}.add_kv_proj.weight"
        if add_kv_proj_name in state_dict:
            logger.info(f"Processing FLUX packed add_kv_proj for block {block_idx}")

            weight_f32 = state_dict[add_kv_proj_name].float()
            qweight, qzeros, scales = quantize_tensor_to_gptq_marlin(weight_f32)

            # Add packed quantized weights
            quantized_state_dict[f"{base_prefix}.add_kv_proj.qweight"] = qweight
            quantized_state_dict[f"{base_prefix}.add_kv_proj.qzeros"] = qzeros
            quantized_state_dict[f"{base_prefix}.add_kv_proj.scales"] = scales.half()

            # Track these for size calculation
            quantized_size += qweight.numel() * qweight.element_size()
            quantized_size += qzeros.numel() * qzeros.element_size()
            quantized_size += scales.numel() * scales.element_size()

            quantized_weights += 1

    # Handle single-stream transformer blocks
    for block_idx in single_transformer_blocks.keys():
        base_prefix = f"single_transformer_blocks.{block_idx}.attn"

        # Check for FLUX to_qkv weight (packed QKV for single-stream attention) - only if already packed
        to_qkv_name = f"{base_prefix}.to_qkv.weight"
        if to_qkv_name in state_dict:
            logger.info(f"Processing FLUX to_qkv for single transformer block {block_idx}")

            weight_f32 = state_dict[to_qkv_name].float()
            qweight, qzeros, scales = quantize_tensor_to_gptq_marlin(weight_f32)

            # Add packed quantized weights
            quantized_state_dict[f"{base_prefix}.to_qkv.qweight"] = qweight
            quantized_state_dict[f"{base_prefix}.to_qkv.qzeros"] = qzeros
            quantized_state_dict[f"{base_prefix}.to_qkv.scales"] = scales.half()

            # Track these for size calculation
            quantized_size += qweight.numel() * qweight.element_size()
            quantized_size += qzeros.numel() * qzeros.element_size()
            quantized_size += scales.numel() * scales.element_size()

            quantized_weights += 1

    # Process remaining weights normally
    for name, tensor in state_dict.items():
        original_size += tensor.numel() * tensor.element_size()

        # Skip already processed packed weights
        if name in processed_packed_weights:
            continue

        # Check if this is a quantizable linear layer weight
        should_quantize = False
        if "weight" in name and tensor.dim() == 2 and tensor.numel() > 1000:
            # Check against quantizable patterns
            for pattern in quantizable_patterns:
                if pattern in name:
                    should_quantize = True
                    break

        if should_quantize:
            logger.info(f"Quantizing {name}: {tensor.shape} ({tensor.dtype})")

            # Convert to float32 for quantization if needed
            weight_f32 = tensor.float()

            # Perform actual INT4 quantization using GPTQ-Marlin format
            qweight, qzeros, scales = quantize_tensor_to_gptq_marlin(weight_f32, group_size)

            # Add quantized weights to state dict with proper naming
            base_name = name.replace('.weight', '')
            quantized_state_dict[f"{base_name}.qweight"] = qweight
            quantized_state_dict[f"{base_name}.qzeros"] = qzeros
            quantized_state_dict[f"{base_name}.scales"] = scales.half()  # Use FP16 for scales

            # Calculate compressed size
            quantized_size += qweight.numel() * qweight.element_size()
            quantized_size += qzeros.numel() * qzeros.element_size()
            quantized_size += scales.numel() * scales.element_size()

            quantized_weights += 1

        else:
            # Keep non-quantizable tensors as-is (biases, norms, embeddings, etc.)
            quantized_state_dict[name] = tensor
            quantized_size += tensor.numel() * tensor.element_size()

    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0

    logger.info(f"Quantized {quantized_weights}/{total_weights} linear layer weights")
    logger.info(f"Original size: {original_size / 1024**3:.2f} GB")
    logger.info(f"Quantized size: {quantized_size / 1024**3:.2f} GB")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")

    return quantized_state_dict


def copy_and_quantize_model(input_path: Path, output_path: Path, config: Dict[str, Any],
                           weight_bits: int = 4, group_size: int = 128):
    """Load model weights, perform actual quantization, and save in GPTQ-Marlin format."""

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Copying and quantizing model files...")

    # Copy non-transformer components (vae, tokenizer, etc.)
    for item in input_path.iterdir():
        if item.name != "transformer":  # Skip transformer, we'll handle it separately
            if item.is_dir():
                shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output_path / item.name)

    # Create transformer directory
    transformer_output_dir = output_path / "transformer"
    transformer_output_dir.mkdir(exist_ok=True)

    # Copy transformer config (non-weight files)
    transformer_input_dir = input_path / "transformer"
    for item in transformer_input_dir.iterdir():
        if not item.name.endswith(('.safetensors', '.bin')):
            if item.is_dir():
                shutil.copytree(item, transformer_output_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, transformer_output_dir / item.name)

    # Load and quantize transformer weights
    logger.info("Loading transformer weights for quantization...")

    # Find safetensors files
    safetensors_files = list(transformer_input_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise ValueError("No safetensors files found in transformer directory")

    # Load index file if it exists
    index_file = transformer_input_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        weight_map = index_data.get('weight_map', {})
        safetensors_files = [transformer_input_dir / f for f in set(weight_map.values())]

    # Load all weights
    all_state_dict = {}
    for safetensors_file in safetensors_files:
        logger.info(f"Loading weights from {safetensors_file.name}")
        state_dict = load_file(safetensors_file)
        all_state_dict.update(state_dict)

    logger.info(f"Loaded {len(all_state_dict)} tensors from {len(safetensors_files)} files")

    # Perform actual quantization
    quantized_state_dict = quantize_state_dict(all_state_dict, weight_bits, group_size)

    # Save quantized weights
    logger.info("Saving quantized weights...")

    # Determine number of shards needed (aim for ~5GB per shard)
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in quantized_state_dict.values())
    bytes_per_shard = 5 * 1024**3  # 5GB
    num_shards = max(1, (total_size + bytes_per_shard - 1) // bytes_per_shard)

    if num_shards == 1:
        # Single file
        output_file = transformer_output_dir / "diffusion_pytorch_model.safetensors"
        save_file(quantized_state_dict, output_file)
        logger.info(f"Saved quantized model to {output_file}")
    else:
        # Multiple shards
        # Create weight map and save shards
        weight_map = {}
        current_shard = {}
        current_size = 0
        shard_idx = 0

        for name, tensor in quantized_state_dict.items():
            tensor_size = tensor.numel() * tensor.element_size()

            # Check if we need to start a new shard
            if current_size + tensor_size > bytes_per_shard and current_shard:
                # Save current shard
                shard_filename = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
                shard_path = transformer_output_dir / shard_filename
                save_file(current_shard, shard_path)
                logger.info(f"Saved shard {shard_idx + 1}/{num_shards}: {shard_filename}")

                shard_idx += 1
                current_shard = {}
                current_size = 0

            # Add tensor to current shard
            current_shard[name] = tensor
            weight_map[name] = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
            current_size += tensor_size

        # Save final shard
        if current_shard:
            shard_filename = f"diffusion_pytorch_model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
            shard_path = transformer_output_dir / shard_filename
            save_file(current_shard, shard_path)
            logger.info(f"Saved final shard: {shard_filename}")

        # Create index file
        index_data = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }
        index_path = transformer_output_dir / "diffusion_pytorch_model.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Saved weight map index: {index_path}")

    # Update the transformer config with GPTQ settings
    transformer_config_path = transformer_output_dir / "config.json"
    with open(transformer_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated transformer config with GPTQ settings at {transformer_config_path}")
    return output_path



def main():
    parser = argparse.ArgumentParser(description="Quantize FLUX.1-dev model to GPTQ-Marlin w4a16 format")

    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input full-precision model directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output quantized model directory")

    # Quantization parameters
    parser.add_argument("--weight-bits", type=int, default=4, choices=[4, 8],
                       help="Number of bits for weight quantization (default: 4) - GPTQ-Marlin supports 4 or 8 bits")
    parser.add_argument("--group-size", type=int, default=128,
                       choices=[-1, 32, 64, 128, 256, 512, 1024],
                       help="Group size for quantization (default: 128, -1 for per-channel)")
    parser.add_argument("--desc-act", action="store_true",
                       help="Use descending order for activation quantization")
    parser.add_argument("--lm-head-quantized", action="store_true",
                       help="Quantize the LM head (output projection layer)")

    # Additional options
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for quantization (default: auto-detect)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting FLUX.1-dev GPTQ-Marlin quantization")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Parameters: bits={args.weight_bits}, group_size={args.group_size}, "
                f"desc_act={args.desc_act}, lm_head_quantized={args.lm_head_quantized}")
    logger.info("Note: GPTQ-Marlin uses symmetric quantization for optimal performance")

    try:
        # Validate input
        input_path = validate_model_path(args.input)
        output_path = Path(args.output)

        # Load model configuration
        config = load_model_config(input_path)

        # Update config for GPTQ-Marlin
        updated_config = update_config_for_gptq_marlin(
            config,
            weight_bits=args.weight_bits,
            group_size=args.group_size,
            desc_act=args.desc_act,
            lm_head_quantized=args.lm_head_quantized
        )

        # Copy model with GPTQ-Marlin config and perform actual quantization
        logger.info("Performing real INT4 quantization (GPTQ-Marlin format)...")
        final_output = copy_and_quantize_model(
            input_path,
            output_path,
            updated_config,
            weight_bits=args.weight_bits,
            group_size=args.group_size
        )

        # Calculate original size
        original_size = sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())

        logger.info(f"GPTQ-Marlin quantization completed successfully!")
        logger.info(f"Original size: {original_size / 1024**3:.2f} GB")
        logger.info(f"Quantized model saved at: {output_path}")

        print(f"\n✅ Successfully quantized FLUX.1-dev model with GPTQ-Marlin at {args.output}")
        print("Linear layer weights have been quantized to 4-bit integers with symmetric quantization")
        print("Compatible with GPTQ-Marlin kernels for optimized inference on modern GPUs")
        print(f"To use with vLLM-Omni:")
        print(f"  python examples/offline_inference/text_to_image/text_to_image.py \\")
        print(f"    --model {args.output} \\")
        print(f"    --quantization gptq \\")
        print(f"    --prompt \"your prompt here\"")

    except Exception as e:
        logger.error(f"Model quantization failed: {e}")
        raise


if __name__ == "__main__":
    main()