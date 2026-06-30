"""Test loading the quantized Cosmos3 pipeline model through omni-wm."""
import time
import torch

if __name__ == '__main__':
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import build_text_to_image_prompt
    from PIL import Image

    start = time.perf_counter()
    omni = Omni(
        model='/storage/yiliu7/cosmos3-nano-w4a16-pipeline',
        model_config={"guardrails": False},
        mode='text-to-image',
        enforce_eager=True,
    )
    print(f'Omni initialized in {time.perf_counter() - start:.1f}s')

    generator = torch.Generator(device='cuda').manual_seed(42)

    prompt_dict = build_text_to_image_prompt(
        model_class_name="Cosmos3OmniDiffusersPipeline",
        prompt="A photorealistic red sports car at golden hour, cinematic lighting.",
        negative_prompt="blurry, distorted, low quality",
        height=256, width=256,
    )

    diffusion_params = OmniDiffusionSamplingParams(
        height=256, width=256,
        seed=42,
        generator=generator,
        guidance_scale=1.0,
        num_inference_steps=40,
        num_outputs_per_prompt=1,
    )

    gen_start = time.perf_counter()
    outputs = omni.generate(prompt_dict, sampling_params_list=[diffusion_params])
    gen_time = time.perf_counter() - gen_start
    print(f'Generated in {gen_time:.1f}s')

    for output in outputs:
        images = getattr(output, 'images', None)
        if not images:
            req_out = getattr(output, 'request_output', None)
            images = getattr(req_out, 'images', None) if req_out else None
        if images:
            for i, img in enumerate(images):
                import numpy as np
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                path = f'/tmp/cosmos3_quant_omni_{i}.png'
                img.save(path)
                print(f'Saved to {path} ({img.size})')
        else:
            print(f'No images found in output. Keys: {dir(output)}')
            req_out = getattr(output, 'request_output', None)
            if req_out:
                print(f'  request_output keys: {[x for x in dir(req_out) if not x.startswith("_")]}')
