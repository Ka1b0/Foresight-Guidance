import torch
import os
import yaml
import numpy as np
import argparse
import warnings
from diffusers import DDIMScheduler, DDIMInverseScheduler
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

warnings.filterwarnings('ignore')
import logging
logging.getLogger('diffusers').setLevel(logging.ERROR)


MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
PROMPTS = [
    'A fantasy portrait of a handsome girl with glowing blue hair and floating vivid butterflies. Ethereal aqua and pink light.',
    'A serene Chinese garden with cherry blossoms, koi pond, and traditional pagoda',
    'A sleeping baby tiger cub resting its head on a large cheeseburger with a golden bun and melted cheese. Warm, soft-focus background.'
]
 

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_config = config['CFG'] if 'CFG' in config else {}
    
    for method, method_params in config.items():
        merged_params = base_config.copy()
        merged_params.update(method_params)
        if method == 'Foresight-sampling' and 'foresight_scheduler' in merged_params:
            merged_params['foresight_scheduler'] = eval(merged_params['foresight_scheduler'])
        config[method] = merged_params
    return config


def setup_pipeline(device, dtype=torch.float16):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant='fp16',
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inv_scheduler = DDIMInverseScheduler.from_pretrained(MODEL_ID, subfolder='scheduler')
    return pipe.to(device)


def get_init_latents(seed, shape, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device=device)


def generate_image(pipe, prompt, method, params, shape, init_latent):
    kwargs = {
        'prompt': prompt,
        'shape': shape,
        'latents': init_latent,
        'method': method,
        **params
    }
    return pipe(**kwargs).images[0]


def main():
    parser = argparse.ArgumentParser(description="Diffusion Sampling Methods")
    parser.add_argument('--config_dir', type=str, default='configs/', help='YAML config file path')
    parser.add_argument('--nfe', type=int, default=50, help='NFE')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--save_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = load_config(f'{args.config_dir}/NFE-{args.nfe}.yaml')
    save_dir = f'{args.save_dir}/NFE-{args.nfe}'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}')
    dtype = torch.float16

    print(f"Loading model: {MODEL_ID}")
    pipe = setup_pipeline(device, dtype)

    for method, params in config.items():
        image_size = params.get('image_size', 1024)
        shape = (1, 4, image_size // 8, image_size // 8)

        print(f"\nMethod: {method}")
        print(f"Parameters:")
        for k, v in params.items():
            print(f"{k:<20}: {v}")

        for idx, prompt in enumerate(PROMPTS):
            init_latent = get_init_latents(args.seed + idx, shape, device, dtype)
            print(f"[{idx+1}/{len(PROMPTS)}] Generating: {prompt}")
            image = generate_image(pipe, prompt, method, params, shape, init_latent)
            filename = f"{idx}_{method}.png"
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
            print(f"Saved: {save_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
