import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionLatentUpscalePipeline
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="/root/gpufree-data/checkpoints/hf_models/sd-x2-latent-upscaler",
        type=str,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Single image path or a directory of images.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/sr_backbone_smoke",
        type=str,
    )
    parser.add_argument(
        "--num_images",
        default=4,
        type=int,
        help="Only used when --input is a directory.",
    )
    parser.add_argument(
        "--num_inference_steps",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--guidance_scale",
        default=0.0,
        type=float,
        help="Use 0 to suppress text guidance during backbone validation.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    return parser.parse_args()


def collect_images(input_path, num_images):
    path = Path(input_path)
    if path.is_file():
        return [path]

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    images = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return images[:num_images]


def load_pipeline(model_id, device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if Path(model_id).exists():
        kwargs["local_files_only"] = True
    pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, **kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def main():
    args = parse_args()
    images = collect_images(args.input, args.num_images)
    if not images:
        raise FileNotFoundError(f"No images found under {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')}")
    print(f"device={device}")
    print(f"model_id={args.model_id}")

    pipe = load_pipeline(args.model_id, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for index, image_path in enumerate(images):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            result = pipe(
                prompt="",
                image=image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            out_image = result.images[0]

        out_name = f"{index:03d}_{image_path.stem}_x2.png"
        out_path = output_dir / out_name
        out_image.save(out_path)
        print(f"saved={out_path} input={image_path}")


if __name__ == "__main__":
    main()
