import os
from loggers import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["http_proxy"] = "http://10.24.116.74:7890"
os.environ["https_proxy"] = "http://10.24.116.74:7890"
# os.environ["https_proxy"] = "https://127.0.0.1:7890"
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import diff_latent_attack
from PIL import Image
import numpy as np
import os
import glob

import random

from natsort import ns, natsorted
import argparse
import other_attacks
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir',
                    default="/root/gpufree-data/dataset/genimage_adversial",
                    type=str, help='Where to save the adversarial examples, and other results')
parser.add_argument('--mode',
                    default="double",
                    type=str, help='')
parser.add_argument('--images_root',
                    default="/root/gpufree-data/dataset/test/genimage/fake",
                    type=str, help='The clean images root directory')
parser.add_argument('--pretrained_diffusion_path',
                    default="stabilityai/stable-diffusion-2-1-base",
                    type=str,
                    help='Change the path to `stabilityai/stable-diffusion-2-base` if want to use the pretrained model')
parser.add_argument('--diffusion_steps', default=20, type=int, help='Total DDIM sampling steps')
parser.add_argument('--start_step',
                    default=18,
                    type=int, help='Which DDIM step to start the attack')
parser.add_argument('--iterations',
                    default=10,
                    type=int, help='Iterations of optimizing the adv_image')
parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--model_name',
                    default="E,R,D,S",
                    type=str, help='The surrogate model from which the adversarial examples are crafted')
parser.add_argument('--dataset_name',
                    default="ours_try",
                    type=str,
                    choices=["ours_try"],
                    help='The dataset name for generating adversarial examples')
parser.add_argument('--is_encoder', default=1, type=int)
parser.add_argument('--encoder_weights',
                    default="/root/gpufree-data/checkpoints/Controlvae.pt",
                    type=str)

parser.add_argument('--guidance', default=0., type=float, help='guidance scale of diffusion models')
parser.add_argument('--eps', default=4 / 255, type=float, help='guidance scale of diffusion models')
parser.add_argument('--attack_loss_weight', default=10, type=int, help='attack loss weight factor')
parser.add_argument('--pgd_steps', default=10, type=int, help='PGD steps for initializing perturbations')
parser.add_argument('--pgd_alpha', default=None, type=float, help='PGD step size (default: eps/pgd_steps)')
parser.add_argument('--lambda_l1', default=1.0, type=float, help='L1 loss coefficient (paper: alpha)')
parser.add_argument('--lambda_lpips', default=1.0, type=float, help='LPIPS loss coefficient (paper: beta)')
parser.add_argument('--lambda_latent', default=0.0, type=float, help='Latent L1 regularization coefficient (not in paper; default 0)')
parser.add_argument('--attack_only_fake', default=1, type=int, help='If 1, only craft adversarial examples for fake images')
parser.add_argument('--max_per_source', default=0, type=int, help='If >0, sample up to N fake images per generator/source (paper: 100)')
parser.add_argument('--seed', default=42, type=int, help='Random seed used for sampling')

def seed_torch(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(42)


def run_diffusion_attack(image, label, diffusion_model, diffusion_steps, guidance=2.5, save_dir="", res=224,
                         model_name="inception", start_step=15, iterations=30, classes=None, logger=None, args=None):
    adv_image = None


    for i in range(1):
        if adv_image is None:
            image = image.resize((res, res), resample=Image.LANCZOS)
            adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnr, ssim = diff_latent_attack.diffattack(diffusion_model, label,
                                                                          num_inference_steps=diffusion_steps,
                                                                          guidance_scale=guidance,
                                                                          image=image, compare=image,
                                                                          save_path=save_dir, res=res, model_name=model_name,
                                                                          start_step=start_step, classes=classes,
                                                                          iterations=iterations, logger=logger, args=args, idx=i)
        else:
            # image = adv_image
            adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnr, ssim = diff_latent_attack.diffattack(
                diffusion_model, label, None,
                num_inference_steps=diffusion_steps,
                guidance_scale=guidance,
                image=adv_image, compare=image,
                save_path=save_dir, res=res, model_name=model_name,
                start_step=start_step, classes=classes,
                iterations=iterations, logger=logger, args=args, idx=i)

    adv_image = np.array(adv_image)
    return adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnr, ssim


if __name__ == "__main__":
    args = parser.parse_args()
    guidance = args.guidance
    diffusion_steps = args.diffusion_steps  # Total DDIM sampling steps.
    start_step = args.start_step  # Which DDIM step to start the attack.
    iterations = args.iterations  # Iterations of optimizing the adv_image.
    res = args.res  # Input image resized resolution.
    model_name = args.model_name.split(",")  # The surrogate model from which the adversarial examples are crafted.

    if args.dataset_name == "ours_try":
        assert model_name in [["E", "R", "D", "S"], ["R", "E", "D", "S"], ["D", "E", "R", "S"], ["S", "E", "R", "D"]], f"There is no pretrained weight of {model_name} for RealFake dataset."


    name = ""
    for model in model_name:
        name += model
    name += str(args.is_encoder)
    save_dir = args.save_dir  # Where to save the adversarial examples, and other results.
    save_dir = save_dir.format(name)
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(name='train', log_path='{}/{}.log'.format(save_dir, name))

    images_root = args.images_root  # The clean images' root directory.


    logger.info(f"\n******Attack based on Diffusion, Attacked Dataset: {args.dataset_name}*********")

    # Change the path to "stabilityai/stable-diffusion-2-base" if you want to use the pretrained model.
    pretrained_diffusion_path = args.pretrained_diffusion_path

    ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_diffusion_path).to('cuda')
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)


    dic_ = {"E":"efficientnet-b0",
            "R":"resnet50",
            "D":"deit",
            "S":"swin-t",
            }

    # Collect image files. Datasets may store images under `fake/` and `real/` subfolders.
    # We support:
    # - pointing `--images_root` at a single class folder (labels inferred as constant)
    # - pointing `--images_root` at a parent folder containing both classes (labels inferred per-path)
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    root = Path(images_root)
    if root.is_file():
        all_images = [str(root)]
    else:
        # Recurse so `--images_root .../fake` works when images are stored under
        # `fake/<subset>/*.png` (GenImage-style layout).
        all_images = [
            str(p)
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in img_exts
        ]
    all_images = natsorted(all_images, alg=ns.PATH)
    if len(all_images) == 0:
        subdirs = []
        if root.is_dir():
            subdirs = [p.name for p in root.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"No image files found under images_root={images_root!r}. "
            "Point --images_root at a folder containing images (or a single image file). "
            + (f"Subdirectories found: {subdirs}. " if subdirs else "")
            + "If your dataset is split into subfolders, pass e.g. "
            "`--images_root /path/to/genimage/fake`."
        )

    # Infer labels from path components.
    # Convention used in this repo: `fake` -> 0 (ai), `real` -> 1 (nature).
    def _infer_label(p: str) -> int:
        parts = [x.lower() for x in Path(p).parts]
        # Accept folders like `fake/`, `fake_100x6/`, `fake_by_subset/` for convenience.
        has_fake = any(x == "fake" or x.startswith("fake") for x in parts)
        has_real = any(x == "real" or x.startswith("real") for x in parts)
        if has_fake and not has_real:
            return 0
        if has_real and not has_fake:
            return 1
        raise ValueError(
            f"Can't infer label from path {p!r}. Expected one of path components to be 'fake' or 'real'."
        )

    label = np.array([_infer_label(p) for p in all_images], dtype=np.int64)

    def _infer_source(p: str) -> str:
        """
        Best-effort generator/source inference for GenImage fake images.
        If filename contains one of known substrings, use it; otherwise fall back to a token.
        """
        name = Path(p).name.lower()
        # Keep this list small but practical for GenImage naming patterns.
        known = [
            "biggan",
            "gaugan",
            "glide",
            "ldm",
            "sdv4",
            "wukong",
            "midjourney",
            "adm",
            "vqdm",
        ]
        for k in known:
            if k in name:
                return k
        parts = name.split("_")
        return parts[1] if len(parts) >= 2 else "unknown"

    # Match paper protocol: craft adversarial examples for fake images (ai) and evaluate transfer to others.
    if args.attack_only_fake:
        keep = [i for i, y in enumerate(label) if int(y) == 0]
        all_images = [all_images[i] for i in keep]
        label = label[keep]

    # Optional: sample a fixed number of fake images per source/generator (paper: 100 per validation set).
    if args.max_per_source and args.max_per_source > 0:
        rng = random.Random(args.seed)
        groups = {}
        for p in all_images:
            groups.setdefault(_infer_source(p), []).append(p)
        sampled = []
        for k in sorted(groups.keys()):
            imgs = groups[k][:]
            rng.shuffle(imgs)
            sampled.extend(imgs[: args.max_per_source])
        all_images = natsorted(sampled, alg=ns.PATH)
        label = np.array([_infer_label(p) for p in all_images], dtype=np.int64)

    adv_images = []
    images = []
    adv_all_acc = 0
    adv_all_acc1 = 0
    adv_all_acc2 = 0
    adv_all_acc3 = 0

    # s = 0
    psnrss = []
    ssimss = []
    model_name1, model_name2, model_name3, model_name4 = model_name

    classifier = other_attacks.model_selection(model_name1).eval()
    classifier.requires_grad_(False)

    classifier_supp = other_attacks.model_selection(model_name2).eval()
    classifier_supp.requires_grad_(False)

    classifier_supp1 = other_attacks.model_selection(model_name3).eval()
    classifier_supp1.requires_grad_(False)

    classifier_supp2 = other_attacks.model_selection(model_name4).eval()
    classifier_supp2.requires_grad_(False)

    classes = [classifier, classifier_supp, classifier_supp1, classifier_supp2]
    for ind, image_path in enumerate(all_images):

        tmp_image = Image.open(image_path).convert('RGB')
        tmp_image.save(os.path.join(save_dir, str(ind).rjust(4, '0') + "_originImage.png"))

        adv_image, adv_acc, adv_acc1, adv_acc2, adv_acc3, psnrv, ssimv = run_diffusion_attack(tmp_image, label[ind:ind + 1],
                                                             ldm_stable,
                                                             diffusion_steps, guidance=guidance,
                                                             res=res, model_name=model_name,
                                                             classes=classes,
                                                             start_step=start_step,
                                                             iterations=iterations,
                                                             logger=logger,
                                                             save_dir=os.path.join(save_dir,
                                                                                   str(ind).rjust(4, '0')), args=args)

        adv_image = adv_image.astype(np.float32) / 255.0
        adv_images.append(adv_image[None].transpose(0, 3, 1, 2))

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        images.append(tmp_image)


        adv_all_acc += adv_acc
        adv_all_acc1 += adv_acc1
        adv_all_acc2 += adv_acc2
        adv_all_acc3 += adv_acc3
        psnrss.append(psnrv)
        ssimss.append(ssimv)
        logger.info('final PSNR: {:.2f} dB; final SSIM: {:.4f}.'.format(psnrv, ssimv))


    logger.info("Adv acc: {}%".format(adv_all_acc / len(all_images) * 100))
    logger.info("Adv acc1: {}%".format(adv_all_acc1 / len(all_images) * 100))
    logger.info("Adv acc2: {}%".format(adv_all_acc2 / len(all_images) * 100))
    logger.info("Adv acc3: {}%".format(adv_all_acc3 / len(all_images) * 100))
    logger.info('mean PSNR: {:.2f} dB; mean SSIM: {:.4f}.'.format(np.mean(psnrss), np.mean(ssimss)))
    logger.info('std PSNR: {:.2f} dB; std SSIM: {:.4f}.'.format(np.std(psnrss), np.std(ssimss)))
