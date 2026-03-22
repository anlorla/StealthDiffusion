import copy
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["all_proxy"] = "http://10.24.116.74:7890"
from tqdm import tqdm
from torch import nn, optim
import torch, argparse, math

# from diffusers.models.vae \
#     import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput
from diffusers.models.autoencoder_kl \
    import AutoencoderKL, Encoder, Decoder, AutoencoderKLOutput, DiagonalGaussianDistribution, DecoderOutput



torch.backends.cudnn.enabled = True

torch.backends.cudnn.benchmark = True
# from model.vq_model import VQModel
try:
    from lossers.lpips import LPIPS
except ModuleNotFoundError:
    from lpips import LPIPS
# from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms, utils
from loggers import Logger
from torch.utils.data.sampler import WeightedRandomSampler
from ControlVAE import NewEncoder, NewDecoder
import os
from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def binary_dataset(root, transform):
    dset = datasets.ImageFolder(root, transform)
    return dset

def get_dataset(paths, transform):
    dset_lst = []
    for path in paths:
        dset = binary_dataset(path, transform)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    # w = torch.tensor([0.7, 0.3])
    w = torch.tensor([0.0, 1.0])
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

def create_dataloader(args, paths, transform):
    shuffle = True
    dataset = get_dataset(paths, transform)
    # sampler = SubsetRandomSampler(list(range(len(dataset))))
    sampler = get_bal_sampler(dataset)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              # shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(args.num_threads))
    return data_loader

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


def build_dncnn_filter(device):
    n_channels = 3
    nb = 20
    filter_path = CHECKPOINT_ROOT / "dncnn_color_blind.pth"
    from models.network_dncnn import DnCNN as net

    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R').to(device)
    model.load_state_dict(torch.load(filter_path, map_location=device), strict=True)
    model = model.eval()
    model.requires_grad_(False)
    return model


class FFL(nn.Module):

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False, use_filter=False, use_single_filter=False):
        super(FFL, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.use_filter = use_filter
        self.use_single_filter = use_single_filter

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        y = torch.stack(patch_list, 1)

        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        if self.use_filter:
            n_channels = 3
            nb = 20
            filter_path = CHECKPOINT_ROOT / "dncnn_color_blind.pth"
            from models.network_dncnn import DnCNN as net
            filter = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R').cuda()
            filter.load_state_dict(torch.load(filter_path), strict=True)
            filter = filter.eval()
            if self.use_single_filter:
                with torch.no_grad():
                    pred = filter.noise(pred)
            else:
                pred = filter.noise(pred)
                target = filter.noise(target)
        
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight



class RealDataset(Dataset):
    def __init__(self, paths=None, files=None, res=224):
        self.paths = paths or []
        self.files = files or []
        self.data = self.get_data()
        self.res = res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        image = Image.open(path).convert("RGB")

        image = image.resize((self.res, self.res), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)[:, :3, :, :].squeeze(0).cuda()
        return 2.0 * image - 1.0

        return sample

    def get_data(self):
        if self.files:
            return list(self.files)

        data_list = []
        for path in self.paths:
            if os.path.isdir(path):
                for candidate in glob.glob(os.path.join(path, "**", "*"), recursive=True):
                    if os.path.isfile(candidate):
                        data_list.append(candidate)
            elif os.path.isfile(path):
                data_list.append(path)
        data_list = sorted(set(data_list))
        return data_list


class FakeDataset(Dataset):
    def __init__(self, paths, res=224):
        self.paths = paths
        self.data = self.get_data()
        self.res = res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        dataname = path.split("/")[4]
        imgname = path.split("/")[-1]


        image = Image.open(path).convert("RGB")

        image = image.resize((self.res, self.res), resample=Image.LANCZOS)

        image = np.array(image).astype(np.float32) / 255.0
        # image1 = np.array(image1).astype(np.float32) / 255.0

        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)[:, :3, :, :].squeeze(0).cuda()


        image = 2.0 * image - 1.0
        return image

    def get_data(self):
        data_list = []
        for path in self.paths:
            data_list = data_list + glob.glob(path + "/*")
        return data_list

if __name__ == '__main__':

        parser = argparse.ArgumentParser(description="Train VQModel")
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--name", type=str, default="1")
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument("--iter", type=int, default=2, help="total training iterations")
        parser.add_argument("--batch_size", type=int, default=32, help="batch sizes for each gpus")
        parser.add_argument("--size", type=int, default=224, help="image sizes for the model")
        parser.add_argument("--freq_log", type=int, default=20, help="")
        parser.add_argument("--freq_save", type=int, default=10000, help="")
        parser.add_argument("--cache_dir", type=str, default="./.cache")
        parser.add_argument("--arch", type=str,
                            default="resnet50")
        parser.add_argument("--flag", type=bool, default=True)
        parser.add_argument("--save_dir", type=str,
                            default = "checkpoints")
        parser.add_argument("--resume", type=str,
                            default = "")
        parser.add_argument("--noise_prototype", type=str,
                            default="/path/to/pretrain")
        parser.add_argument("--dir", type=str,
                            default="")
        parser.add_argument("--real_root", type=str, default="",
                            help="Root directory that contains training real images.")
        parser.add_argument("--manifest", type=str, default="",
                            help="Optional text/csv manifest listing one image path per line.")
        parser.add_argument("--pipeline_type", type=str, default="sd", choices=["sd", "sr"],
                            help="Load SD VAE or SR latent-upscaler VAE.")
        parser.add_argument("--pretrained_diffusion_path", type=str, default="stabilityai/stable-diffusion-2-1-base",
                            help="Diffusers pipeline path.")
        parser.add_argument("--log_path", type=str, default="",
                            help="Explicit log file path.")
        parser.add_argument("--output_ckpt", type=str, default="",
                            help="Explicit checkpoint path for the latest encoder weights.")
        parser.add_argument("--seed", type=int, default=8888)
        parser.add_argument("--lambda_recon_l1", type=float, default=1.0,
                            help="L1 reconstruction weight.")
        parser.add_argument("--lambda_recon_lpips", type=float, default=1.0,
                            help="LPIPS reconstruction weight.")
        parser.add_argument("--legacy_freq_weight", type=float, default=0.02,
                            help="Legacy FFL(x_batch, rec) weight from the current training script.")
        parser.add_argument("--enable_npl", type=int, default=0,
                            help="If 1, enable prototype-based NPL frequency loss.")
        parser.add_argument("--lambda_npl", type=float, default=10.0,
                            help="Weight for prototype-based NPL loss when enable_npl=1.")
        # parser.add_argument('-d', '--dir', nargs='+', type=str, default=[
        #     "/path/to/nature",
        # ])

        args = parser.parse_args()
        device = torch.device(args.device)
        if args.pipeline_type == "sr":
            ldm_stable = StableDiffusionLatentUpscalePipeline.from_pretrained(
                args.pretrained_diffusion_path
            ).to(device)
        else:
            ldm_stable = StableDiffusionPipeline.from_pretrained(
                args.pretrained_diffusion_path
            ).to(device)

        if args.log_path:
            log_path = args.log_path
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        else:
            if not os.path.exists("log"):
                os.mkdir("log")
            log_path = 'log/train{}.log'.format(args.name)

        logger = Logger(name='demofiles', log_path=log_path)
        try:
            lpips = LPIPS(net='vgg', cache_dir=args.cache_dir).cuda()
        except TypeError:
            lpips = LPIPS(net='vgg').cuda()

        def load_manifest(manifest_path):
            files = []
            with open(manifest_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if "," in line:
                        candidate = line.split(",")[-1].strip()
                    else:
                        candidate = line
                    if candidate.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                        files.append(candidate)
            return files

        manifest_files = load_manifest(args.manifest) if args.manifest else []
        if manifest_files:
            real_dataset = RealDataset(files=manifest_files, res=args.size)
        elif args.real_root:
            real_dataset = RealDataset(paths=[args.real_root], res=args.size)
        else:
            l = ['ADMnew/imagenet_ai_0508_adm/train/nature', 'BigGAN/imagenet_ai_0419_biggan/train/nature', 'glide/imagenet_glide/train/nature',
             'Midjourney/imagenet_midjourney/train/nature', 'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature',
             'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train/nature', 'VQDM/imagenet_ai_0419_vqdm/train/nature',
             'wukong/imagenet_ai_0424_wukong/train/nature']
            paths_real = [args.dir + "/" + item for item in l]
            real_dataset = RealDataset(paths_real, res=args.size)

        if len(real_dataset) == 0:
            raise RuntimeError("No training real images found for ControlVAE training.")

        noise_prototype_exists = bool(args.noise_prototype) and os.path.exists(args.noise_prototype)
        npl_enabled = bool(int(args.enable_npl))
        loss_terms = [
            f"{args.lambda_recon_lpips} * lpips",
            f"{args.lambda_recon_l1} * l1",
        ]
        if args.legacy_freq_weight != 0:
            loss_terms.append(f"{args.legacy_freq_weight} * ffl(x_batch, rec)")
        if npl_enabled:
            loss_terms.append(f"{args.lambda_npl} * npl(rec, noise_prototype)")
        logger.info(
            json.dumps(
                {
                    "name": args.name,
                    "pipeline_type": args.pipeline_type,
                    "pretrained_diffusion_path": args.pretrained_diffusion_path,
                    "real_root": args.real_root,
                    "manifest": args.manifest,
                    "dataset_size": len(real_dataset),
                    "resume": args.resume,
                    "init_mode": "resume_controlvae" if args.resume else "sr_vae_encoder_only",
                    "save_dir": args.save_dir,
                    "output_ckpt": args.output_ckpt,
                    "size": args.size,
                    "batch_size": args.batch_size,
                    "iter": args.iter,
                    "seed": args.seed,
                    "noise_prototype_path": args.noise_prototype,
                    "noise_prototype_exists_before_run": noise_prototype_exists,
                    "npl_status": "active" if npl_enabled else "inactive",
                    "actual_loss": " + ".join(loss_terms),
                    "lambda_recon_l1": args.lambda_recon_l1,
                    "lambda_recon_lpips": args.lambda_recon_lpips,
                    "legacy_freq_weight": args.legacy_freq_weight,
                    "enable_npl": int(npl_enabled),
                    "lambda_npl": args.lambda_npl,
                    "ffl_filter_defined": True,
                    "ffl_filter_used": bool(npl_enabled),
                },
                ensure_ascii=True,
            )
        )
        if not noise_prototype_exists:
            noise_dataset = real_dataset
            noise_prototype = torch.zeros_like(noise_dataset[0].unsqueeze(0))
            filter = build_dncnn_filter(device)
            for i in tqdm(range(len(noise_dataset))):
                real = noise_dataset[i].unsqueeze(0)
                with torch.no_grad():
                    real_noise = filter.noise(real)
                noise_prototype += real_noise
            noise_prototype /= len(noise_dataset)
            if args.noise_prototype:
                os.makedirs(os.path.dirname(args.noise_prototype), exist_ok=True)
                torch.save(noise_prototype, args.noise_prototype)
            logger.info(
                json.dumps(
                    {
                        "noise_prototype_generated": True,
                        "noise_prototype_path": args.noise_prototype,
                        "noise_prototype_num_images": len(noise_dataset),
                    },
                    ensure_ascii=True,
                )
            )
        else:
            noise_prototype = torch.load(args.noise_prototype)
            logger.info(
                json.dumps(
                    {
                        "noise_prototype_generated": False,
                        "noise_prototype_loaded": True,
                        "noise_prototype_path": args.noise_prototype,
                    },
                    ensure_ascii=True,
                )
            )
        noise_prototype = noise_prototype.to(device=device, dtype=torch.float32)
        logger.info(
            json.dumps(
                {
                    "noise_prototype_shape": list(noise_prototype.shape),
                    "noise_prototype_dtype": str(noise_prototype.dtype),
                    "noise_prototype_device": str(noise_prototype.device),
                },
                ensure_ascii=True,
            )
        )



        # fake_dataset = FakeDataset(paths)

        real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True)

        params = {"act_fn": "silu",
                  "block_out_channels": [
                      128,
                      256,
                      512,
                      512
                  ],
                  "down_block_types": [
                      "DownEncoderBlock2D",
                      "DownEncoderBlock2D",
                      "DownEncoderBlock2D",
                      "DownEncoderBlock2D"
                  ],
                  "in_channels": 3,
                  "latent_channels": 4,
                  "layers_per_block": 2,
                  "norm_num_groups": 32,
                  "out_channels": 3,
                  "sample_size": 512,
                  "up_block_types": [
                      "UpDecoderBlock2D",
                      "UpDecoderBlock2D",
                      "UpDecoderBlock2D",
                      "UpDecoderBlock2D"
                  ]
                  }

        params_encoder = {
            "in_channels": params["in_channels"],
            "out_channels": params["latent_channels"],
            "down_block_types": params["down_block_types"],
            "block_out_channels": params["block_out_channels"],
            "layers_per_block": params["layers_per_block"],
            "act_fn": params["act_fn"],
            "norm_num_groups": params["norm_num_groups"],
            "double_z": True,
        }
        params_decoder = {
            "in_channels": params["latent_channels"],
            "out_channels": params["out_channels"],
            "up_block_types": params["up_block_types"],
            "block_out_channels": params["block_out_channels"],
            "layers_per_block": params["layers_per_block"],
            "act_fn": params["act_fn"],
            "norm_num_groups": params["norm_num_groups"],
        }

        encoder = NewEncoder(**params_encoder)

        if args.resume == "":
            encoder.encoder.load_state_dict(ldm_stable.vae.encoder.state_dict(), strict=False)
        else:
            p = args.resume
            state_dict = torch.load(p)
            encoder.load_state_dict(state_dict["encoder"], strict=False)

        decoder = NewDecoder(**params_decoder)

        decoder.load_state_dict(ldm_stable.vae.decoder.state_dict(), strict=False)

        requires_grad(encoder, True)
        ldm_stable.vae.decoder = decoder
        requires_grad(decoder, False)
        ldm_stable.vae.requires_grad_(False)
        ldm_stable.text_encoder.requires_grad_(False)
        ldm_stable.unet.requires_grad_(False)

        encoder = encoder.cuda()
        decoder = decoder.cuda()



        loss_l1 = torch.nn.L1Loss()
        loss_adv = torch.nn.CrossEntropyLoss()
        optimizer = optim.AdamW(encoder.parameters(), lr=0.001)
        ffl = FFL(loss_weight=1.0, alpha=1.0, log_matrix=False)
        ffl_filter = FFL(loss_weight=1.0, alpha=1.0, log_matrix=False, ave_spectrum=True, use_filter=True, use_single_filter=True)
        residual_filter = build_dncnn_filter(device) if npl_enabled else None

        pbar = tqdm(range(args.iter))

        generator = torch.Generator().manual_seed(args.seed)

        for idx in pbar:
            for idy, batch in enumerate(tqdm(real_dataloader)):
                x_batch = batch
                x_batch = x_batch.cuda()
                gpu_generator = torch.Generator(device=x_batch.device)
                gpu_generator.manual_seed(generator.initial_seed())
                if args.flag:
                    down_features = encoder(x_batch)[1]
                    latents = 0.18215 * ldm_stable.vae.encode(x_batch).latent_dist.sample(generator=gpu_generator)

                else:
                    a = encoder(x_batch)[0]
                    down_features = None
                    down_features_y = None
                    moments = ldm_stable.vae.quant_conv(a)
                    posterior = DiagonalGaussianDistribution(moments)

                    latents = 0.18215 * AutoencoderKLOutput(latent_dist=posterior).latent_dist.sample(generator=gpu_generator)

                decode_latents = ldm_stable.vae.post_quant_conv(1 / 0.1825 * latents)
                rec = decoder(decode_latents, down_features)
                lpips_loss = lpips(x_batch, rec).mean()
                l1_loss = loss_l1(x_batch, rec)
                loss_ffl_filter = ffl(x_batch, rec)
                if npl_enabled:
                    prototype_batch = noise_prototype.expand(rec.shape[0], -1, -1, -1)
                    rec_noise = residual_filter.noise(rec)
                    loss_npl = ffl.loss_formulation(
                        ffl.tensor2freq(rec_noise).mean(0, keepdim=True),
                        ffl.tensor2freq(prototype_batch).mean(0, keepdim=True),
                    )
                else:
                    loss_npl = torch.zeros((), device=rec.device, dtype=rec.dtype)

                loss = (
                    args.lambda_recon_lpips * lpips_loss
                    + args.lambda_recon_l1 * l1_loss
                    + args.legacy_freq_weight * loss_ffl_filter
                    + args.lambda_npl * loss_npl
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idy % args.freq_log == 0:
                    logger.info(
                        "lpips_loss:{}  l1_loss:{} loss_ffl_filter:{} loss_npl:{} total_loss:{}".format(
                            lpips_loss, l1_loss, loss_ffl_filter, loss_npl, loss
                        )
                    )

                    with torch.no_grad():
                        x_batch = x_batch.cuda()

                        gpu_generator = torch.Generator(device=x_batch.device)
                        gpu_generator.manual_seed(generator.initial_seed())
                        if args.flag:
                            down_features = encoder(x_batch)[1]

                            latents = 0.18215 * ldm_stable.vae.encode(x_batch).latent_dist.sample(generator=gpu_generator)
                            decode_latents = ldm_stable.vae.post_quant_conv(1 / 0.1825 * latents)
                        else:
                            a = encoder(x_batch)[0]
                            down_features = None
                            moments = ldm_stable.vae.quant_conv(a)
                            posterior = DiagonalGaussianDistribution(moments)
                            latents = 0.18215 * AutoencoderKLOutput(latent_dist=posterior).latent_dist.sample(
                                generator=gpu_generator)

                        rec = decoder(decode_latents, down_features)

                        rec = (rec / 2 + 0.5).clamp(0, 1)
                        x_batch = (x_batch / 2 + 0.5).clamp(0, 1)

                        save_dir = args.save_dir
                        os.makedirs(save_dir, exist_ok=True)

                        save_dir_raw = os.path.join(save_dir, "sampleraw")
                        save_dir_recon = os.path.join(save_dir, "samplerecon")
                        os.makedirs(save_dir_raw, exist_ok=True)
                        os.makedirs(save_dir_recon, exist_ok=True)

                        utils.save_image(
                            x_batch, save_dir+"/"+f"sampleraw/{str(idy).zfill(6)}.png",
                            nrow=int(math.sqrt(args.batch_size)), normalize=False, value_range=(0, 1),
                        )
                        utils.save_image(
                            rec, save_dir+"/"+f"samplerecon/{str(idy).zfill(6)}.png",
                            nrow=int(math.sqrt(args.batch_size)), normalize=False, value_range=(0, 1),
                        )
                if idy % args.freq_save == 0:
                    save_dir = args.save_dir
                    os.makedirs(save_dir, exist_ok=True)
                    latest_state = {
                        'encoder': encoder.state_dict(),
                    }
                    torch.save(
                        latest_state,
                        "{}/{}_{}.pt".format(save_dir, idx, str(idy).zfill(6)),
                    )
                    if args.output_ckpt:
                        out_dir = os.path.dirname(args.output_ckpt)
                        if out_dir:
                            os.makedirs(out_dir, exist_ok=True)
                        torch.save(latest_state, args.output_ckpt)

        if args.output_ckpt:
            out_dir = os.path.dirname(args.output_ckpt)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            torch.save({'encoder': encoder.state_dict()}, args.output_ckpt)
