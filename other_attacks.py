import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import network
import timm

import warnings

from efficientnet_pytorch import EfficientNet
import re
from collections import OrderedDict
from timm.models.swin_transformer import SwinTransformer
warnings.filterwarnings("ignore")

class OURS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, image_input):
        # if isTrain:
        #     logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
        #     return None, logits_per_image
        # else:
        output = self.model(image_input)
        zero_output = torch.zeros_like(output)
        # return torch.cat([torch.tensor([0.]).to(image_input.device), output[0]]).unsqueeze(0).to(image_input.device)
        # return torch.cat([torch.tensor([0.]).to(image_input.device), output[0]]).unsqueeze(0).to(image_input.device)
        return torch.cat((zero_output, output), dim=1).to(image_input.device)
        # return torch.concatenate([torch.tensor([0.]).to(image_input.device), output[0]]).unsqueeze(0).to(image_input.device)
        # return torch.concatenate([output[0], torch.tensor([0.]).to(image_input.device)]).unsqueeze(0).to(image_input.device)
        # return torch.tensor([output[0], 0.]).to(image_input.device)


def model_selection(name, device=None):
    dic = {
        "R": "/root/gpufree-data/checkpoints/resnet50.pth",
        "E": "/root/gpufree-data/checkpoints/efficientnet-b0.pth",
        "D": "/root/gpufree-data/checkpoints/deit.pth",
        "S": "/root/gpufree-data/checkpoints/swin-t.pth",
    }
    if name == "R":
        model_path = dic[name]
        model = network.resnet50(num_classes=1)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model = OURS(model)
    elif name == "E":
        model_path = dic[name]
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=1, image_size=None)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        model = OURS(model)
    elif name == "D":
        model_path = dic[name]
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=False
        )
        model.head = torch.nn.Linear(in_features=768, out_features=1, bias=True)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict["model"])
        model = OURS(model)
    elif name == "S":
        model_path = dic[name]
        print(f"[S] load ckpt: {model_path}")
        ckpt = torch.load(model_path, map_location="cpu")
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        use_ape = "absolute_pos_embed" in sd
        print(f"[S] APE: {use_ape}")

        # Build a Swin-Base classifier (embed_dim=128, depths=(2,2,18,2), heads=(4,8,16,32)).
        # The checkpoint key format matches older timm naming (head.weight, layers.0.downsample...).
        model = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=1,
            ape=use_ape,
        )

        # Remap checkpoint keys to the current timm naming where needed.
        remapped = OrderedDict()
        for k, v in sd.items():
            k2 = k
            if k == "head.weight":
                k2 = "head.fc.weight"
            elif k == "head.bias":
                k2 = "head.fc.bias"
            else:
                m = re.match(r"^layers\.(\d+)\.downsample\.(.+)$", k)
                if m:
                    # In current timm, downsample modules are keyed under the *next* layer index.
                    k2 = f"layers.{int(m.group(1)) + 1}.downsample.{m.group(2)}"
            remapped[k2] = v

        # Load weights and wrap to produce 2 logits ([0, z]) like other detectors.
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if len(missing) or len(unexpected):
            print(f"[S] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        model = OURS(model)
    else:
        raise NotImplementedError("No such model!")
    if device is None:
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
    return model.to(device)
