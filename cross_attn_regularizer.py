from __future__ import annotations

import re
import types

import torch


class CrossAttentionRegularizer:
    def __init__(self, named_modules, token_indices, patch_method: str = "forward"):
        self.patch_method = patch_method
        self.named_modules = list(named_modules)
        self.layer_names = [name for name, _ in self.named_modules]
        self.token_indices = [int(idx) for idx in token_indices if int(idx) >= 0]
        self.loss = None
        self.active = False
        self.call_counts: dict[str, int] = {name: 0 for name in self.layer_names}
        self._original_methods = []
        self._install()

    @staticmethod
    def _pick_from_group(items, picks):
        chosen = []
        for idx in picks:
            if not items:
                continue
            idx = max(0, min(len(items) - 1, idx))
            value = items[idx]
            if value not in chosen:
                chosen.append(value)
        return chosen

    @classmethod
    def from_unet(cls, unet, token_indices, layers: str = "", max_layers: int = 5):
        attn_modules = [(name, module) for name, module in unet.named_modules() if name.endswith("attn2")]
        selected = cls._resolve_unet_layers(attn_modules, layers=layers, max_layers=max_layers)
        return cls(selected, token_indices=token_indices, patch_method="forward")

    @classmethod
    def from_pixart(cls, transformer, token_indices, layers: str = ""):
        attn_modules = [(name, module) for name, module in transformer.named_modules() if name.endswith("attn2")]
        selected = cls._resolve_pixart_layers(attn_modules, layers=layers)
        return cls(selected, token_indices=token_indices, patch_method="forward")

    @classmethod
    def _resolve_unet_layers(cls, attn_modules, layers: str = "", max_layers: int = 5):
        if layers:
            requested = [item.strip() for item in layers.split(",") if item.strip()]
            selected = []
            for req in requested:
                for name, module in attn_modules:
                    if name == req:
                        selected.append((name, module))
                        break
            if selected:
                return selected[:max_layers]

        groups = {"down": [], "mid": [], "up": []}
        for name, module in attn_modules:
            if name.startswith("down_blocks."):
                groups["down"].append((name, module))
            elif name.startswith("mid_block."):
                groups["mid"].append((name, module))
            elif name.startswith("up_blocks."):
                groups["up"].append((name, module))

        selected = []
        selected.extend(cls._pick_from_group(groups["down"], [0, len(groups["down"]) // 2]))
        selected.extend(cls._pick_from_group(groups["mid"], [len(groups["mid"]) // 2]))
        selected.extend(cls._pick_from_group(groups["up"], [len(groups["up"]) // 2, len(groups["up"]) - 1]))

        dedup = []
        seen = set()
        for item in selected:
            if item[0] not in seen:
                dedup.append(item)
                seen.add(item[0])
        if dedup:
            return dedup[:max_layers]
        return attn_modules[:max_layers]

    @classmethod
    def _resolve_pixart_layers(cls, attn_modules, layers: str = ""):
        block_map = {}
        for name, module in attn_modules:
            match = re.search(r"transformer_blocks\.(\d+)\.attn2$", name)
            if match:
                block_map[int(match.group(1))] = (name, module)
        if layers:
            requested = []
            for item in layers.split(","):
                item = item.strip()
                if not item:
                    continue
                if item.isdigit():
                    requested.append(int(item))
                else:
                    for name, module in attn_modules:
                        if name == item:
                            requested.append(name)
                            break
            selected = []
            for item in requested:
                if isinstance(item, int) and item in block_map:
                    selected.append(block_map[item])
                elif isinstance(item, str):
                    for name, module in attn_modules:
                        if name == item:
                            selected.append((name, module))
                            break
            if selected:
                return selected
        defaults = [0, 7, 14, 21, 27]
        selected = [block_map[idx] for idx in defaults if idx in block_map]
        return selected or attn_modules[:5]

    def _install(self):
        for name, module in self.named_modules:
            original = module.forward

            def wrapped(
                module_self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                *args,
                _name=name,
                **kwargs,
            ):
                residual = hidden_states
                input_ndim = hidden_states.ndim
                if module_self.spatial_norm is not None:
                    hidden_states = module_self.spatial_norm(hidden_states, temb)

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                else:
                    batch_size, _, _ = hidden_states.shape
                    channel = height = width = None

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif module_self.norm_cross:
                    encoder_hidden_states = module_self.norm_encoder_hidden_states(encoder_hidden_states)

                if module_self.group_norm is not None:
                    hidden_states = module_self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

                query = module_self.to_q(hidden_states)
                key = module_self.to_k(encoder_hidden_states)
                value = module_self.to_v(encoder_hidden_states)

                query = module_self.head_to_batch_dim(query)
                key = module_self.head_to_batch_dim(key)
                value = module_self.head_to_batch_dim(value)

                attn_scores = torch.einsum("b i d, b j d -> b i j", query, key) * module_self.scale
                attn_probs = attn_scores.softmax(dim=-1)
                self._record(_name, attn_probs)

                hidden_states = torch.einsum("b i j, b j d -> b i d", attn_probs, value)
                hidden_states = module_self.batch_to_head_dim(hidden_states)
                hidden_states = module_self.to_out[0](hidden_states)
                hidden_states = module_self.to_out[1](hidden_states)

                if input_ndim == 4 and channel is not None:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if getattr(module_self, "residual_connection", False):
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / module_self.rescale_output_factor
                return hidden_states

            module.forward = types.MethodType(wrapped, module)
            self._original_methods.append((module, "forward", original))

    def close(self):
        for module, attr, original in self._original_methods:
            setattr(module, attr, original)
        self._original_methods.clear()

    def begin(self):
        self.active = True
        self.loss = None
        for name in self.layer_names:
            self.call_counts[name] = 0

    def end(self, device):
        self.active = False
        if self.loss is None:
            return torch.tensor(0.0, device=device)
        return self.loss

    def _record(self, name: str, attn_probs: torch.Tensor):
        if not self.active:
            return
        branch_attn = self._extract_branch_attention(attn_probs)
        if branch_attn is None:
            return
        valid_indices = [idx for idx in self.token_indices if idx < branch_attn.shape[-1]]
        if not valid_indices:
            return
        token_attn = branch_attn[:, valid_indices].float()
        current = token_attn.var(unbiased=False)
        self.loss = current if self.loss is None else self.loss + current
        self.call_counts[name] += 1

    @staticmethod
    def _extract_branch_attention(attn_probs: torch.Tensor):
        if attn_probs.ndim != 3:
            return None
        total = attn_probs.shape[0]
        for batch_size, attack_ids in ((4, [1, 3]), (2, [1]), (1, [0])):
            if total % batch_size == 0:
                heads = total // batch_size
                attn = attn_probs.reshape(batch_size, heads, attn_probs.shape[1], attn_probs.shape[2])
                attn = attn.mean(dim=1)
                valid_ids = [idx for idx in attack_ids if idx < attn.shape[0]]
                if not valid_ids:
                    valid_ids = [attn.shape[0] - 1]
                return attn[valid_ids].mean(dim=0).float()
        return attn_probs.float().mean(dim=0)
