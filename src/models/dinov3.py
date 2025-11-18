# Based off of `dinov2.py` from Pulfer et al. and the facebookresearch/dinov3 repository.
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import Module

SUPPORTED_DINOV3_MODELS = [
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vitl16plus",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
]

DINOV3_WEIGHT_FILES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vitl16plus": "dinov3_vitl16plus_pretrain_lvd1689m-46503df0.pth",
    "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
}

BASE_DIR = Path(__file__).resolve().parents[2]
DINOV3_REPO_DIR = BASE_DIR / "dinov3"
WEIGHTS_DIR = BASE_DIR / "weights" / "dinov3"


def _load_dinov3_model(model_name: str):
    filename = DINOV3_WEIGHT_FILES.get(model_name)
    weight_file = WEIGHTS_DIR / filename
    if not weight_file.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {model_name}. Place {filename} inside {WEIGHTS_DIR}."
        )
    if not DINOV3_REPO_DIR.exists():
        raise FileNotFoundError(
            f"Local DINOv3 repository not found at {DINOV3_REPO_DIR}. Clone facebookresearch/dinov3 there."
        )
    weight_path = str(weight_file)
    return torch.hub.load(str(DINOV3_REPO_DIR), model_name, source="local", weights=weight_path)


def get_model(name, n_rtokens=10, enbable_robust=True):
    if name in SUPPORTED_DINOV3_MODELS:
        return DinoV3Robustifier(
            model_name=name,
            enable_robust=enbable_robust,
            n_rtokens=n_rtokens,
        )
    raise KeyError(f"Model {name} not supported. Pick one of {SUPPORTED_DINOV3_MODELS}")


class DinoV3Robustifier(Module):
    def __init__(self, model_name, n_rtokens=10, enable_robust=False):
        super(DinoV3Robustifier, self).__init__()

        assert (
            model_name in SUPPORTED_DINOV3_MODELS
        ), f"{model_name} not supported. Pick one of {SUPPORTED_DINOV3_MODELS}"

        self.model_name = model_name
        self.model = _load_dinov3_model(model_name).eval()
        self.enable_robust = enable_robust
        self.n_rtokens = max(0, n_rtokens)

        if self.n_rtokens > 0:
            hidden_dim = self.model.cls_token.shape[-1]
            self.rtokens = torch.nn.Parameter(
                1e-2 * torch.randn(1, n_rtokens, hidden_dim)
            )

    def get_trainable_parameters(self):
        if self.n_rtokens > 0:
            return [self.rtokens]
        return self.parameters()

    def store_rtokens(self, path=None):
        if self.n_rtokens > 0:
            if path is None:
                path = f"{self.model_name}_rtokens.pt"
            torch.save({"rtokens": self.rtokens}, path)

    def load_rtokens(self, path=None, device=None):
        if self.n_rtokens > 0:
            if path is None:
                path = f"{self.model_name}_rtokens.pt"
            self.rtokens = torch.load(path, map_location=device)["rtokens"]

    def forward(self, x, enable_robust=None, return_all=False, return_layers=None):
        running_robust = (
            enable_robust if enable_robust is not None else self.enable_robust
        )

        return self.dino_forward(x, running_robust, return_all, return_layers)

    def _prepare_tokens(self, x: Tensor):
        prepared, hw_tuple = self.model.prepare_tokens_with_masks(x)
        return prepared, hw_tuple

    def _apply_norms(self, x: Tensor, running_robust: bool):
        n_storage_tokens = getattr(self.model, "n_storage_tokens", 0)
        robt = self.n_rtokens if running_robust else 0
        total_without_rt = x[:, : (None if robt == 0 else -robt)]

        if getattr(self.model, "untie_cls_and_patch_norms", False) or getattr(
            self.model, "untie_global_and_local_cls_norm", False
        ):
            cls_norm_layer = getattr(self.model, "cls_norm", None)
            if cls_norm_layer is None:
                cls_norm_layer = self.model.norm
            x_norm_cls_reg = cls_norm_layer(total_without_rt[:, : n_storage_tokens + 1])
            patch_slice = total_without_rt[:, n_storage_tokens + 1 :]
            x_norm_patch = self.model.norm(patch_slice)
            x_norm_rtokens = (
                self.model.norm(x[:, -robt:]) if robt > 0 else None
            )
        else:
            x_norm = self.model.norm(x)
            x_norm_cls_reg = x_norm[:, : n_storage_tokens + 1]
            x_norm_patch = x_norm[:, n_storage_tokens + 1 : (None if robt == 0 else -robt)]
            x_norm_rtokens = x_norm[:, -robt:] if robt > 0 else None

        return x_norm_cls_reg, x_norm_patch, x_norm_rtokens

    def dino_forward(self, x, running_robust, return_all=False, return_layers=None):
        b = x.shape[0]
        x, (H, W) = self._prepare_tokens(x)

        # Appending robustness tokens
        if running_robust and self.n_rtokens > 0:
            x = torch.cat((x, self.rtokens.repeat(b, 1, 1)), dim=1)

        if return_layers is not None:
            activations: List[Tensor] = [x.clone().detach().cpu().double()]

        for idx, blk in enumerate(self.model.blocks):
            rope_sincos = self.model.rope_embed(H=H, W=W) if self.model.rope_embed is not None else None
            x = blk(x, rope_sincos)
            if return_layers is not None and idx in return_layers:
                activations.append(x.clone().detach().cpu().double())

        if return_layers is not None:
            return activations

        x_norm_cls_reg, x_norm_patch, x_norm_rtokens = self._apply_norms(x, running_robust)

        n_storage_tokens = getattr(self.model, "n_storage_tokens", 0)
        robt = self.n_rtokens if running_robust else 0

        cls_tokens = x_norm_cls_reg[:, 0]
        storage_tokens = (
            x_norm_cls_reg[:, 1 : n_storage_tokens + 1] if n_storage_tokens > 0 else None
        )
        patch_tokens = x_norm_patch

        if return_all:
            return {
                "x_norm_clstoken": cls_tokens,
                "x_norm_regtokens": storage_tokens,
                "x_norm_patchtokens": patch_tokens,
                "x_norm_rtokens": x_norm_rtokens if robt > 0 else None,
                "x_prenorm": x,
            }

        return torch.cat((cls_tokens.unsqueeze(1), patch_tokens), dim=1)
