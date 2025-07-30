"""Test-Time Normalization (TTN).
This implementation follows the basic description from the paper
and replaces all BatchNorm2d layers with a TTNLayer.
"""

import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from datasets.data_loading import get_source_loader
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


class TTNLayer(nn.Module):
    """Batch norm layer that interpolates between source and test statistics."""

    def __init__(self, num_features: int, alpha: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def compute_alpha(self, batch_size: int, alpha_min: float = 0.1) -> torch.Tensor:
        c = 16 * (1 - self.alpha)
        value = 1 - (c / torch.sqrt(torch.tensor(float(batch_size))))
        return torch.clamp(value, min=alpha_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        mu_t = x.mean(dim=(0, 2, 3))
        var_t = x.var(dim=(0, 2, 3), unbiased=False)

        mu_s = self.running_mean
        var_s = self.running_var

        alpha = self.compute_alpha(x.size(0))
        mu = alpha * mu_t + (1 - alpha) * mu_s
        var = alpha * var_t + (1 - alpha) * var_s + alpha * (1 - alpha) * (mu_t - mu_s) ** 2

        x_hat = (x - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

    @staticmethod
    def adapt_model(model: nn.Module, alpha: float = 0.5) -> nn.Module:
        """Recursively replace all BatchNorm2d layers with TTNLayer."""
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                new_layer = TTNLayer(module.num_features, alpha)
                new_layer.running_mean = module.running_mean.clone()
                new_layer.running_var = module.running_var.clone()
                new_layer.gamma.data = module.weight.data.clone()
                new_layer.beta.data = module.bias.data.clone()
                setattr(model, name, new_layer)
            else:
                TTNLayer.adapt_model(module, alpha)
        return model


@ADAPTATION_REGISTRY.register()
class TTN(TTAMethod):
    """Test-Time Normalization as a test-time adaptation method."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = TTNLayer.adapt_model(self.model).to(self.device)

        # optional post-training using source data
        train_loader = get_source_loader(
            dataset_name=self.cfg.CORRUPTION.DATASET,
            adaptation=self.cfg.MODEL.ADAPTATION,
            preprocess=self.model.model_preprocess,
            data_root_dir=self.cfg.DATA_DIR,
            batch_size=self.cfg.TEST.BATCH_SIZE,
            use_clip=self.cfg.MODEL.USE_CLIP,
            train_split=False,
            ckpt_path=self.cfg.MODEL.CKPT_PATH,
            workers=min(self.cfg.TEST.NUM_WORKERS, os.cpu_count()),
        )[1]
        device = next(self.model.parameters()).device
        self._post_train(self.model, train_loader, device)

        for module in self.model.modules():
            if isinstance(module, TTNLayer):
                module.alpha.requires_grad = False

    def _post_train(self, model: nn.Module, loader, device):
        model.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad and p.ndim == 0], lr=1e-3)
        mse = nn.MSELoss()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_aug = self._augment(x)
            out_clean = model(x)
            out_aug = model(x_aug)
            loss = F.cross_entropy(out_clean, y) + mse(out_clean, out_aug)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        aug = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        )
        return torch.stack([aug(img) for img in x])

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def forward_and_adapt(self, x):
        imgs_test = x[0]
        return self.model(imgs_test)
