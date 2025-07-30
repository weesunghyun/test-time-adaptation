import torch
import torch.nn as nn
from copy import deepcopy

from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY


class LeanTTALayer(nn.Module):
    """Normalization layer updating statistics per sample."""

    def __init__(self, layer: nn.modules.batchnorm._BatchNorm, tau: float = 0.9, lam: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.tau = tau
        self.lam = lam
        self.eps = eps
        self.gamma = nn.Parameter(layer.weight.data.clone())
        self.beta = nn.Parameter(layer.bias.data.clone())
        self.register_buffer("mu_s", layer.running_mean.detach().clone())
        self.register_buffer("var_s", layer.running_var.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("Expected input with at least 2 dims")
        dims = (0,) + tuple(range(2, x.dim()))
        mu_t = x.mean(dim=dims)
        var_t = x.var(dim=dims, unbiased=False)

        mu_b = self.tau * self.mu_s + (1 - self.tau) * mu_t
        var_b = self.tau * self.var_s + (1 - self.tau) * var_t

        diff = mu_b - self.mu_s
        mahal = (diff * diff / (self.var_s + self.eps)).sum()
        d = 1 - torch.exp(-mahal)
        scale = d * self.lam

        mu_new = scale * self.mu_s + (1 - scale) * mu_b
        var_new = scale * self.var_s + (1 - scale) * var_b

        shape = [1, -1] + [1] * (x.dim() - 2)
        mu_new = mu_new.view(*shape)
        var_new = var_new.view(*shape)
        gamma = self.gamma.view(*shape)
        beta = self.beta.view(*shape)

        x_hat = (x - mu_new) / torch.sqrt(var_new + self.eps)
        return gamma * x_hat + beta

    @staticmethod
    def adapt_model(model: nn.Module, tau: float = 0.9, lam: float = 0.9, eps: float = 1e-5) -> nn.Module:
        for name, module in model.named_children():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                new_layer = LeanTTALayer(module, tau=tau, lam=lam, eps=eps)
                setattr(model, name, new_layer)
            else:
                LeanTTALayer.adapt_model(module, tau=tau, lam=lam, eps=eps)
        return model


@ADAPTATION_REGISTRY.register()
class LeanTTA(TTAMethod):
    """Backpropagation-free test-time adaptation using LeanTTA."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.tau = cfg.LEANTTA.TAU
        self.lam = cfg.LEANTTA.LAMBDA

    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        return self.model(imgs_test)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = LeanTTALayer.adapt_model(self.model, tau=self.tau, lam=self.lam).to(self.device)

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        optimizer_state = None
        return model_states, optimizer_state

    def reset(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
