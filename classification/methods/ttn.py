import torch
import torch.nn as nn
from methods.source import Source
from utils.registry import ADAPTATION_REGISTRY


class TTNLayer(nn.Module):
    """Test-Time Normalization layer with adaptive mixing."""

    def __init__(self, num_features, alpha=0.5, eps=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(num_features), requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def compute_alpha(self, batch_size, alpha_min=0.1):
        c = 16 * (1.0 - self.alpha)
        alpha_B = 1.0 - c / torch.sqrt(torch.tensor(float(batch_size)))
        return torch.clamp(alpha_B, min=alpha_min, max=1.0)

    def forward(self, x):
        mu_t = x.mean(dim=(0, 2, 3))
        var_t = x.var(dim=(0, 2, 3), unbiased=False)
        mu_s = self.running_mean
        var_s = self.running_var
        alpha = self.compute_alpha(x.shape[0])
        mu = alpha * mu_t + (1 - alpha) * mu_s
        var = alpha * var_t + (1 - alpha) * var_s + alpha * (1 - alpha) * (mu_t - mu_s) ** 2
        x_hat = (x - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        return self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]

    @staticmethod
    def _find_bns(module, alpha):
        replacements = []
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                new_layer = TTNLayer(child.num_features, alpha, child.eps)
                new_layer.running_mean = child.running_mean.clone()
                new_layer.running_var = child.running_var.clone()
                new_layer.gamma.data = child.weight.data.clone()
                new_layer.beta.data = child.bias.data.clone()
                replacements.append((module, name, new_layer))
            else:
                replacements.extend(TTNLayer._find_bns(child, alpha))
        return replacements

    @staticmethod
    def adapt_model(model, alpha=0.5):
        mods = TTNLayer._find_bns(model, alpha)
        print(f"| Found {len(mods)} modules to be replaced.")
        for parent, name, repl in mods:
            setattr(parent, name, repl)
        return model


@ADAPTATION_REGISTRY.register()
class TTN(Source):
    """Test-Time Normalization for classification."""

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = TTNLayer.adapt_model(self.model, alpha=self.cfg.BN.ALPHA).to(self.device)

