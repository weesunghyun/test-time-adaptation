import torch
import torch.nn as nn


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight Through Estimator for the backward pass."""
    return (x.round() - x).detach() + x


def _fake_quantize(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Quantize with STE-based fake quantization."""
    if tensor.numel() == 0:
        return tensor
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    max_val = tensor.abs().max()
    scale = max_val / qmax if max_val != 0 else 1.0
    return torch.clamp(_ste_round(tensor / scale), qmin, qmax) * scale


def _quantize_module_params(module: nn.Module, bits: int) -> None:
    """Quantize parameters of the given module in-place."""
    for name, param in module.named_parameters(recurse=False):
        if param is not None:
            param.data = _fake_quantize(param.data, bits)


def _activation_pre_hook(module: nn.Module, inputs, bits: int):
    return (_fake_quantize(inputs[0], bits),)


def _activation_post_hook(module: nn.Module, inputs, output, bits: int):
    return _fake_quantize(output, bits)


def apply_quantization(model: nn.Module, weight_bits: int = 8, act_bits: int = 8) -> nn.Module:
    """Apply fake quantization with STE to weights and activations of conv/linear layers."""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            _quantize_module_params(module, weight_bits)
            module.register_forward_pre_hook(lambda m, inp, b=act_bits: _activation_pre_hook(m, inp, b))
            module.register_forward_hook(lambda m, inp, out, b=act_bits: _activation_post_hook(m, inp, out, b))
    return model
