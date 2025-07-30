"""On-demand Test-time Adaptation (OD-TTA).
Implementation based on methodology description.
"""
import os
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn

from methods.base import TTAMethod, forward_decorator
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


@ADAPTATION_REGISTRY.register()
class ODTTA(TTAMethod):
    """OD-TTA adapts the model when a domain shift is detected."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.momentum = cfg.ODTTA.MOMENTUM
        self.threshold = cfg.ODTTA.THRESHOLD
        self.buffer_size = cfg.ODTTA.BUFFER_SIZE
        self.candidate_path = cfg.ODTTA.CANDIDATE_POOL

        self.softmax_entropy = Entropy()
        self.ema_entropy = None
        self.baseline_entropy = None
        self.buffer: List[torch.Tensor] = []
        self.candidates = self.load_candidates(self.candidate_path)
        self.bn_feature_layer = self.get_second_bn()

    # ---------------------------------------------------------------------
    def load_candidates(self, path: str):
        if path and os.path.exists(path):
            return torch.load(path, map_location="cpu")
        else:
            # at least contain the source model
            return [{"state": deepcopy(self.model.state_dict()), "feature": None}]

    def get_second_bn(self):
        count = 0
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                count += 1
                if count == 2:
                    return m
        return None

    @torch.no_grad()
    def compute_entropy(self, outputs):
        ent = self.softmax_entropy(outputs).mean().item()
        if self.ema_entropy is None:
            self.ema_entropy = ent
        else:
            self.ema_entropy = self.momentum * self.ema_entropy + (1 - self.momentum) * ent
        if self.baseline_entropy is None:
            self.baseline_entropy = self.ema_entropy
        return ent

    def shift_detected(self):
        if self.baseline_entropy is None or self.ema_entropy is None:
            return False
        return (self.ema_entropy - self.baseline_entropy) > self.threshold

    def update_baseline(self):
        self.baseline_entropy = self.ema_entropy

    # ------------------------------------------------------------------
    def extract_domain_feature(self, data_list: List[torch.Tensor]):
        features = []
        handles = []

        def hook(_module, _in, out):
            feat = out.mean(dim=[0] + list(range(2, out.dim())))
            features.append(feat.detach().cpu())

        if self.bn_feature_layer is not None:
            handles.append(self.bn_feature_layer.register_forward_hook(hook))
        self.model.eval()
        with torch.no_grad():
            for x in data_list:
                self.model(x)
        for h in handles:
            h.remove()
        if len(features) == 0:
            return None
        feat = torch.stack(features).mean(dim=0)
        return feat

    def select_candidate(self, feature):
        if feature is None:
            return self.candidates[0]
        distances = []
        for cand in self.candidates:
            if cand.get("feature") is None:
                distances.append(float("inf"))
            else:
                distances.append(torch.norm(feature - cand["feature"], p=2))
        idx = int(torch.tensor(distances).argmin()) if len(distances) > 0 else 0
        return self.candidates[idx]

    def update_bn_statistics(self, data_list: List[torch.Tensor]):
        k = max(len(data_list), 1)
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.momentum = 1.0 / k
        self.model.train()
        with torch.no_grad():
            for x in data_list:
                self.model(x)
        self.model.eval()

    def finetune_bn_params(self, data_list: List[torch.Tensor], anchor_model):
        self.model.train()
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.requires_grad_(True)
        for x in data_list:
            outputs = self.model(x)
            loss = self.softmax_entropy(outputs).mean()
            with torch.no_grad():
                anchor_out = anchor_model(x)
            anchor_out = anchor_out.detach()
            contrastive = -((outputs - anchor_out) / (torch.norm(outputs - anchor_out) + 1e-6) * outputs).sum()
            loss = loss + 0.05 * contrastive
            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.model.eval()

    # ------------------------------------------------------------------
    @forward_decorator
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        self.buffer.append(imgs_test)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        outputs = self.model(imgs_test)
        self.compute_entropy(outputs)

        if self.shift_detected() and len(self.buffer) > 0:
            feature = self.extract_domain_feature(self.buffer)
            candidate = self.select_candidate(feature)
            anchor_model = self.copy_model(self.model)
            self.model.load_state_dict(candidate["state"], strict=False)
            self.update_bn_statistics(self.buffer)
            self.finetune_bn_params(self.buffer, anchor_model)
            self.update_baseline()
            self.buffer.clear()
        return outputs

    # configuration -----------------------------------------------------
    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = True
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

