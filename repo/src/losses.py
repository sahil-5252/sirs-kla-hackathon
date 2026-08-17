import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import ssim


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class SSIMLoss(nn.Module):
    def forward(self, pred, target, window_size=11):
        return 1.0 - ssim(pred, target, window_size=window_size)


class CombinedLoss(nn.Module):
    """Weighted sum of multiple losses, e.g. l1 + 0.3 * ssim."""
    def __init__(self, losses):
        """
        Args:
            losses: list of (weight, nn.Module) tuples.
        """
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = []
        for weight, loss_fn in losses:
            self.weights.append(float(weight))
            self.losses.append(loss_fn)

    def forward(self, pred, target):
        total = torch.tensor(0.0, device=pred.device)
        for weight, loss_fn in zip(self.weights, self.losses):
            total = total + weight * loss_fn(pred, target)
        return total


LOSS_FACTORY = {
    'l1': L1Loss,
    'charbonnier': CharbonnierLoss,
    'ssim': SSIMLoss,
}


def build_loss(name):
    """
    Build a loss from a config string, e.g. 'l1' or 'l1:1.0,ssim:0.3'.
    """
    name = name.strip()
    if ':' in name or ',' in name:
        modules = []
        for item in name.split(','):
            item = item.strip()
            if ':' in item:
                loss_name, weight = item.split(':')
                modules.append((float(weight), LOSS_FACTORY[loss_name.strip()]()))
            else:
                modules.append((1.0, LOSS_FACTORY[item]()))
        return CombinedLoss(modules)
    return LOSS_FACTORY[name]()