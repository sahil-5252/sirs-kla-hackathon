import torch
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    _1d = g[:, None] * g[None, :]
    window = _1d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    """
    Structural Similarity Index between two tensors of shape (B, C, H, W).
    Values are clamped to [0, data_range] before computation.
    """
    pred = pred.clamp(0, data_range)
    target = target.clamp(0, data_range)
    (_, channel, _, _) = pred.size()

    window = _gaussian_window(window_size, sigma, channel, pred.device)

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
    mu2 = F.conv2d(target, window, padding=pad, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def psnr(pred, target, data_range=1.0):
    """Peak Signal-to-Noise Ratio between two tensors of shape (B, C, H, W)."""
    pred = pred.clamp(0, data_range)
    target = target.clamp(0, data_range)
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(torch.tensor(data_range ** 2, device=pred.device) / (mse + 1e-8))


def mae(pred, target):
    """Mean Absolute Error between two tensors of shape (B, C, H, W)."""
    return F.l1_loss(pred, target)