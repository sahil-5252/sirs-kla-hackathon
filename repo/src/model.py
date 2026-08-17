import torch
import torch.nn as nn
import torch.nn.functional as F

from restormer_arch import Restormer


class RestormerSR(nn.Module):
    """
    Restormer backbone adapted for 2x learned super-resolution.

    Input:  (B, 1, 128, 128) noisy-LR tensor (NOT clipped/normalized)
    Output: (B, 1, 256, 256) restored HR image

    The official Restormer encoder-decoder runs at input resolution and its
    `forward_features()` returns a refined feature map (dim*2 channels) at the
    input size. A PixelShuffle-based head then performs 2x learned upsampling,
    followed by a global residual skip of the bicubically-upsampled input.
    """
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inp_channels = inp_channels
        self._cfg = {
            'inp_channels': inp_channels,
            'out_channels': out_channels,
            'dim': dim,
            'num_blocks': list(num_blocks),
            'num_refinement_blocks': num_refinement_blocks,
            'heads': list(heads),
            'ffn_expansion_factor': ffn_expansion_factor,
            'bias': bias,
            'LayerNorm_type': LayerNorm_type,
        }

        self.backbone = Restormer(
            inp_channels=inp_channels,
            out_channels=dim * 2,  # dummy; forward_features() is used instead
            dim=dim,
            num_blocks=list(num_blocks),
            num_refinement_blocks=num_refinement_blocks,
            heads=list(heads),
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.up_head = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2 * 4, kernel_size=3, padding=1, bias=bias),
            nn.PixelShuffle(2),          # dim*2 channels at 2x resolution
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
        )
        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1, bias=bias)

    def config(self):
        """Constructor kwargs, saved in checkpoints so the arch can be rebuilt."""
        return dict(self._cfg)

    @classmethod
    def from_checkpoint(cls, ckpt):
        cfg = ckpt.get('model_config')
        if cfg is None:
            raise ValueError("Checkpoint has no 'model_config'. Re-train or provide --dim/--num_blocks/--heads matching the saved weights.")
        model = cls(**cfg)
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    def forward(self, x):
        feats = self.backbone.forward_features(x)          # (B, dim*2, 128, 128)
        out = self.up_head(feats)                           # (B, dim*2, 256, 256)
        out = self.output(out)                              # (B, out_ch, 256, 256)
        up_inp = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = out + up_inp                                   # global residual
        return out