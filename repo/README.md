# Restormer Image Restoration (SemiCon AI Hackathon)

2x super-resolution + denoising of grayscale semiconductor images using the
official [Restormer](https://github.com/swz30/Restormer) backbone adapted for
2x learned upsampling.

- **Input:** `(B, 1, 128, 128)` noisy-LR .npy (float32, may exceed [0,1])
- **Output:** `(B, 1, 256, 256)` restored .npy (clamped to [0,1])
- **GT:** `(256, 256)` float32 in [0,1]
- **Pairs are matched by identical filename.**

## Repository layout

```
repo/                          <- this repository
├── src/
│   ├── restormer_arch.py      # official Restormer (vendored)
│   ├── model.py               # RestormerSR: Restormer + 2x PixelShuffle head
│   ├── dataset.py             # paired .npy datasets + identical-pair augmentations
│   ├── losses.py              # modular losses (L1 default, SSIM, combined)
│   ├── metrics.py             # PSNR, SSIM, MAE
│   ├── paths.py               # robust path resolution (local + Kaggle)
│   ├── train.py               # training pipeline
│   └── evaluate.py            # standalone inference
├── checkpoints/               # saved weights (best_model.pth, last_model.pth)
├── outputs/                   # restored test outputs (.npy)
├── requirements.txt
└── README.md

datasets/                      <- sibling of repo/ (NOT inside it)
├── train/GT/                  # 3200 ground-truth (256x256)
├── train/NoisyLR/             # 3200 noisy-LR (128x128)
├── NoisyLR/                   # 400 test images (128x128)
├── splits.csv                 # filename,split (90/10 train/val)
└── make_split.py
```

## Setup

```bash
git clone <this-repo-url> repo
# place the datasets/ folder as a SIBLING of repo/ (see layout above)
pip install -r requirements.txt
```

The scripts auto-locate the datasets folder by checking, in order:
1. `$DATA_DIR` environment variable
2. `<repo>/datasets`
3. `<repo>/../datasets` (sibling layout shown above)
4. `/kaggle/input/*`

You can always override explicitly: `--data_dir /path/to/datasets`.

## Training

```bash
# full run (T4 x2 on Kaggle: 2000 steps/epoch, bs 8 per GPU)
python src/train.py --epochs 100 --steps_per_epoch 2000 --batch_size 8 --multi_gpu

# quick sanity run on CPU
python src/train.py --epochs 1 --steps_per_epoch 2 --batch_size 2 --num_workers 0
```

Per-epoch logging prints train/val loss, val PSNR/SSIM/MAE, and current LR.
The best checkpoint (by validation SSIM) is saved to `checkpoints/best_model.pth`.

Key defaults: AdamW `lr=3e-4`, `weight_decay=1e-4`, `betas=(0.9,0.999)`,
cosine scheduler, L1 loss. All are configurable via CLI flags
(`--loss`, `--scheduler`, `--lr`, `--batch_size`, `--epochs`, `--seed`, ...).

## Inference

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
# writes restored .npy files to outputs/
```

The evaluate script loads the model, runs inference on all test images, and
writes restored outputs. No manual edits required.

## Augmentations (identical on both pair members)

Applied only to the `train` split, using the same random choices for the noisy
and GT members of each pair:

- Random rotation by a multiple of 90 degrees (0/90/180/270)
- Random horizontal flip (mirror)
- Random vertical flip (mirror)