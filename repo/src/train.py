import os
import sys
import io
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from paths import default_args
from model import RestormerSR
from dataset import TrainDataset, TestDataset
from losses import build_loss
from metrics import psnr, ssim, mae


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, args, total_steps):
    name = args.scheduler.lower()
    if name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if name == 'step':
        return StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    if name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_gamma,
                                 patience=args.lr_patience, min_lr=args.min_lr)
    if name == 'none':
        return None
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def step_scheduler(scheduler, val_ssim=None):
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_ssim)
    else:
        scheduler.step()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_mae = 0.0
    total_loss = 0.0
    count = 0

    for noisy, gt in loader:
        noisy, gt = noisy.to(device), gt.to(device)
        pred = model(noisy)
        total_loss += F.l1_loss(pred, gt).item() * noisy.shape[0]
        total_psnr += psnr(pred, gt).item() * noisy.shape[0]
        total_ssim += ssim(pred, gt).item() * noisy.shape[0]
        total_mae += mae(pred, gt).item() * noisy.shape[0]
        count += noisy.shape[0]

    return {
        'loss': total_loss / count,
        'psnr': total_psnr / count,
        'ssim': total_ssim / count,
        'mae': total_mae / count,
    }


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, metrics, is_best):
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.module.config() if isinstance(model, nn.DataParallel) else model.config(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
        'epoch': epoch,
        'args': vars(args),
        'metrics': metrics,
        'is_best': is_best,
    }, path)


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    csv_path = os.path.join(args.data_dir, 'splits.csv')
    train_ds = TrainDataset(
        csv_path=csv_path,
        gt_dir=os.path.join(args.data_dir, 'train', 'GT'),
        noisy_dir=os.path.join(args.data_dir, 'train', 'NoisyLR'),
        split='train', augment=True,
    )
    val_ds = TrainDataset(
        csv_path=csv_path,
        gt_dir=os.path.join(args.data_dir, 'train', 'GT'),
        noisy_dir=os.path.join(args.data_dir, 'train', 'NoisyLR'),
        split='val', augment=False,
    )

    if args.val_limit > 0:
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(args.val_limit, len(val_ds)))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    model = RestormerSR(
        inp_channels=1,
        out_channels=1,
        dim=args.dim,
        num_blocks=tuple(args.num_blocks),
        num_refinement_blocks=args.num_refinement_blocks,
        heads=tuple(args.heads),
    ).to(device)

    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
        print(f"Using DataParallel over {torch.cuda.device_count()} GPUs")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = build_loss(args.loss)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=tuple(args.betas), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, args.epochs * args.steps_per_epoch)

    os.makedirs(args.save_dir, exist_ok=True)
    best_ssim = -1.0
    start_epoch = 0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim = ckpt['metrics'].get('ssim', -1.0)
        print(f"Resumed from {args.resume} at epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        # 2000 optimizer steps per epoch (cycle/reshuffle over the data).
        loader_iter = iter(train_loader)
        while epoch_steps < args.steps_per_epoch:
            try:
                noisy, gt = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)  # reshuffle + cycle
                continue

            noisy, gt = noisy.to(device), gt.to(device)
            pred = model(noisy)
            loss = criterion(pred, gt)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

        train_loss = epoch_loss / epoch_steps

        val_metrics = evaluate(model, val_loader, device)
        step_scheduler(scheduler, val_metrics['ssim'])
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"PSNR: {val_metrics['psnr']:.2f} dB | "
            f"SSIM: {val_metrics['ssim']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        is_best = val_metrics['ssim'] > best_ssim
        if is_best:
            best_ssim = val_metrics['ssim']
        save_checkpoint(
            os.path.join(args.save_dir, 'last_model.pth'),
            model, optimizer, scheduler, args, epoch, val_metrics, is_best,
        )
        if is_best:
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model.module.config() if isinstance(model, nn.DataParallel) else model.config(),
                'metrics': val_metrics,
            }, best_path)
            print(f"  -> New best model saved (val SSIM {best_ssim:.4f})")

    print(f"\nTraining complete. Best val SSIM: {best_ssim:.4f}")


def run_test(args, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint, map_location=device)
    model = RestormerSR.from_checkpoint(ckpt).to(device)
    model.eval()

    test_ds = TestDataset(args.test_dir)
    if args.test_limit > 0:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(args.test_limit, len(test_ds)))))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    os.makedirs(args.test_out_dir, exist_ok=True)
    print(f"Running inference on {len(test_ds)} test images -> {args.test_out_dir}")

    with torch.no_grad():
        for noisy, filename in test_loader:
            pred = model(noisy.to(device)).clamp(0, 1).cpu().numpy()
            for i, fn in enumerate(filename):
                np.save(os.path.join(args.test_out_dir, fn), pred[i, 0])
    print("Inference done.")


if __name__ == '__main__':
    d = default_args()
    parser = argparse.ArgumentParser(description='Restormer 2x super-resolution training')
    parser.add_argument('--data_dir', type=str, default=d['data_dir'])
    parser.add_argument('--save_dir', type=str, default=d['save_dir'])
    parser.add_argument('--test_dir', type=str, default=d['test_dir'])
    parser.add_argument('--test_out_dir', type=str, default=d['test_out_dir'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=2000)
    parser.add_argument('--val_limit', type=int, default=0, help='Cap #val images for quick tests (0 = all)')
    parser.add_argument('--test_limit', type=int, default=0, help='Cap #test images for quick tests (0 = all)')
    parser.add_argument('--batch_size', type=int, default=8, help='Per-GPU batch size (T4 x2: 8/GPU)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument('--loss', type=str, default='l1', help="'l1', 'charbonnier', 'ssim', or 'l1:1.0,ssim:0.3'")
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'])
    parser.add_argument('--lr_step', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 6, 6, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--run_test', action='store_true', help='Run inference on test set after training')
    args = parser.parse_args()

    train(args)
    if args.run_test:
        run_test(args, os.path.join(args.save_dir, 'best_model.pth'))