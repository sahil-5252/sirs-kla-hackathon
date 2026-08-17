import os
import sys
import io
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from paths import default_args
from model import RestormerSR
from dataset import TestDataset


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RestormerSR.from_checkpoint(ckpt).to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    test_ds = TestDataset(args.test_dir)
    if args.test_limit > 0:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(args.test_limit, len(test_ds)))))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(test_ds)} test images -> {args.output_dir}")

    with torch.no_grad():
        for i, (noisy, filenames) in enumerate(test_loader):
            pred = model(noisy.to(device)).clamp(0, 1).cpu().numpy()
            for j, fn in enumerate(filenames):
                np.save(os.path.join(args.output_dir, fn), pred[j, 0])
            if (i + 1) % 20 == 0 or (i + 1) == len(test_loader):
                print(f"  [{min((i + 1) * args.batch_size, len(test_ds))}/{len(test_ds)}]")

    print(f"Done. Results saved to {args.output_dir}")


if __name__ == '__main__':
    d = default_args()
    parser = argparse.ArgumentParser(description='Restormer inference')
    parser.add_argument('--test_dir', type=str, default=d['test_dir'])
    parser.add_argument('--output_dir', type=str, default=d['test_out_dir'])
    parser.add_argument('--checkpoint', type=str, default=os.path.join(d['save_dir'], 'best_model.pth'))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_limit', type=int, default=0, help='Cap #test images for quick tests (0 = all)')
    args = parser.parse_args()
    evaluate(args)