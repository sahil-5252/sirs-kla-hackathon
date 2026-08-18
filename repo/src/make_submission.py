import os
import sys
import io
import base64
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from io import BytesIO
from tqdm import tqdm

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

from paths import default_args, find_datasets_dir
from model import RestormerSR
from dataset import TestDataset


@torch.no_grad()
def generate_predictions(model, test_dir, out_dir, device, use_tta, batch_size, num_workers, test_limit):
    test_ds = TestDataset(test_dir)
    if test_limit > 0:
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(test_limit, len(test_ds)))))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Predicting on {len(test_ds)} test images -> {out_dir}")

    for noisy, filenames in tqdm(loader, desc="Predicting test set", unit='batch',
                                 leave=True, mininterval=5, ncols=120):
        pred = predict_batch(model, noisy.to(device), use_tta).cpu().numpy()
        for j, fn in enumerate(filenames):
            p = np.nan_to_num(pred[j, 0], nan=0.0, posinf=1.0, neginf=0.0)
            np.save(os.path.join(out_dir, fn), p.astype(np.float32))

    files = sorted(os.listdir(out_dir))
    print(f"Saved {len(files)} predictions to {out_dir}")
    return files


@torch.no_grad()
def predict_batch(model, x, use_tta):
    if not use_tta:
        return model(x).clamp(0, 1)

    transforms = [
        (lambda t: t, lambda t: t),
        (lambda t: torch.flip(t, dims=[3]), lambda t: torch.flip(t, dims=[3])),
        (lambda t: torch.flip(t, dims=[2]), lambda t: torch.flip(t, dims=[2])),
        (lambda t: torch.flip(torch.flip(t, dims=[2]), dims=[3]),
         lambda t: torch.flip(torch.flip(t, dims=[2]), dims=[3])),
    ]
    preds = []
    for apply, unapply in transforms:
        p = model(apply(x)).clamp(0, 1)
        preds.append(unapply(p))
    return torch.stack(preds).mean(dim=0)


def build_submission_csv(out_dir, submission_csv):
    files = sorted([f for f in os.listdir(out_dir) if f.endswith('.npy')])
    rows = []
    for idx, fname in enumerate(files, start=1):
        arr = np.load(os.path.join(out_dir, fname))
        buf = BytesIO()
        np.save(buf, arr)
        encoded = base64.b64encode(buf.getvalue()).decode()
        rows.append({"id": idx, "npy_base64": encoded})
    df = pd.DataFrame(rows)
    df.to_csv(submission_csv, index=False)
    print(f"Submission CSV written: {submission_csv}  ({len(df)} rows)")
    return df


def sanity_check_against_sample_submission(data_root, produced_df):
    for root, _dirs, files in os.walk(data_root):
        for f in files:
            if f.lower() == "sample_submission.csv":
                sample_path = os.path.join(root, f)
                sample_df = pd.read_csv(sample_path)
                print(f"Found sample_submission.csv at {sample_path}")
                print(f"  sample columns : {list(sample_df.columns)}")
                print(f"  produced columns: {list(produced_df.columns)}")
                if list(sample_df.columns) != list(produced_df.columns):
                    print("  [!] Column names differ -- double check before submitting.")
                if len(sample_df) != len(produced_df):
                    print(f"  [!] Row count differs: sample={len(sample_df)} vs produced={len(produced_df)}")
                return sample_df
    print("No sample_submission.csv found under data_root -- skipping format cross-check.")
    return None


if __name__ == '__main__':
    d = default_args()
    parser = argparse.ArgumentParser(description='Build a Kaggle submission CSV from a trained RestormerSR checkpoint')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(d['save_dir'], 'best_model.pth'))
    parser.add_argument('--test_dir', type=str, default=d['test_dir'])
    parser.add_argument('--out_dir', type=str, default=d['test_out_dir'])
    parser.add_argument('--submission_csv', type=str, default=os.path.join(os.path.dirname(d['save_dir']), 'submission.csv'))
    parser.add_argument('--sample_dir', type=str, default=d['data_dir'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_limit', type=int, default=0, help='Cap #test images for quick tests (0 = all)')
    parser.add_argument('--tta', action='store_true', help='Use 4-fold flip test-time augmentation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RestormerSR.from_checkpoint(ckpt).to(device)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    generate_predictions(model, args.test_dir, args.out_dir, device,
                         args.tta, args.batch_size, args.num_workers, args.test_limit)
    df = build_submission_csv(args.out_dir, args.submission_csv)
    sanity_check_against_sample_submission(args.sample_dir, df)
