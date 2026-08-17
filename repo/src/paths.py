import os


def repo_root():
    """Absolute path to the repo root (the directory containing src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_datasets_dir():
    """
    Locate the datasets directory robustly.

    Candidates, in priority order:
      1. $DATA_DIR environment variable (explicit override)
      2. <repo_root>/datasets
      3. <repo_root>/../datasets   (repo pushed as a subfolder, e.g. repo/ + datasets/ siblings)
      4. /kaggle/input/* (Kaggle dataset mount)

    Returns the first candidate that actually contains train/GT.
    """
    root = repo_root()

    candidates = []
    env = os.environ.get('DATA_DIR')
    if env:
        candidates.append(env)
    candidates += [
        os.path.join(root, 'datasets'),
        os.path.join(os.path.dirname(root), 'datasets'),
    ]
    if os.path.isdir('/kaggle/input'):
        for d in sorted(os.listdir('/kaggle/input')):
            candidates.append(os.path.join('/kaggle/input', d))

    for c in candidates:
        if os.path.isdir(os.path.join(c, 'train', 'GT')):
            return c

    # Fallback: return the first plausible candidate so the error message is clear.
    return os.path.join(root, 'datasets')


def default_args():
    """Resolve default paths for the CLI, handling both local and Kaggle layouts."""
    root = repo_root()
    data_dir = find_datasets_dir()
    return {
        'data_dir': data_dir,
        'save_dir': os.path.join(root, 'checkpoints'),
        'test_dir': os.path.join(data_dir, 'NoisyLR'),
        'test_out_dir': os.path.join(root, 'outputs'),
    }