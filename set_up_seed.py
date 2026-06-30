import os

# Required by CUDA for reproducible cuBLAS matmul when deterministic mode is on.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Some Mamba operations have no deterministic implementation.  Warn about
    # them instead of turning a reproducibility aid into a runtime failure.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        # Older torch releases do not support warn_only; do not force the
        # stricter mode there because it can break Mamba execution.
        pass

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        transforms = getattr(worker_info.dataset, "image_transforms", None)
        if transforms is not None and hasattr(transforms, "set_random_seed"):
            transforms.set_random_seed(worker_seed)
