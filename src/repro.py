# src/repro.py
from __future__ import annotations

import os
import random
import numpy as np

def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            # Deterministic algorithms where available (may raise if an op is nondeterministic)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed or not used in a given phase
        pass
