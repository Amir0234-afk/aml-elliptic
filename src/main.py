# src/main.py
from __future__ import annotations

# Allows running either:
#   python -m src.main ...
# OR (less recommended, but supported):
#   python src/main.py ...
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import os
import torch

from src.config import ExperimentConfig


def _torch_cuda_version_str() -> str:
    # Pylance sometimes complains about torch.version; avoid hard attribute access.
    v = getattr(torch, "version", None)
    cuda = getattr(v, "cuda", None) if v is not None else None
    return str(cuda) if cuda is not None else "None"


def print_env() -> None:
    cfg = ExperimentConfig()
    print("torch:", torch.__version__)
    print("torch cuda compiled:", _torch_cuda_version_str())
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device 0:", torch.cuda.get_device_name(0))
    print("cfg.device:", cfg.device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "5", "all"])
    p.add_argument("--print-env", action="store_true", help="Print torch/cuda + resolved cfg.device and exit.")
    p.add_argument("--device", default=None, help="Override device: auto|cpu|cuda|cuda:0 (sets AML_DEVICE)")
    p.add_argument("--fast", action="store_true", help="Enable fast mode (sets AML_FAST=1)")

    args = p.parse_args()

    if args.device:
        os.environ["AML_DEVICE"] = str(args.device).strip()
    if args.fast:
        os.environ["AML_FAST"] = "1"

    if args.print_env:
        print_env()
        return

    # Import phases after env overrides so every phase resolves device consistently.
    from src.phase01_preprocessing import main as phase01
    from src.phase02_eda import main as phase02
    from src.phase03_models import main as phase03
    from src.phase04_tuning import main as phase04
    from src.phase05_eval_infer import main as phase05

    if args.phase in ("1", "all"):
        phase01()
    if args.phase in ("2", "all"):
        phase02()
    if args.phase in ("3", "all"):
        phase03()
    if args.phase in ("4", "all"):
        phase04()
    if args.phase in ("5", "all"):
        phase05()


if __name__ == "__main__":
    main()
