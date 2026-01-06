from __future__ import annotations

import argparse

from src.phase01_preprocessing import main as phase01
from src.phase02_eda import main as phase02
from src.phase03_models import main as phase03
from src.phase04_tuning import main as phase04
from src.phase05_eval_infer import main as phase05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        default="all",
        choices=["1", "2", "3", "4", "5", "all"],
        help="Which phase to run.",
    )
    args = p.parse_args()

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
