#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hcbf.figures import make_all
from hcbf.reproducibility import verify_frozen_checksums, verify_frozen_results


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/frozen/figure_data"
OUTPUT = ROOT / "results/figures"


def verify() -> None:
    problems = verify_frozen_checksums(ROOT) + verify_frozen_results(ROOT)
    if problems:
        print("Verification FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("Verification PASSED.")
    print("Frozen files, protocol anchors, split counts, and headline results agree.")


def figures() -> None:
    make_all(DATA, OUTPUT)
    count = len(list(OUTPUT.glob("*.pdf")))
    print(f"Generated {count} PDF figures in {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify frozen HCBF results and regenerate publication figures."
    )
    parser.add_argument(
        "command",
        choices=("verify", "figures", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args()

    if args.command in ("verify", "all"):
        verify()
    if args.command in ("figures", "all"):
        figures()


if __name__ == "__main__":
    main()
