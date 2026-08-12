#!/usr/bin/env python
from __future__ import annotations

import argparse
import runpy
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the qualitative RISE figures from frozen raw saliency maps."
    )
    parser.add_argument("target", choices=("main", "galleries"))
    args, remaining = parser.parse_known_args()
    module = (
        "hcbf._rise_main_figure"
        if args.target == "main"
        else "hcbf._rise_galleries"
    )
    sys.argv = [module, *remaining]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
