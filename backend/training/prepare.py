from __future__ import annotations

import argparse
from pathlib import Path

from training.datasets import assert_no_split_leakage, prepare_arafa


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an Arabic fact-verification dataset")
    parser.add_argument("--dataset", choices=["arafa"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/arafa"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_license:
        parser.error("ARAFA is CC-BY-NC-SA-4.0; pass --accept-license after reviewing its terms")
    manifest = prepare_arafa(args.source, args.output, args.seed)
    assert_no_split_leakage(args.output)
    print(f"Prepared {sum(item['count'] for item in manifest['splits'].values())} examples")


if __name__ == "__main__":
    main()
