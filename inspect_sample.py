from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from prepared_dataset_library import load_prepared_dataset


def _format_value(value: np.ndarray) -> object:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.dtype.kind in {"S", "U"}:
            return value.astype(str).tolist()
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Log one sample from the prepared dataset.")
    parser.add_argument("directory", type=Path, help="Path to the folder with prepared npy files.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--index", type=int, default=0, help="Sample index inside the selected split.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset = load_prepared_dataset(args.directory, shuffle=False)
    split_view = getattr(dataset, args.split)
    if not len(split_view):  # pragma: no cover - runtime guard
        logging.error("Split '%s' is empty", args.split)
        return

    clamped_index = max(0, min(args.index, len(split_view) - 1))
    if clamped_index != args.index:
        logging.warning("Requested index %s out of bounds, using %s", args.index, clamped_index)

    sample = split_view[clamped_index]

    for key, value in sample.items():
        if key == "x_data":
            continue
        logging.info("%s: %s", key, _format_value(value))


if __name__ == "__main__":
    main()
