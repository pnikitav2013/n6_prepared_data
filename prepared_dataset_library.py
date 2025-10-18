"""Utilities for consuming prepared LibriSpeech-style datasets without loading them into RAM.

This module keeps the original memory-mapped arrays on disk and only
stores lightweight index arrays for train/validation/test splits.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


DATA_FILENAMES = {
    "x_data": "x_data.npy",
    "y_labels": "y_labels.npy",
    "x_input_length": "x_input_length.npy",
    "y_label_length": "y_label_length.npy",
    "sample_names": "xy_sample_names.npy",
    "metadata": "metadata.json",
}


@dataclass(frozen=True)
class DatasetMetadata:
    sample_count: int
    time_steps: int
    feature_dim: int
    label_max_len: int | None
    frames_per_step: int | None
    frame_duration_ms: float | None
    mel_bins: int | None
    extra: Mapping[str, object]


class MemmapDatasetView(Sequence[Mapping[str, np.ndarray]]):
    """Read-only view over a subset of the dataset using shared memmaps."""

    def __init__(
        self,
        indices: np.ndarray,
        arrays: Mapping[str, np.memmap],
    ) -> None:
        self._indices = indices.astype(np.int64, copy=False)
        self._arrays = arrays

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self._indices.size)

    def __getitem__(self, item: int | slice | Sequence[int] | np.ndarray) -> Mapping[str, np.ndarray]:
        if isinstance(item, slice):
            idx = self._indices[item]
            payload: dict[str, np.ndarray] = {
                "x_data": self._arrays["x_data"][idx],
                "x_input_length": self._arrays["x_input_length"][idx],
            }
            if "y_labels" in self._arrays and "y_label_length" in self._arrays:
                payload["y_labels"] = self._arrays["y_labels"][idx]
                payload["y_label_length"] = self._arrays["y_label_length"][idx]
            if "sample_names" in self._arrays:
                payload["sample_names"] = self._arrays["sample_names"][idx]
            return payload
        if isinstance(item, (np.ndarray, list, tuple)):
            pos = np.asarray(item, dtype=np.int64)
            idx = self._indices[pos]
            payload: dict[str, np.ndarray] = {
                "x_data": self._arrays["x_data"][idx],
                "x_input_length": self._arrays["x_input_length"][idx],
            }
            if "y_labels" in self._arrays and "y_label_length" in self._arrays:
                payload["y_labels"] = self._arrays["y_labels"][idx]
                payload["y_label_length"] = self._arrays["y_label_length"][idx]
            if "sample_names" in self._arrays:
                payload["sample_names"] = self._arrays["sample_names"][idx]
            return payload
        idx = self._indices[int(item)]
        single_idx = int(idx)
        result: dict[str, np.ndarray] = {
            "x_data": self._arrays["x_data"][single_idx],
            "x_input_length": np.asarray(self._arrays["x_input_length"][single_idx]),
        }
        if "y_labels" in self._arrays and "y_label_length" in self._arrays:
            result["y_labels"] = self._arrays["y_labels"][single_idx]
            result["y_label_length"] = np.asarray(self._arrays["y_label_length"][single_idx])
        if "sample_names" in self._arrays:
            result["sample_names"] = np.asarray(self._arrays["sample_names"][single_idx])
        return result

    def iter_indices(self, batch_size: int) -> Iterator[np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = len(self)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            yield np.arange(start, stop, dtype=np.int64)


@dataclass(frozen=True)
class PreparedDataset:
    metadata: DatasetMetadata
    train: MemmapDatasetView
    validation: MemmapDatasetView
    test: MemmapDatasetView


def _open_memmaps(directory: Path) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    x_path = directory / DATA_FILENAMES["x_data"]
    if not x_path.exists():
        raise FileNotFoundError(f"Не найден файл {x_path}")
    arrays["x_data"] = np.load(x_path, mmap_mode="r")

    x_len_path = directory / DATA_FILENAMES["x_input_length"]
    if not x_len_path.exists():
        raise FileNotFoundError(f"Не найден файл {x_len_path}")
    arrays["x_input_length"] = np.load(x_len_path, mmap_mode="r")

    y_path = directory / DATA_FILENAMES["y_labels"]
    y_len_path = directory / DATA_FILENAMES["y_label_length"]
    if y_path.exists() and y_len_path.exists():
        arrays["y_labels"] = np.load(y_path, mmap_mode="r")
        arrays["y_label_length"] = np.load(y_len_path, mmap_mode="r")

    sample_names_path = directory / DATA_FILENAMES["sample_names"]
    if sample_names_path.exists():
        arrays["sample_names"] = np.load(sample_names_path, mmap_mode="r")

    return arrays


def _load_metadata(directory: Path, arrays: Mapping[str, np.memmap]) -> DatasetMetadata:
    meta_path = directory / DATA_FILENAMES["metadata"]
    data: dict[str, object] = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    x_memmap = arrays["x_data"]
    sample_count = int(x_memmap.shape[0])
    time_steps = int(x_memmap.shape[1]) if x_memmap.ndim >= 2 else 0
    feature_dim = int(x_memmap.shape[2]) if x_memmap.ndim >= 3 else 0
    label_max_len = None
    if "y_labels" in arrays:
        label_max_len = int(arrays["y_labels"].shape[1])

    frames_per_step_value: int | None = None
    frame_duration_value: float | None = None
    mel_bins_value: int | None = None
    if isinstance(data, dict):
        frames_per_step = data.get("frames_per_step")
        if frames_per_step is not None:
            try:
                frames_per_step_value = int(frames_per_step)
            except (TypeError, ValueError):
                frames_per_step_value = None
        frame_duration = data.get("frame_duration_ms")
        if frame_duration is not None:
            try:
                frame_duration_value = float(frame_duration)
            except (TypeError, ValueError):
                frame_duration_value = None
        mel_bins = data.get("mel_bins")
        if mel_bins is not None:
            try:
                mel_bins_value = int(mel_bins)
            except (TypeError, ValueError):
                mel_bins_value = None

    return DatasetMetadata(
        sample_count=sample_count,
        time_steps=time_steps,
        feature_dim=feature_dim,
        label_max_len=label_max_len,
    frames_per_step=frames_per_step_value,
        frame_duration_ms=frame_duration_value,
        mel_bins=mel_bins_value,
        extra=data,
    )


def _resolve_split_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    if total <= 0:
        raise ValueError("Dataset is empty")
    if len(ratios) != 3:
        raise ValueError("Expected three ratios for train/validation/test")
    ratios = np.asarray(ratios, dtype=float)
    if not np.isfinite(ratios).all():
        raise ValueError("Ratios must be finite numbers")
    total_ratio = ratios.sum()
    if total_ratio <= 0:
        raise ValueError("Ratios must sum to a positive value")
    ratios = ratios / total_ratio

    cumulative = np.cumsum(ratios)
    boundaries = np.floor(cumulative * total + 1e-8).astype(int)
    train = int(boundaries[0])
    valid = int(boundaries[1] - boundaries[0])
    test = int(total - boundaries[1])

    counts = [train, valid, test]
    if total >= 3:
        for idx, value in enumerate(counts):
            if value == 0:
                donor = max(range(3), key=lambda i: counts[i])
                if donor != idx and counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    imbalance = total - sum(counts)
    if imbalance > 0:
        for i in range(imbalance):
            counts[i % 3] += 1
    elif imbalance < 0:
        for _ in range(-imbalance):
            donor = max(range(3), key=lambda i: counts[i])
            if counts[donor] == 0:
                break
            counts[donor] -= 1

    if any(count < 0 for count in counts):
        raise ValueError("Invalid split configuration produced negative counts")

    return counts[0], counts[1], counts[2]


def load_prepared_dataset(
    directory: str | os.PathLike[str],
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    shuffle: bool = True,
    seed: int | None = None,
) -> PreparedDataset:
    """Open the prepared dataset and return train/validation/test views.

    The returned object keeps references to the original memmap arrays so
    the underlying `.npy` files are never copied into RAM.
    """

    directory = Path(directory).resolve()
    arrays = _open_memmaps(directory)
    metadata = _load_metadata(directory, arrays)

    indices = np.arange(metadata.sample_count, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_count, valid_count, test_count = _resolve_split_counts(metadata.sample_count, split_ratios)

    train_idx = indices[:train_count]
    valid_idx = indices[train_count : train_count + valid_count]
    test_idx = indices[train_count + valid_count : train_count + valid_count + test_count]

    arrays_views = {
        "x_data": arrays["x_data"],
        "x_input_length": arrays["x_input_length"],
    }
    if "y_labels" in arrays and "y_label_length" in arrays:
        arrays_views["y_labels"] = arrays["y_labels"]
        arrays_views["y_label_length"] = arrays["y_label_length"]
    if "sample_names" in arrays:
        arrays_views["sample_names"] = arrays["sample_names"]

    train_view = MemmapDatasetView(train_idx, arrays_views)
    valid_view = MemmapDatasetView(valid_idx, arrays_views)
    test_view = MemmapDatasetView(test_idx, arrays_views)

    return PreparedDataset(metadata=metadata, train=train_view, validation=valid_view, test=test_view)


__all__ = [
    "DatasetMetadata",
    "MemmapDatasetView",
    "PreparedDataset",
    "load_prepared_dataset",
]
