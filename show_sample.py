from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from data_processing import PHONEMES
from preparation_data import detokenize_text_with_space_train


PHONEME_INDEX_TO_SYMBOL = {idx + 1: symbol for idx, symbol in enumerate(PHONEMES)}


def _load_optional(path: Path) -> np.memmap | None:
    return np.load(path, mmap_mode="r") if path.exists() else None


def _decode_sample_name(raw: np.ndarray | None) -> str | None:
    if raw is None:
        return None
    value = raw.item() if hasattr(raw, "item") else raw
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", "ignore").rstrip("\x00").strip()
    if getattr(value, "dtype", None) is not None and value.dtype.kind in {"S", "U"}:  # type: ignore[attr-defined]
        try:
            return value.tobytes().decode("utf-8", "ignore").rstrip("\x00").strip()
        except AttributeError:
            pass
    return str(value)


def _decode_transcript(tokens: np.ndarray, length: int) -> str:
    slice_ = tokens[:length]
    if slice_.size == 0:
        return ""
    return detokenize_text_with_space_train(slice_)


def _decode_phonemes(tokens: np.ndarray, length: int) -> list[str]:
    result: list[str] = []
    for value in tokens[:length]:
        symbol = PHONEME_INDEX_TO_SYMBOL.get(int(value))
        result.append(symbol if symbol is not None else f"<{int(value)}>")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print textual fields (transcript, phonemes, lengths, names) for a single sample without loading the mel spectrogram."
    )
    parser.add_argument("directory", type=Path, help="Path to prepared dataset directory.")
    parser.add_argument("index", type=int, help="Sample index (0-based).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    directory = args.directory.resolve()
    x_len = np.load(directory / "x_input_length.npy", mmap_mode="r")
    y_labels = np.load(directory / "y_labels.npy", mmap_mode="r")
    y_len = np.load(directory / "y_label_length.npy", mmap_mode="r")

    sample_count = int(x_len.shape[0])
    if sample_count == 0:
        logging.error("Dataset is empty")
        return

    index = max(0, min(args.index, sample_count - 1))
    if index != args.index:
        logging.warning("Requested index %s out of bounds, using %s", args.index, index)

    sample_name_mem = _load_optional(directory / "xy_sample_names.npy")
    phoneme_labels = _load_optional(directory / "y_labels_phoneme.npy")
    phoneme_len = _load_optional(directory / "y_label_length_phoneme.npy")

    metadata_path = directory / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("Failed to read metadata.json: %s", exc)

    logging.info("Sample index: %s", index)
    if sample_name_mem is not None:
        logging.info("sample_name: %s", _decode_sample_name(sample_name_mem[index]))

    logging.info("x_input_length: %s", int(x_len[index]))

    label_length = int(y_len[index])
    logging.info("y_label_length: %s", label_length)
    transcript_tokens = y_labels[index, :label_length]
    logging.info("y_labels(tokens): %s", transcript_tokens.tolist())
    logging.info("transcript(text): %s", _decode_transcript(transcript_tokens, label_length))

    if phoneme_labels is not None and phoneme_len is not None:
        phoneme_length = int(phoneme_len[index])
        phoneme_tokens = phoneme_labels[index, :phoneme_length]
        logging.info("y_label_length_phoneme: %s", phoneme_length)
        logging.info("y_labels_phoneme(tokens): %s", phoneme_tokens.tolist())
        logging.info("phonemes: %s", " ".join(_decode_phonemes(phoneme_tokens, phoneme_length)))

    if metadata:
        logging.info("metadata: %s", metadata)


if __name__ == "__main__":
    main()
