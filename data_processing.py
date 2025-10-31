"""High-level data preparation helpers for LibriSpeech CTC training.

This module extracts reusable pieces from the legacy scripts so that
`main.py` can orchestrate them via a GUI or CLI entrypoint.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from preparation_data import (
    detokenize_text_with_space_train,
    process_audio_and_text_nornalize,
)
from file_utility import extract_pairs_from_file, get_txt_files, list_third_level_directories


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for preparing the CTC-ready dataset."""

    root_dir: str
    output_dir: str
    time_steps: int = 1000
    label_max_len: int = 524
    mel_bins: int = 128
    frame_duration_ms: float | None = None
    frames_per_step: int = 1
    max_workers: int = 10
    shuffle: bool = False
    max_audio_duration_s: float | None = None


@dataclass(frozen=True)
class PhonemeConfig:
    """Configuration for optional phoneme conversion."""

    dataset_dir: str
    output_suffix: str = "_phoneme"
    max_phoneme_len: int = 2000


# Canonical CMU phoneme inventory with leading blank token for CTC.
PHONEMES: Sequence[str] = (
" ",
    "AA0","AA1","AA2",
    "AE0","AE1","AE2",
    "AH0","AH1","AH2",
    "AO0","AO1","AO2",
    "AW0","AW1","AW2",
    "AY0","AY1","AY2",
    "EH0","EH1","EH2",
    "ER0","ER1","ER2",
    "EY0","EY1","EY2",
    "IH0","IH1","IH2",
    "IY0","IY1","IY2",
    "OW0","OW1","OW2",
    "OY0","OY1","OY2",
    "UH0","UH1","UH2",
    "UW0","UW1","UW2",
    "B", "CH", "D", 
    "DH", "F", "G",
    "HH", "JH", "K",
    "L", "M", "N",
    "NG", "P", "R",
    "S", "SH", "T",
    "TH", "V", "W",
    "Y", "Z", "ZH",
    "'",
)

class DatasetPreparationError(RuntimeError):
    """Base error raised when dataset preparation fails."""


def _default_log(message: str) -> None:
    print(message)

def _collect_librispeech_directories(root_dir: str) -> tuple[list[str], list[str]]:
    """Return chapter directories containing data and the roots they were discovered under."""

    abs_root = os.path.abspath(root_dir)
    candidate_roots: list[str] = []
    for dirpath, dirnames, _ in os.walk(abs_root):
        if os.path.basename(dirpath).lower() == "librispeech":
            candidate_roots.append(dirpath)
            continue
    if candidate_roots:
        search_roots = sorted(set(candidate_roots))
    else:
        search_roots = [abs_root]

    directories: list[str] = []
    seen: set[str] = set()
    for base in search_roots:
        for directory in list_third_level_directories(base):
            if directory not in seen:
                directories.append(directory)
                seen.add(directory)

    return directories, search_roots


def _iter_librispeech_entries(directories: Sequence[str]) -> Iterator[Tuple[str, str, str]]:
    """Yield (audio_path, transcript, base_name) for every transcript in the directories."""

    for directory in directories:
        txt_files = get_txt_files(directory)
        for txt_file in txt_files:
            for base_name, transcript in extract_pairs_from_file(txt_file):
                audio_path = os.path.join(directory, f"{base_name}.flac")
                yield audio_path, transcript, base_name


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _effective_logit_length(frame_count: int, frames_per_step: int) -> float:
    if frames_per_step <= 0:
        raise ValueError("frames_per_step must be positive")
    return frame_count / float(frames_per_step)


def _shrink_npy_file(path: str, dtype: str, new_shape: tuple[int, ...]) -> None:
    """Rewrite an .npy memmap with a smaller leading dimension."""

    if new_shape[0] < 0:
        raise ValueError("new_shape must have non-negative length")
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    source = np.load(path, mmap_mode="r")
    target: np.memmap | None = None
    try:
        target = open_memmap(tmp_path, mode="w+", dtype=dtype, shape=new_shape)
        target[:] = source[: new_shape[0]]
        target.flush()
    finally:
        if target is not None:
            mmap_obj = getattr(target, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        source_mmap = getattr(source, "_mmap", None)
        if source_mmap is not None:
            source_mmap.close()

    os.replace(tmp_path, path)


def _process_entry(payload: Tuple[int, str, str, str, int, float | None]) -> Tuple[int, np.ndarray, Sequence[int], str]:
    """Heavy lifting function executed in worker processes."""

    index, audio_path, transcript, sample_name, mel_bins, frame_duration_ms = payload
    spectrogram, tokens = process_audio_and_text_nornalize(
        audio_path,
        transcript,
        mel_bins=mel_bins,
        frame_duration_ms=frame_duration_ms,
    )
    return index, spectrogram, tokens, sample_name


def prepare_ctc_dataset(config: DatasetConfig, log: LogFn | None = None) -> int:
    """Create memmapped CTC-ready arrays for the LibriSpeech subset.

    Returns the number of processed samples.
    """

    logger = log or _default_log
    if config.frames_per_step <= 0:
        raise DatasetPreparationError("Параметр frames_per_step должен быть положительным.")
    frames_per_step = int(config.frames_per_step)
    if config.max_audio_duration_s is not None and config.max_audio_duration_s <= 0:
        raise DatasetPreparationError("Параметр max_audio_duration_s должен быть положительным числом или None.")
    max_audio_duration_ms: float | None = (
        float(config.max_audio_duration_s) * 1000.0 if config.max_audio_duration_s is not None else None
    )
    frame_step_ms_value: float | None = None
    max_observed_duration_ms: float = 0.0

    directories, librispeech_roots = _collect_librispeech_directories(config.root_dir)
    if not directories:
        raise DatasetPreparationError("Не удалось найти папки с аудиофайлами и транскриптами.")

    base_abs = os.path.abspath(config.root_dir)
    if librispeech_roots and (len(librispeech_roots) > 1 or librispeech_roots[0] != base_abs):
        display: list[str] = []
        for root in librispeech_roots:
            if root == base_abs:
                display.append(root)
                continue
            try:
                rel = os.path.relpath(root, base_abs)
            except ValueError:
                rel = root
            if rel == ".":
                rel = root
            display.append(rel)
        logger("Обнаружены папки LibriSpeech: " + ", ".join(display))

    logger(f"Найдено {len(directories)} директорий с данными.")
    entries = list(_iter_librispeech_entries(directories))
    if config.shuffle and len(entries) > 1:
        logger("Перемешиваем порядок примеров перед обработкой.")
        rng = np.random.default_rng()
        rng.shuffle(entries)

    total_entries = len(entries)
    if total_entries == 0:
        raise DatasetPreparationError("В выбранной директории нет данных LibriSpeech.")

    logger(f"Подготовка массива на {total_entries} примеров...")

    dropped_short_inputs = 0
    dropped_exceeds_max_audio = 0
    max_observed_frames = 0

    def _frame_step_from_params(params: dict[str, float] | None) -> float | None:
        if not params:
            return float(config.frame_duration_ms) if config.frame_duration_ms is not None else None
        for key in ("effective_frame_duration_ms", "window_duration_ms", "requested_frame_duration_ms"):
            value = params.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return float(config.frame_duration_ms) if config.frame_duration_ms is not None else None

    first_valid_index: int | None = None
    first_spec_params: dict[str, float] | None = None
    first_spectrogram: np.ndarray | None = None
    first_tokens: Sequence[int] | None = None
    first_name: str | None = None

    for idx, (audio_path, transcript, sample_name) in enumerate(entries):
        spectrogram, tokens, spec_params = process_audio_and_text_nornalize(
            audio_path,
            transcript,
            mel_bins=config.mel_bins,
            frame_duration_ms=config.frame_duration_ms,
            return_params=True,
        )
        spec_length = spectrogram.shape[0]
        frame_step_candidate = _frame_step_from_params(spec_params)
        if frame_step_candidate is not None:
            frame_step_ms_value = frame_step_candidate
        if max_audio_duration_ms is not None:
            if frame_step_ms_value is None:
                raise DatasetPreparationError(
                    "Не удалось определить длительность кадра для проверки ограничения по длительности аудио."
                )
            sample_duration_ms = spec_length * frame_step_ms_value
            if sample_duration_ms > max_audio_duration_ms:
                dropped_exceeds_max_audio += 1
                logger(
                    f"Пропуск образца '{sample_name}': длительность {sample_duration_ms / 1000.0:.2f} с превышает максимум {config.max_audio_duration_s:.2f} с."
                )
                continue
        else:
            sample_duration_ms = spec_length * frame_step_ms_value if frame_step_ms_value is not None else None
        effective_length = _effective_logit_length(spec_length, frames_per_step)
        if effective_length < len(tokens):
            dropped_short_inputs += 1
            logger(
                f"Пропуск образца '{sample_name}': эффективная длина входа {effective_length:.2f} меньше длины меток {len(tokens)}."
            )
            continue
        first_valid_index = idx
        first_spec_params = spec_params
        first_spectrogram = spectrogram
        first_tokens = tokens
        first_name = sample_name
        if spec_length > max_observed_frames:
            max_observed_frames = spec_length
        if sample_duration_ms is not None and sample_duration_ms > max_observed_duration_ms:
            max_observed_duration_ms = sample_duration_ms
        break

    if first_valid_index is None or first_spectrogram is None or first_tokens is None or first_name is None:
        raise DatasetPreparationError(
            "Все примеры были удалены: длина входной последовательности меньше длины целевой."
        )

    feature_dim = first_spectrogram.shape[1]
    if first_spectrogram.shape[0] > config.time_steps:
        raise DatasetPreparationError(
            "Длина спектрограммы первого файла превышает выбранное окно. Увеличьте параметр 'размер окна'."
        )
    if len(first_tokens) > config.label_max_len:
        raise DatasetPreparationError(
            "Длина токенизированного текста превышает допустимую. Увеличьте максимальную длину метки."
        )

    _ensure_dir(config.output_dir)
    x_path = os.path.join(config.output_dir, "x_data.npy")
    y_path = os.path.join(config.output_dir, "y_labels.npy")
    x_len_path = os.path.join(config.output_dir, "x_input_length.npy")
    y_len_path = os.path.join(config.output_dir, "y_label_length.npy")
    names_path = os.path.join(config.output_dir, "xy_sample_names.npy")

    x_data = open_memmap(x_path, mode="w+", dtype="float32", shape=(total_entries, config.time_steps, feature_dim))
    y_labels = open_memmap(y_path, mode="w+", dtype="int64", shape=(total_entries, config.label_max_len))
    x_input_length = open_memmap(x_len_path, mode="w+", dtype="int64", shape=(total_entries,))
    y_label_length = open_memmap(y_len_path, mode="w+", dtype="int64", shape=(total_entries,))
    sample_names = open_memmap(names_path, mode="w+", dtype="S128", shape=(total_entries,))

    def write_sample(index: int, spectrogram: np.ndarray, tokens: Sequence[int], sample_name: str) -> None:
        nonlocal max_observed_frames
        if spectrogram.shape[0] > config.time_steps:
            raise DatasetPreparationError(
                f"Пример {index + 1}: спектрограмма длиной {spectrogram.shape[0]} превышает окно {config.time_steps}."
            )
        if len(tokens) > config.label_max_len:
            raise DatasetPreparationError(
                f"Пример {index + 1}: длина меток {len(tokens)} превышает максимум {config.label_max_len}."
            )
        x_data[index].fill(0.0)
        y_labels[index].fill(0)
        x_data[index, : spectrogram.shape[0], : spectrogram.shape[1]] = spectrogram.astype(np.float32)
        y_labels[index, : len(tokens)] = np.asarray(tokens, dtype=np.int64)
        x_input_length[index] = spectrogram.shape[0]
        y_label_length[index] = len(tokens)
        sample_names[index] = sample_name.encode("utf-8", "ignore")
        if spectrogram.shape[0] > max_observed_frames:
            max_observed_frames = spectrogram.shape[0]

    write_position = 0
    write_sample(write_position, first_spectrogram, first_tokens, first_name)
    write_position += 1

    worker_count = config.max_workers or os.cpu_count() or 1
    worker_count = max(1, worker_count)
    if worker_count > 1:
        logger(f"Используем {worker_count} процессов для обработки аудио.")

    def remaining_entries() -> Iterator[Tuple[int, str, str, str, int, float | None]]:
        for idx in range(first_valid_index + 1, total_entries):
            audio_path, transcript, sample_name = entries[idx]
            yield idx, audio_path, transcript, sample_name, config.mel_bins, config.frame_duration_ms

    if total_entries > 1:
        if worker_count == 1:
            for index, audio_path, transcript, sample_name, mel_bins, frame_duration_ms in remaining_entries():
                if (index + 1) % 100 == 0 or index + 1 == total_entries:
                    logger(f"Обработано {index + 1} / {total_entries} файлов...")
                spectrogram, tokens = process_audio_and_text_nornalize(
                    audio_path,
                    transcript,
                    mel_bins=mel_bins,
                    frame_duration_ms=frame_duration_ms,
                )
                spec_length = spectrogram.shape[0]
                sample_duration_ms = spec_length * frame_step_ms_value if frame_step_ms_value is not None else None
                if max_audio_duration_ms is not None:
                    if frame_step_ms_value is None:
                        raise DatasetPreparationError(
                            "Не удалось определить длительность кадра для проверки ограничения по длительности аудио."
                        )
                    if sample_duration_ms is None:
                        sample_duration_ms = spec_length * frame_step_ms_value
                    if sample_duration_ms > max_audio_duration_ms:
                        dropped_exceeds_max_audio += 1
                        logger(
                            f"Пропуск образца '{sample_name}': длительность {sample_duration_ms / 1000.0:.2f} с превышает максимум {config.max_audio_duration_s:.2f} с."
                        )
                        continue
                effective_length = _effective_logit_length(spec_length, frames_per_step)
                if effective_length < len(tokens):
                    dropped_short_inputs += 1
                    logger(
                        f"Пропуск образца '{sample_name}': эффективная длина входа {effective_length:.2f} меньше длины меток {len(tokens)}."
                    )
                    continue
                write_sample(write_position, spectrogram, tokens, sample_name)
                if sample_duration_ms is not None and sample_duration_ms > max_observed_duration_ms:
                    max_observed_duration_ms = sample_duration_ms
                write_position += 1
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for index, spectrogram, tokens, sample_name in executor.map(
                    _process_entry,
                    remaining_entries(),
                    chunksize=1,
                ):
                    if (index + 1) % 100 == 0 or index + 1 == total_entries:
                        logger(f"Обработано {index + 1} / {total_entries} файлов...")
                    spec_length = spectrogram.shape[0]
                    sample_duration_ms = spec_length * frame_step_ms_value if frame_step_ms_value is not None else None
                    if max_audio_duration_ms is not None:
                        if frame_step_ms_value is None:
                            raise DatasetPreparationError(
                                "Не удалось определить длительность кадра для проверки ограничения по длительности аудио."
                            )
                        if sample_duration_ms is None:
                            sample_duration_ms = spec_length * frame_step_ms_value
                        if sample_duration_ms > max_audio_duration_ms:
                            dropped_exceeds_max_audio += 1
                            logger(
                                f"Пропуск образца '{sample_name}': длительность {sample_duration_ms / 1000.0:.2f} с превышает максимум {config.max_audio_duration_s:.2f} с."
                            )
                            continue
                    effective_length = _effective_logit_length(spec_length, frames_per_step)
                    if effective_length < len(tokens):
                        dropped_short_inputs += 1
                        logger(
                            f"Пропуск образца '{sample_name}': эффективная длина входа {effective_length:.2f} меньше длины меток {len(tokens)}."
                        )
                        continue
                    write_sample(write_position, spectrogram, tokens, sample_name)
                    if sample_duration_ms is not None and sample_duration_ms > max_observed_duration_ms:
                        max_observed_duration_ms = sample_duration_ms
                    write_position += 1

    x_data.flush()
    y_labels.flush()
    x_input_length.flush()
    y_label_length.flush()
    sample_names.flush()

    del x_data
    del y_labels
    del x_input_length
    del y_label_length
    del sample_names

    valid_samples = write_position

    if valid_samples <= 0:
        raise DatasetPreparationError(
            "После фильтрации не осталось допустимых примеров: длина входной последовательности меньше длины целевой."
        )

    if valid_samples < total_entries:
        _shrink_npy_file(x_path, "float32", (valid_samples, config.time_steps, feature_dim))
        _shrink_npy_file(y_path, "int64", (valid_samples, config.label_max_len))
        _shrink_npy_file(x_len_path, "int64", (valid_samples,))
        _shrink_npy_file(y_len_path, "int64", (valid_samples,))
        _shrink_npy_file(names_path, "S128", (valid_samples,))
        removed_total = total_entries - valid_samples
        logger(f"Удалено {removed_total} примеров, не прошедших фильтры.")
        if dropped_short_inputs:
            logger(f"  - {dropped_short_inputs} из-за короткой входной последовательности.")
        if dropped_exceeds_max_audio:
            logger(f"  - {dropped_exceeds_max_audio} из-за превышения максимальной длины аудио.")

    metadata_path = os.path.join(config.output_dir, "metadata.json")
    metadata = {
        "time_steps": config.time_steps,
        "label_max_len": config.label_max_len,
        "mel_bins": config.mel_bins,
        "frame_duration_ms": config.frame_duration_ms,
        "frames_per_step": frames_per_step,
        "sample_count": valid_samples,
        "shuffle": bool(config.shuffle),
    }
    if config.max_audio_duration_s is not None:
        metadata["max_audio_duration_s"] = float(config.max_audio_duration_s)
        if max_audio_duration_ms is not None:
            metadata["max_audio_duration_ms"] = float(max_audio_duration_ms)
    if first_spec_params:
        metadata["spectrogram_params"] = first_spec_params
    frame_step_ms = frame_step_ms_value
    if frame_step_ms is None and first_spec_params:
        frame_step_ms = _frame_step_from_params(first_spec_params)
    if frame_step_ms is None and config.frame_duration_ms is not None:
        frame_step_ms = float(config.frame_duration_ms)
    if max_observed_frames:
        metadata["max_observed_audio_frames"] = int(max_observed_frames)
    if max_observed_duration_ms and max_observed_duration_ms > 0:
        metadata["max_observed_audio_duration_ms"] = float(max_observed_duration_ms)
        metadata["max_observed_audio_duration_s"] = float(max_observed_duration_ms / 1000.0)
    elif frame_step_ms is not None and max_observed_frames:
        computed = float(max_observed_frames * frame_step_ms)
        metadata["max_observed_audio_duration_ms"] = computed
        metadata["max_observed_audio_duration_s"] = computed / 1000.0
    if dropped_short_inputs:
        metadata["dropped_shorter_than_label"] = dropped_short_inputs
    if dropped_exceeds_max_audio:
        metadata["dropped_longer_than_max_audio"] = dropped_exceeds_max_audio
    try:
        with open(metadata_path, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger(f"Не удалось сохранить metadata.json: {exc}")

    logger(f"Готово: обработано {valid_samples} из {total_entries} файлов.")
    logger("Файлы сохранены:")
    logger(f"  {x_path}")
    logger(f"  {y_path}")
    logger(f"  {x_len_path}")
    logger(f"  {y_len_path}")
    logger(f"  {names_path}")
    if os.path.exists(metadata_path):
        logger(f"  {metadata_path}")

    return valid_samples


def convert_dataset_to_phonemes(
    config: PhonemeConfig,
    log: LogFn | None = None,
) -> int:
    """Convert token sequences to phoneme indices and persist alongside the dataset."""

    logger = log or _default_log
    try:
        from g2p_en import G2p
    except ImportError as exc:  # pragma: no cover - informative path only
        raise DatasetPreparationError(
            "Пакет g2p_en не установлен. Установите его командой 'pip install g2p_en'."
        ) from exc

    dataset_dir = config.dataset_dir
    y_path = os.path.join(dataset_dir, "y_labels.npy")
    y_len_path = os.path.join(dataset_dir, "y_label_length.npy")

    if not os.path.exists(y_path) or not os.path.exists(y_len_path):
        raise DatasetPreparationError(
            "Не найдены файлы y_labels.npy и y_label_length.npy для конвертации в фонемы."
        )

    y_labels = np.load(y_path, mmap_mode="r")
    y_label_length = np.load(y_len_path, mmap_mode="r")
    sample_count = y_labels.shape[0]

    g2p = G2p()
    phoneme_map = {phoneme: idx + 1 for idx, phoneme in enumerate(PHONEMES)}

    def encode_phonemes(items: Sequence[str]) -> np.ndarray:
        encoded = []
        for phoneme in items:
            try:
                encoded.append(phoneme_map[phoneme])
            except KeyError as exc:
                raise DatasetPreparationError(f"Неизвестная фонема '{exc.args[0]}'") from exc
        return np.asarray(encoded, dtype=np.int64)

    suffix = config.output_suffix.strip() or "_phoneme"
    phoneme_y_path = os.path.join(dataset_dir, f"y_labels{suffix}.npy")
    phoneme_len_path = os.path.join(dataset_dir, f"y_label_length{suffix}.npy")

    y_labels_phoneme = open_memmap(
        phoneme_y_path,
        mode="w+",
        dtype="int64",
        shape=(sample_count, config.max_phoneme_len),
    )
    y_label_length_phoneme = open_memmap(
        phoneme_len_path,
        mode="w+",
        dtype="int64",
        shape=(sample_count,),
    )

    processed = 0
    for idx in range(sample_count):
        label_length = int(y_label_length[idx])
        token_sequence = y_labels[idx, :label_length]
        text = detokenize_text_with_space_train(token_sequence)
        phoneme_seq = g2p(text)
        encoded = encode_phonemes(phoneme_seq)
        if encoded.shape[0] > config.max_phoneme_len:
            raise DatasetPreparationError(
                f"Пример {idx + 1}: последовательность фонем длиной {encoded.shape[0]} превышает максимум {config.max_phoneme_len}."
            )
        y_labels_phoneme[idx].fill(0)
        y_labels_phoneme[idx, : encoded.shape[0]] = encoded
        y_label_length_phoneme[idx] = encoded.shape[0]
        processed += 1
        if processed % 100 == 0:
            logger(f"Фонемы: обработано {processed} / {sample_count} примеров...")

    y_labels_phoneme.flush()
    y_label_length_phoneme.flush()

    logger("Фонемные файлы сохранены:")
    logger(f"  {phoneme_y_path}")
    logger(f"  {phoneme_len_path}")

    return processed


__all__ = [
    "DatasetConfig",
    "PhonemeConfig",
    "PHONEMES",
    "DatasetPreparationError",
    "prepare_ctc_dataset",
    "convert_dataset_to_phonemes",
]
