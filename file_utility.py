from __future__ import annotations

import glob
import os
import re
from typing import List, Tuple


_TRANSCRIPT_PATTERN = re.compile(r"(\d+-\d+-\d{4})\s+(.*)")

def list_third_level_directories(path: str) -> List[str]:
    """Return directories containing LibriSpeech .flac files.

    LibriSpeech хранит файлы в структуре subset/speaker/chapter.
    Функция возвращает абсолютные пути к каталогам chapter (содержащим .flac файлы).
    """

    abs_root = os.path.abspath(path)
    results: List[str] = []
    for dirpath, _, filenames in os.walk(abs_root):
        # Проверяем, есть ли в текущей папке .flac файлы
        has_flac = any(f.endswith('.flac') for f in filenames)
        if has_flac:
            results.append(dirpath)
    results.sort()
    return results

def get_flac_files(path: str) -> List[str]:
    files = glob.glob(os.path.join(path, "**/*.flac"), recursive=True)
    files.sort()
    return files


def get_txt_files(path: str) -> List[str]:
    files = glob.glob(os.path.join(path, "**/*.txt"), recursive=True)
    files.sort()
    return files


def extract_pairs_from_file(file_path: str) -> List[Tuple[str, str]]:
    """Parse LibriSpeech transcript file into (base_name, transcript) pairs."""

    with open(file_path, "r", encoding="utf-8") as source:
        lines = source.readlines()

    pairs: List[Tuple[str, str]] = []
    for line in lines:
        match = _TRANSCRIPT_PATTERN.match(line.strip())
        if match:
            pairs.append((match.group(1), match.group(2)))

    return pairs

