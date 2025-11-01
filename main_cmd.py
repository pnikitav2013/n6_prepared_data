from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from data_processing import (
    DatasetConfig,
    DatasetPreparationError,
    PhonemeConfig,
    convert_dataset_to_phonemes,
    prepare_ctc_dataset,
)


DEFAULT_CONFIG_NAME = "dataset_app_settings_cmd.json"


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Не удалось разобрать JSON в {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Файл {path} должен содержать объект JSON с параметрами.")
    return data


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Нельзя использовать логическое значение для параметра {name}.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("пустая строка")
        return int(cleaned, 10)
    raise ValueError(f"неподдерживаемый тип {type(value).__name__}")


def _parse_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Нельзя использовать логическое значение для параметра {name}.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        if not cleaned:
            raise ValueError("пустая строка")
        return float(cleaned)
    raise ValueError(f"неподдерживаемый тип {type(value).__name__}")


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "t", "yes", "y", "on", "да", "истина"}:
            return True
        if cleaned in {"0", "false", "f", "no", "n", "off", "нет", "ложь"}:
            return False
    raise ValueError(f"некорректное значение {value!r} для {name}")


def _parse_str(value: Any, name: str) -> str:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("пустая строка")
        return cleaned
    if value is None:
        raise ValueError("значение отсутствует")
    return str(value)


def _pick(
    name: str,
    cli_value: Any,
    config: dict[str, Any],
    parser: Callable[[Any, str], Any],
    *,
    key: str | None = None,
    required: bool = False,
) -> Any:
    if cli_value is not None:
        return cli_value
    lookup_key = key or name
    missing_message = (
        f"Параметр '{name}' обязателен: задайте его флагом '--{name.replace('_', '-')}' "
        f"или в разделе '{lookup_key}' файла настроек."
    )
    if lookup_key not in config:
        if required:
            raise SystemExit(missing_message)
        return None
    raw_value = config[lookup_key]
    if raw_value is None:
        if required:
            raise SystemExit(missing_message)
        return None
    if isinstance(raw_value, str) and not raw_value.strip():
        if required:
            raise SystemExit(missing_message)
        return None
    try:
        return parser(raw_value, lookup_key)
    except ValueError as exc:
        raise SystemExit(f"Ошибочное значение для '{lookup_key}': {exc}") from exc


def _validate_positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise SystemExit(f"Параметр '{name}' должен быть положительным числом.")


def _print_separator() -> None:
    print("-" * 60)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Подготовка датасета LibriSpeech через консоль.",
    )
    parser.add_argument("--config", help="Путь к JSON с параметрами.")
    parser.add_argument("--root-dir", help="Корневая директория LibriSpeech.")
    parser.add_argument("--output-dir", help="Директория для сохранения результатов.")
    parser.add_argument("--time-steps", type=int, help="Максимум временных шагов спектрограммы.")
    parser.add_argument("--label-max-len", type=int, help="Максимальная длина последовательности меток.")
    parser.add_argument("--frames-per-step", type=int, help="Количество кадров на шаг модели.")
    parser.add_argument("--max-audio-duration", type=float, help="Максимальная длительность аудио в секундах.")
    parser.add_argument("--mel-bins", type=int, help="Количество мел-коэффициентов.")
    parser.add_argument("--frame-duration-ms", type=float, help="Длительность кадра в миллисекундах.")
    parser.add_argument("--max-workers", type=int, help="Количество процессов для обработки аудио.")
    parser.add_argument(
        "--shuffle",
        dest="shuffle_examples",
        action="store_true",
        help="Перемешивать примеры перед обработкой.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle_examples",
        action="store_false",
        help="Отключить перемешивание примеров.",
    )
    parser.add_argument(
        "--convert-phoneme",
        dest="convert_phoneme",
        action="store_true",
        help="Сохранить последовательности фонем.",
    )
    parser.add_argument(
        "--no-convert-phoneme",
        dest="convert_phoneme",
        action="store_false",
        help="Не выполнять конвертацию в фонемы.",
    )
    parser.add_argument("--phoneme-max-len", type=int, help="Максимальная длина фонемной последовательности.")
    parser.add_argument(
        "--phoneme-output-suffix",
        help="Суффикс для файлов фонем (по умолчанию '_phoneme').",
    )
    parser.add_argument(
        "--phoneme-dataset-dir",
        help="Директория с подготовленным датасетом для фонемной конвертации.",
    )
    parser.add_argument(
        "--phoneme-lexicon-path",
        help="Путь к пользовательскому лексикону (формат CMU).",
    )
    parser.add_argument(
        "--phoneme-use-lexicon",
        dest="phoneme_use_lexicon",
        action="store_true",
        help="Принудительно использовать лексикон при конвертации фонем (включено по умолчанию).",
    )
    parser.add_argument(
        "--phoneme-disable-lexicon",
        dest="phoneme_use_lexicon",
        action="store_false",
        help="Не использовать лексикон, всегда применять g2p_en.",
    )
    parser.set_defaults(convert_phoneme=None, shuffle_examples=None, phoneme_use_lexicon=None)
    return parser


def merge_parameters(args: argparse.Namespace, config: dict[str, Any]) -> tuple[DatasetConfig, PhonemeConfig | None]:
    root_dir = _pick("root_dir", args.root_dir, config, _parse_str, required=True)
    output_dir = _pick("output_dir", args.output_dir, config, _parse_str, required=True)
    time_steps = _pick("time_steps", args.time_steps, config, _parse_int, required=True)
    label_max_len = _pick("label_max_len", args.label_max_len, config, _parse_int, required=True)
    frames_per_step = _pick("frames_per_step", args.frames_per_step, config, _parse_int, required=True)
    mel_bins = _pick("mel_bins", args.mel_bins, config, _parse_int, required=True)
    frame_duration_ms = _pick("frame_duration_ms", args.frame_duration_ms, config, _parse_float, required=True)
    max_audio_duration = _pick(
        "max_audio_duration",
        args.max_audio_duration,
        config,
        _parse_float,
        key="max_audio_duration_s",
        required=False,
    )
    max_workers = _pick("max_workers", args.max_workers, config, _parse_int)  # optional

    shuffle_default = config.get("shuffle_examples")
    if args.shuffle_examples is None and shuffle_default is not None:
        shuffle_flag = _parse_bool(shuffle_default, "shuffle_examples")
    elif args.shuffle_examples is None:
        shuffle_flag = False
    else:
        shuffle_flag = bool(args.shuffle_examples)

    convert_default = config.get("convert_phoneme")
    if args.convert_phoneme is None and convert_default is not None:
        convert_flag = _parse_bool(convert_default, "convert_phoneme")
    elif args.convert_phoneme is None:
        convert_flag = False
    else:
        convert_flag = bool(args.convert_phoneme)

    _validate_positive(time_steps, "time_steps")
    _validate_positive(label_max_len, "label_max_len")
    _validate_positive(frames_per_step, "frames_per_step")
    _validate_positive(mel_bins, "mel_bins")
    _validate_positive(frame_duration_ms, "frame_duration_ms")
    if max_audio_duration is not None:
        _validate_positive(max_audio_duration, "max_audio_duration_s")
    if max_workers is not None:
        _validate_positive(max_workers, "max_workers")

    dataset_config = DatasetConfig(
        root_dir=root_dir,
        output_dir=output_dir,
        time_steps=time_steps,
        label_max_len=label_max_len,
        mel_bins=mel_bins,
        frame_duration_ms=frame_duration_ms,
        frames_per_step=frames_per_step,
        max_workers=max_workers if max_workers is not None else 10,
        shuffle=shuffle_flag,
        max_audio_duration_s=max_audio_duration,
    )

    phoneme_config: PhonemeConfig | None = None
    if convert_flag:
        phoneme_use_lexicon = _pick(
            "phoneme_use_lexicon",
            args.phoneme_use_lexicon,
            config,
            _parse_bool,
            key="phoneme_use_lexicon",
        )
        phoneme_max_len = _pick(
            "phoneme_max_len",
            args.phoneme_max_len,
            config,
            _parse_int,
        )
        phoneme_suffix = _pick(
            "phoneme_output_suffix",
            args.phoneme_output_suffix,
            config,
            _parse_str,
            key="phoneme_output_suffix",
        )
        phoneme_dataset_dir = _pick(
            "phoneme_dataset_dir",
            args.phoneme_dataset_dir,
            config,
            _parse_str,
            key="phoneme_dataset_dir",
        )
        phoneme_lexicon_path = _pick(
            "phoneme_lexicon_path",
            args.phoneme_lexicon_path,
            config,
            _parse_str,
            key="phoneme_lexicon_path",
        )
        if phoneme_max_len is None:
            phoneme_max_len = 2000
        else:
            _validate_positive(phoneme_max_len, "phoneme_max_len")
        output_suffix = phoneme_suffix if phoneme_suffix is not None else "_phoneme"
        dataset_dir_for_phoneme = phoneme_dataset_dir if phoneme_dataset_dir is not None else output_dir
        use_lexicon_flag = phoneme_use_lexicon if phoneme_use_lexicon is not None else True
        phoneme_config = PhonemeConfig(
            dataset_dir=dataset_dir_for_phoneme,
            output_suffix=output_suffix,
            max_phoneme_len=phoneme_max_len,
            lexicon_path=phoneme_lexicon_path,
            use_lexicon=use_lexicon_flag,
        )

    return dataset_config, phoneme_config


def _print_effective_settings(dataset_config: DatasetConfig, phoneme_config: PhonemeConfig | None) -> None:
    _print_separator()
    print("Параметры подготовки датасета:")
    print(f"  Корневой каталог: {dataset_config.root_dir}")
    print(f"  Каталог вывода: {dataset_config.output_dir}")
    print(f"  Временных шагов: {dataset_config.time_steps}")
    print(f"  Максимальная длина меток: {dataset_config.label_max_len}")
    print(f"  Кадров на шаг: {dataset_config.frames_per_step}")
    print(f"  Мел-коэффициентов: {dataset_config.mel_bins}")
    print(f"  Длительность кадра, мс: {dataset_config.frame_duration_ms}")
    if dataset_config.max_audio_duration_s is not None:
        print(f"  Лимит аудио, с: {dataset_config.max_audio_duration_s}")
    print(f"  Перемешивание: {'да' if dataset_config.shuffle else 'нет'}")
    print(f"  Процессов: {dataset_config.max_workers}")
    if phoneme_config is not None:
        print("  Фонемы: да")
        print(f"    Целевая папка: {phoneme_config.dataset_dir}")
        print(f"    Суффикс файлов: {phoneme_config.output_suffix}")
        print(f"    Максимум фонем: {phoneme_config.max_phoneme_len}")
        print(
            "    Использовать лексикон: "
            f"{'да' if phoneme_config.use_lexicon else 'нет'}"
        )
        if phoneme_config.use_lexicon:
            lexicon_label = phoneme_config.lexicon_path or "встроенный CMU лексикон"
            print(f"    Лексикон: {lexicon_label}")
    else:
        print("  Фонемы: нет")
    _print_separator()


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    else:
        config_path = Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME)

    config = _read_config(config_path)

    dataset_config, phoneme_config = merge_parameters(args, config)

    if not os.path.isdir(dataset_config.root_dir):
        raise SystemExit(f"Каталог с исходными данными не найден: {dataset_config.root_dir}")
    os.makedirs(dataset_config.output_dir, exist_ok=True)

    _print_effective_settings(dataset_config, phoneme_config)
    try:
        processed = prepare_ctc_dataset(dataset_config, log=print)
    except DatasetPreparationError as exc:
        raise SystemExit(f"Ошибка подготовки датасета: {exc}") from exc

    if phoneme_config is not None:
        try:
            convert_dataset_to_phonemes(phoneme_config, log=print)
        except DatasetPreparationError as exc:
            raise SystemExit(f"Ошибка конвертации в фонемы: {exc}") from exc

    _print_separator()
    print(f"Успешно обработано {processed} примеров.")
    _print_separator()


if __name__ == "__main__":
    main()
