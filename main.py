from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from pathlib import Path

import numpy as np
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Для просмотра спектрограмм требуется matplotlib. Установите пакет командой 'pip install matplotlib'."
    ) from exc

from data_processing import (
    DatasetConfig,
    DatasetPreparationError,
    PHONEMES,
    PhonemeConfig,
    convert_dataset_to_phonemes,
    prepare_ctc_dataset,
)
from preparation_data import detokenize_text_with_space_train


PHONEME_INDEX_TO_SYMBOL = {
    idx + 1: (symbol if symbol.strip() else "SPACE")
    for idx, symbol in enumerate(PHONEMES)
}


class DatasetApp(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("LibriSpeech Dataset Builder")
        self.geometry("920x640")
        self.minsize(720, 520)
        self.configure(padx=12, pady=12)

        self.root_dir_var = StringVar()
        default_output = os.path.join(os.getcwd(), "prepared_output")
        self.output_dir_var = StringVar(value=default_output)
        self.time_steps_var = StringVar(value="1000")
        self.label_len_var = StringVar(value="524")
        self.frames_per_step_var = StringVar(value="1")
        self.convert_phoneme_var = BooleanVar(value=False)
        self.shuffle_var = BooleanVar(value=False)
        self.phoneme_len_var = StringVar(value="2000")
        self.mel_bins_var = StringVar(value="128")
        self.frame_duration_var = StringVar(value="10")
        self.dataset_dir_var = StringVar()
        self.dataset_info_var = StringVar(value="Датасет не загружен.")
        self.sample_index_var = StringVar(value="0")
        self.sample_name_var = StringVar()
        self.transcript_var = StringVar()
        self.phonemes_var = StringVar()
        self.viewer_time_steps_var = StringVar(value="1000")
        self.viewer_mel_bins_var = StringVar(value="128")
        self.viewer_frame_duration_var = StringVar(value="")

        self._settings_path = Path(__file__).resolve().with_name("dataset_app_settings.json")
        self._save_job: str | None = None
        self._suspend_settings_save = False

        self._tracked_string_vars: list[StringVar] = [
            self.root_dir_var,
            self.output_dir_var,
            self.time_steps_var,
            self.label_len_var,
            self.frames_per_step_var,
            self.frame_duration_var,
            self.mel_bins_var,
            self.phoneme_len_var,
            self.dataset_dir_var,
            self.viewer_time_steps_var,
            self.viewer_mel_bins_var,
            self.viewer_frame_duration_var,
        ]
        self._tracked_bool_vars: list[BooleanVar] = [self.convert_phoneme_var, self.shuffle_var]

        self._load_settings()

        for var in [*self._tracked_string_vars, *self._tracked_bool_vars]:
            var.trace_add("write", self._on_setting_changed)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.log_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.loaded_dataset: dict[str, np.memmap] | None = None
        self.loaded_lengths: np.ndarray | None = None

        self.figure = Figure(figsize=(4.5, 3.5), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Нет данных")
        self.axes.axis("off")
        self.canvas: FigureCanvasTkAgg | None = None

        self._build_form()
        self._build_log_panel()
        self._poll_log_queue()

    # ------------------------------------------------------------------ GUI
    def _build_form(self) -> None:
        form = ttk.Frame(self)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Корневая папка LibriSpeech:").grid(row=0, column=0, sticky="w", pady=(0, 6))
        root_entry = ttk.Entry(form, textvariable=self.root_dir_var)
        root_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Button(form, text="Выбрать...", command=self._choose_root_dir).grid(row=0, column=2, pady=(0, 6))

        ttk.Label(form, text="Папка для сохранения результата:").grid(row=1, column=0, sticky="w", pady=(0, 6))
        out_entry = ttk.Entry(form, textvariable=self.output_dir_var)
        out_entry.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Button(form, text="Выбрать...", command=self._choose_output_dir).grid(row=1, column=2, pady=(0, 6))

        ttk.Label(form, text="Размер окна (временные шаги):").grid(row=2, column=0, sticky="w", pady=(0, 6))
        ttk.Entry(form, textvariable=self.time_steps_var, width=12).grid(row=2, column=1, sticky="w", pady=(0, 6))

        length_frame = ttk.LabelFrame(form, text="Ограничения длины")
        length_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        length_frame.columnconfigure(1, weight=1)

        ttk.Label(length_frame, text="Максимальная длина метки:").grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 0))
        ttk.Entry(length_frame, textvariable=self.label_len_var, width=12).grid(row=0, column=1, sticky="w", pady=(6, 0))

        ttk.Label(length_frame, text="Кадров на шаг модели:").grid(row=1, column=0, sticky="w", padx=(6, 6), pady=(6, 0))
        ttk.Entry(length_frame, textvariable=self.frames_per_step_var, width=12).grid(row=1, column=1, sticky="w", pady=(6, 0))

        options_row = ttk.Frame(length_frame)
        options_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=(6, 6), pady=(8, 0))

        phoneme_check = ttk.Checkbutton(
            options_row,
            text="Сохранить последовательности фонем",
            variable=self.convert_phoneme_var,
            command=self._toggle_phoneme_length_state,
        )
        phoneme_check.pack(side="left")

        shuffle_check = ttk.Checkbutton(
            options_row,
            text="Перемешать примеры",
            variable=self.shuffle_var,
        )
        shuffle_check.pack(side="left", padx=(16, 0))

        ttk.Label(length_frame, text="Максимальная длина фонем:").grid(row=3, column=0, sticky="w", padx=(6, 6), pady=(6, 6))
        self.phoneme_len_entry = ttk.Entry(length_frame, textvariable=self.phoneme_len_var, width=12)
        self.phoneme_len_entry.grid(row=3, column=1, sticky="w", pady=(6, 6))
        self._toggle_phoneme_length_state()

        mel_frame = ttk.LabelFrame(form, text="Параметры мел-спектрограммы")
        mel_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 6))
        mel_frame.columnconfigure(1, weight=1)
        mel_frame.columnconfigure(3, weight=1)

        ttk.Label(mel_frame, text="Количество мел-коэффициентов:").grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 6))
        ttk.Entry(mel_frame, textvariable=self.mel_bins_var, width=12).grid(row=0, column=1, sticky="w", pady=(6, 6))

        ttk.Label(mel_frame, text="Длительность кадра (мс):").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=(6, 6))
        ttk.Entry(mel_frame, textvariable=self.frame_duration_var, width=12).grid(row=0, column=3, sticky="w", pady=(6, 6))

        self.start_button = ttk.Button(form, text="Запустить обработку", command=self._start_processing)
        self.start_button.grid(row=5, column=0, columnspan=3, pady=(18, 0), sticky="ew")

        self.progress_bar = ttk.Progressbar(form, mode="indeterminate")
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(12, 0))

    def _build_log_panel(self) -> None:
        log_frame = ttk.Frame(self)
        log_frame.grid(row=0, column=1, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        ttk.Label(log_frame, text="Журнал выполнения:").pack(anchor="w")
        self.log_text = ScrolledText(log_frame, state="disabled", wrap="word", height=8)
        self.log_text.pack(fill="both", expand=False, pady=(6, 0))

        ttk.Separator(log_frame, orient="horizontal").pack(fill="x", pady=8)
        self._build_dataset_viewer(log_frame)

    def _build_dataset_viewer(self, parent: ttk.Frame) -> None:
        viewer = ttk.Labelframe(parent, text="Просмотр подготовленных данных")
        viewer.pack(fill="both", expand=True)

        row1 = ttk.Frame(viewer)
        row1.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(row1, text="Папка с данными:").pack(side="left")
        entry = ttk.Entry(row1, textvariable=self.dataset_dir_var)
        entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        ttk.Button(row1, text="Выбрать...", command=self._choose_dataset_dir).pack(side="left")
        ttk.Button(row1, text="Использовать выход", command=self._use_output_dir_for_viewer).pack(side="left", padx=(6, 0))

        info = ttk.Label(viewer, textvariable=self.dataset_info_var)
        info.pack(anchor="w", padx=8, pady=(6, 0))

        controls = ttk.Frame(viewer)
        controls.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(controls, text="Индекс образца:").pack(side="left")
        self.sample_spin = ttk.Spinbox(
            controls,
            from_=0,
            to=0,
            textvariable=self.sample_index_var,
            width=8,
            command=self._on_sample_change,
        )
        self.sample_spin.pack(side="left", padx=(6, 6))
        self.sample_spin.configure(state="disabled")
        ttk.Button(controls, text="Показать", command=self._on_sample_change).pack(side="left")
        
        # Параметры отображения
        ttk.Label(controls, text="Шаги времени:").pack(side="left", padx=(20, 6))
        time_entry = ttk.Entry(controls, textvariable=self.viewer_time_steps_var, width=6)
        time_entry.pack(side="left", padx=(0, 10))

        ttk.Label(controls, text="Мел-коэф.:").pack(side="left")
        mel_entry = ttk.Entry(controls, textvariable=self.viewer_mel_bins_var, width=6)
        mel_entry.pack(side="left", padx=(6, 0))

        ttk.Label(controls, text="Кадр, мс:").pack(side="left", padx=(12, 6))
        self.viewer_frame_duration_var_entry = ttk.Entry(
            controls,
            textvariable=self.viewer_frame_duration_var,
            width=8,
            state="readonly",
        )
        self.viewer_frame_duration_var_entry.pack(side="left")

        canvas_frame = ttk.Frame(viewer)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Текстовая информация под спектрограммой
        text_frame = ttk.Frame(viewer)
        text_frame.pack(fill="x", expand=False, padx=8, pady=(4, 8))

        ttk.Label(text_frame, text="Имя образца:").pack(anchor="w")
        self.sample_name_label = ttk.Label(text_frame, textvariable=self.sample_name_var)
        self.sample_name_label.pack(anchor="w", pady=(0, 4))

        ttk.Label(text_frame, text="Транскрипт:").pack(anchor="w")
        self.transcript_text = ScrolledText(text_frame, height=2, wrap="word", state="disabled")
        self.transcript_text.pack(fill="x", expand=False, pady=(2, 4))

        ttk.Label(text_frame, text="Фонемы:").pack(anchor="w", pady=(6, 0))
        self.phonemes_text = ScrolledText(text_frame, height=3, wrap="word", state="disabled")
        self.phonemes_text.pack(fill="x", expand=False, pady=(2, 0))

    def _choose_dataset_dir(self) -> None:
        selected = filedialog.askdirectory(title="Выберите папку с подготовленными данными")
        if selected:
            self.dataset_dir_var.set(selected)
            self._load_dataset(selected)

    def _use_output_dir_for_viewer(self) -> None:
        target = self.output_dir_var.get().strip()
        if target:
            self.dataset_dir_var.set(target)
            self._load_dataset(target)

    def _load_dataset(self, directory: str, silent: bool = False) -> None:
        directory = directory.strip()
        if not directory or not os.path.isdir(directory):
            if not silent:
                messagebox.showerror("Просмотр", "Указанная папка не найдена.")
            self._clear_dataset_view()
            return

        x_path = os.path.join(directory, "x_data.npy")
        x_len_path = os.path.join(directory, "x_input_length.npy")
        if not os.path.exists(x_path) or not os.path.exists(x_len_path):
            if not silent:
                messagebox.showerror(
                    "Просмотр",
                    "В выбранной папке нет файлов x_data.npy и x_input_length.npy."
                )
            self._clear_dataset_view()
            return

        try:
            x_data = np.load(x_path, mmap_mode="r")
            x_lengths = np.load(x_len_path, mmap_mode="r")
        except Exception as exc:  # pragma: no cover - I/O failure
            if not silent:
                messagebox.showerror("Просмотр", f"Не удалось открыть файлы: {exc}")
            self._clear_dataset_view()
            return

        sample_count = x_data.shape[0]
        feature_dim = x_data.shape[2] if x_data.ndim == 3 else 0
        time_steps = x_data.shape[1] if x_data.ndim == 3 else 0

        self.loaded_dataset = {"x_data": x_data}
        self.loaded_lengths = x_lengths

        names_path = os.path.join(directory, "xy_sample_names.npy")
        if os.path.exists(names_path):
            try:
                self.loaded_dataset["sample_names"] = np.load(names_path, mmap_mode="r")
            except Exception as exc:
                self._log(f"Не удалось загрузить xy_sample_names.npy: {exc}")

        metadata_path = os.path.join(directory, "metadata.json")
        frame_duration_ms = None
        hop_length_samples: float | None = None
        shuffle_flag: bool | None = None
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as meta_file:
                    metadata = json.load(meta_file)
            except Exception as exc:
                self._log(f"Не удалось прочитать metadata.json: {exc}")
            else:
                if isinstance(metadata, dict):
                    value = metadata.get("frame_duration_ms")
                    if value is not None:
                        try:
                            frame_duration_ms = float(value)
                        except (TypeError, ValueError) as exc:
                            self._log(
                                f"Неверное значение frame_duration_ms в metadata.json: {value!r} ({exc})"
                            )
                    params = metadata.get("spectrogram_params")
                    if isinstance(params, dict):
                        eff = params.get("effective_frame_duration_ms")
                        if eff is not None:
                            try:
                                frame_duration_ms = float(eff)
                            except (TypeError, ValueError) as exc:
                                self._log(
                                    f"Неверное значение effective_frame_duration_ms: {eff!r} ({exc})"
                                )
                        hop_val = params.get("hop_length") if isinstance(params, dict) else None
                        if hop_val is not None:
                            try:
                                hop_length_samples = float(hop_val)
                            except (TypeError, ValueError):
                                hop_length_samples = None
                    if "shuffle" in metadata:
                        try:
                            shuffle_flag = bool(metadata["shuffle"])
                        except Exception:
                            shuffle_flag = None
                else:
                    self._log("metadata.json имеет некорректный формат.")

        # Пытаемся загрузить токенизированные подписи и их длины
        y_label_path = os.path.join(directory, "y_labels.npy")
        y_len_path = os.path.join(directory, "y_label_length.npy")
        if os.path.exists(y_label_path) and os.path.exists(y_len_path):
            try:
                self.loaded_dataset["labels"] = np.load(y_label_path, mmap_mode="r")
                self.loaded_dataset["label_lengths"] = np.load(y_len_path, mmap_mode="r")
            except Exception as exc:
                self._log(f"Не удалось загрузить y_labels: {exc}")

        # Ищем файлы c фонемами (например, y_labels_phoneme.npy)
        phoneme_label_path = None
        phoneme_len_path = None
        for name in sorted(os.listdir(directory)):
            if name.startswith("y_labels") and "phoneme" in name:
                candidate_label = os.path.join(directory, name)
                candidate_len = os.path.join(
                    directory,
                    name.replace("y_labels", "y_label_length", 1),
                )
                if os.path.exists(candidate_len):
                    phoneme_label_path = candidate_label
                    phoneme_len_path = candidate_len
                    break

        if phoneme_label_path and phoneme_len_path:
            try:
                self.loaded_dataset["phoneme_labels"] = np.load(phoneme_label_path, mmap_mode="r")
                self.loaded_dataset["phoneme_lengths"] = np.load(phoneme_len_path, mmap_mode="r")
            except Exception as exc:
                self._log(f"Не удалось загрузить фонемы: {exc}")
        if sample_count > 0:
            self.sample_spin.configure(state="normal", from_=0, to=sample_count - 1)
            self.sample_index_var.set("0")
        else:
            self.sample_spin.configure(state="disabled", from_=0, to=0)
            self.sample_index_var.set("0")

        if time_steps:
            self.viewer_time_steps_var.set(str(time_steps))
        else:
            self.viewer_time_steps_var.set("")

        if feature_dim:
            self.viewer_mel_bins_var.set(str(feature_dim))
        else:
            self.viewer_mel_bins_var.set("")

        if frame_duration_ms is not None:
            formatted_frame = f"{frame_duration_ms:.3f}".rstrip("0").rstrip(".")
            self.viewer_frame_duration_var.set(formatted_frame)
        else:
            fallback = self.frame_duration_var.get().strip()
            self.viewer_frame_duration_var.set(fallback)

        frame_info = (
            f"{frame_duration_ms:.3f}".rstrip("0").rstrip(".")
            if frame_duration_ms is not None
            else self.frame_duration_var.get().strip()
        )
        if not frame_info:
            frame_info = "-"

        hop_suffix = ""
        if hop_length_samples is not None:
            hop_suffix = f" | Шаг: {int(hop_length_samples)} сэмп."

        info = (
            f"Файлы: {os.path.basename(x_path)}, {os.path.basename(x_len_path)}\n"
            f"Примеров: {sample_count} | Окно: {time_steps}x{feature_dim} | Кадр: {frame_info} мс{hop_suffix}"
        )
        if shuffle_flag is not None:
            shuffle_text = "да" if shuffle_flag else "нет"
            info += f" | Перемешано: {shuffle_text}"
        if "sample_names" in self.loaded_dataset and os.path.exists(names_path):
            info += f"\nИмена: {os.path.basename(names_path)}"
        self.dataset_info_var.set(info)
        self._log(f"Загружен датасет из {directory}")

        self._render_sample(0)

    def _clear_dataset_view(self) -> None:
        self.loaded_dataset = None
        self.loaded_lengths = None
        self.sample_spin.configure(state="disabled", from_=0, to=0)
        self.sample_index_var.set("0")
        self.dataset_info_var.set("Датасет не загружен.")
        self.viewer_time_steps_var.set("")
        self.viewer_mel_bins_var.set("")
        self.viewer_frame_duration_var.set("")
        self.sample_name_var.set("")
        self.transcript_var.set("")
        self.phonemes_var.set("")
        if hasattr(self, "transcript_text"):
            self._set_text_widget(self.transcript_text, "")
        if hasattr(self, "phonemes_text"):
            self._set_text_widget(self.phonemes_text, "")
        self.axes.clear()
        self.axes.set_title("Нет данных")
        self.axes.axis("off")
        if self.canvas:
            self.canvas.draw_idle()

    def _on_sample_change(self) -> None:
        if not self.loaded_dataset:
            return
        try:
            index = int(self.sample_index_var.get())
        except ValueError:
            index = 0
        x_data = self.loaded_dataset["x_data"]
        max_index = x_data.shape[0] - 1
        if max_index < 0:
            return
        index = max(0, min(index, max_index))
        self.sample_index_var.set(str(index))
        self._render_sample(index)

    def _render_sample(self, index: int) -> None:
        if not self.loaded_dataset:
            return
        x_data = self.loaded_dataset["x_data"]
        if index < 0 or index >= x_data.shape[0]:
            return
        spectrogram = x_data[index]
        if spectrogram.ndim != 2:
            return
        length = None
        if self.loaded_lengths is not None and index < self.loaded_lengths.shape[0]:
            length = int(self.loaded_lengths[index])
        if length is None or length <= 0 or length > spectrogram.shape[0]:
            length = spectrogram.shape[0]
        trimmed = spectrogram[:length, :]

        self.axes.clear()
        self.axes.imshow(trimmed.T, aspect="auto", origin="lower", interpolation="nearest", cmap="magma")
        self.axes.set_xlabel("Временные шаги")
        self.axes.set_ylabel("Мел-коэффициенты")
        title = f"Образец {index} | длина {length}"
        sample_names = self.loaded_dataset.get("sample_names")
        if sample_names is not None and 0 <= index < sample_names.shape[0]:
            raw_name = sample_names[index]
            try:
                decoded = raw_name.decode("utf-8", "ignore").strip()
            except AttributeError:
                decoded = str(raw_name).strip()
            if decoded:
                title += f" | {decoded}"
        self.axes.set_title(title)
        self.figure.tight_layout()
        if self.canvas:
            self.canvas.draw_idle()
            
        # Обновляем текстовую информацию
        self._update_text_info(index)

    def _update_text_info(self, index: int) -> None:
        """Обновляет отображение транскрипта и фонем для указанного образца"""
        if not self.loaded_dataset:
            self.sample_name_var.set("Датасет не загружен")
            self.transcript_var.set("Датасет не загружен")
            self.phonemes_var.set("")
            if hasattr(self, "transcript_text"):
                self._set_text_widget(self.transcript_text, "Датасет не загружен")
            if hasattr(self, "phonemes_text"):
                self._set_text_widget(self.phonemes_text, "")
            return

        sample_names = self.loaded_dataset.get("sample_names")
        if sample_names is not None and 0 <= index < sample_names.shape[0]:
            raw_name = sample_names[index]
            try:
                decoded = raw_name.decode("utf-8", "ignore").strip()
            except AttributeError:
                decoded = str(raw_name).strip()
            self.sample_name_var.set(decoded or "(пусто)")
        elif sample_names is None:
            self.sample_name_var.set("xy_sample_names.npy не найден")
        else:
            self.sample_name_var.set("(нет имени)")

        labels = self.loaded_dataset.get("labels")
        label_lengths = self.loaded_dataset.get("label_lengths")
        if labels is not None and label_lengths is not None and index < len(label_lengths):
            try:
                length = int(label_lengths[index])
                tokens = labels[index, :length]
                text = detokenize_text_with_space_train(tokens)
                self.transcript_var.set(text or "(пусто)")
                if hasattr(self, "transcript_text"):
                    self._set_text_widget(self.transcript_text, text or "(пусто)")
            except Exception as exc:
                self.transcript_var.set(f"Ошибка чтения транскрипта: {exc}")
                if hasattr(self, "transcript_text"):
                    self._set_text_widget(self.transcript_text, f"Ошибка чтения транскрипта: {exc}")
        elif labels is None:
            self.transcript_var.set("Файл y_labels.npy не найден")
            if hasattr(self, "transcript_text"):
                self._set_text_widget(self.transcript_text, "Файл y_labels.npy не найден")
        else:
            self.transcript_var.set("Нет транскрипта для этого образца")
            if hasattr(self, "transcript_text"):
                self._set_text_widget(self.transcript_text, "Нет транскрипта для этого образца")

        phoneme_labels = self.loaded_dataset.get("phoneme_labels")
        phoneme_lengths = self.loaded_dataset.get("phoneme_lengths")
        if phoneme_labels is not None and phoneme_lengths is not None and index < len(phoneme_lengths):
            try:
                length = int(phoneme_lengths[index])
                seq = phoneme_labels[index, :length]
                decoded = []
                for value in seq:
                    token = int(value)
                    if token <= 0:
                        continue
                    symbol = PHONEME_INDEX_TO_SYMBOL.get(token)
                    if symbol is None:
                        decoded.append(f"?{token}")
                    else:
                        decoded.append(symbol)
                self.phonemes_var.set(" ".join(decoded) or "(пусто)")
                if hasattr(self, "phonemes_text"):
                    self._set_text_widget(self.phonemes_text, " ".join(decoded) or "(пусто)")
            except Exception as exc:
                self.phonemes_var.set(f"Ошибка чтения фонем: {exc}")
                if hasattr(self, "phonemes_text"):
                    self._set_text_widget(self.phonemes_text, f"Ошибка чтения фонем: {exc}")
        elif phoneme_labels is None:
            self.phonemes_var.set("Файл фонем не найден")
            if hasattr(self, "phonemes_text"):
                self._set_text_widget(self.phonemes_text, "Файл фонем не найден")
        else:
            self.phonemes_var.set("Нет фонем для этого образца")
            if hasattr(self, "phonemes_text"):
                self._set_text_widget(self.phonemes_text, "Нет фонем для этого образца")

    def _set_text_widget(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    # ----------------------------------------------------------- settings
    def _on_setting_changed(self, *_: object) -> None:
        if self._suspend_settings_save:
            return
        self._schedule_settings_save()

    def _schedule_settings_save(self) -> None:
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
        self._save_job = self.after(500, self._save_settings)

    def _settings_data(self) -> dict[str, object]:
        return {
            "root_dir": self.root_dir_var.get().strip(),
            "output_dir": self.output_dir_var.get().strip(),
            "time_steps": self.time_steps_var.get().strip(),
            "label_max_len": self.label_len_var.get().strip(),
            "frames_per_step": self.frames_per_step_var.get().strip(),
            "frame_duration_ms": self.frame_duration_var.get().strip(),
            "mel_bins": self.mel_bins_var.get().strip(),
            "convert_phoneme": bool(self.convert_phoneme_var.get()),
            "shuffle_examples": bool(self.shuffle_var.get()),
            "phoneme_max_len": self.phoneme_len_var.get().strip(),
            "dataset_dir": self.dataset_dir_var.get().strip(),
            "viewer_time_steps": self.viewer_time_steps_var.get().strip(),
            "viewer_mel_bins": self.viewer_mel_bins_var.get().strip(),
            "viewer_frame_duration": self.viewer_frame_duration_var.get().strip(),
        }

    def _save_settings(self) -> None:
        self._save_job = None
        data = self._settings_data()
        try:
            self._settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - best effort only
            print(f"Не удалось сохранить настройки: {exc}")

    def _load_settings(self) -> None:
        if not self._settings_path.exists():
            return
        try:
            raw = self._settings_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:  # pragma: no cover - best effort only
            print(f"Не удалось прочитать настройки: {exc}")
            return

        self._suspend_settings_save = True
        try:
            def set_str(var: StringVar, key: str) -> None:
                value = data.get(key)
                if value is not None:
                    var.set(str(value))

            set_str(self.root_dir_var, "root_dir")
            set_str(self.output_dir_var, "output_dir")
            set_str(self.time_steps_var, "time_steps")
            set_str(self.label_len_var, "label_max_len")
            set_str(self.frames_per_step_var, "frames_per_step")
            set_str(self.frame_duration_var, "frame_duration_ms")
            set_str(self.mel_bins_var, "mel_bins")
            set_str(self.phoneme_len_var, "phoneme_max_len")
            set_str(self.dataset_dir_var, "dataset_dir")
            set_str(self.viewer_time_steps_var, "viewer_time_steps")
            set_str(self.viewer_mel_bins_var, "viewer_mel_bins")
            set_str(self.viewer_frame_duration_var, "viewer_frame_duration")

            if "convert_phoneme" in data:
                self.convert_phoneme_var.set(bool(data["convert_phoneme"]))
            if "shuffle_examples" in data:
                self.shuffle_var.set(bool(data["shuffle_examples"]))
        finally:
            self._suspend_settings_save = False

    def _on_close(self) -> None:
        if self._save_job is not None:
            try:
                self.after_cancel(self._save_job)
            except Exception:
                pass
            self._save_job = None
        self._save_settings()
        self.destroy()

    # ----------------------------------------------------------- event utils
    def _toggle_phoneme_length_state(self) -> None:
        state = "normal" if self.convert_phoneme_var.get() else "disabled"
        self.phoneme_len_entry.configure(state=state)

    def _cleanup_output_dir(self, output_dir: str) -> None:
        try:
            entries = os.listdir(output_dir)
        except FileNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - best effort only
            self._log(f"Не удалось перечислить файлы в {output_dir}: {exc}")
            return

        exact_files = {
            "x_data.npy",
            "x_input_length.npy",
            "y_labels.npy",
            "y_label_length.npy",
            "xy_sample_names.npy",
            "metadata.json",
        }
        prefix_files = ("y_labels", "y_label_length")
        removed: list[str] = []

        for name in entries:
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path):
                continue
            should_remove = name in exact_files or (
                name.endswith(".npy") and any(name.startswith(prefix) for prefix in prefix_files)
            )
            if not should_remove:
                continue
            try:
                os.remove(path)
                removed.append(name)
            except Exception as exc:  # pragma: no cover - best effort only
                self._log(f"Не удалось удалить {name}: {exc}")

        if removed:
            removed.sort()
            self._log("Удалены старые файлы: " + ", ".join(removed))

    def _choose_root_dir(self) -> None:
        selected = filedialog.askdirectory(title="Выберите корневую папку LibriSpeech")
        if selected:
            self.root_dir_var.set(selected)

    def _choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Выберите папку для сохранения")
        if selected:
            self.output_dir_var.set(selected)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _log(self, message: str) -> None:
        self.log_queue.put(lambda msg=message: self._append_log(msg))

    def _poll_log_queue(self) -> None:
        try:
            while True:
                callback = self.log_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_log_queue)

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Обработка", "Процесс уже запущен.")
            return

        root_dir = self.root_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        try:
            time_steps = int(self.time_steps_var.get())
            label_max = int(self.label_len_var.get())
            frames_per_step = int(self.frames_per_step_var.get())
            mel_bins = int(self.mel_bins_var.get())
            frame_duration = float(self.frame_duration_var.get())
            phoneme_max = int(self.phoneme_len_var.get()) if self.convert_phoneme_var.get() else None
        except ValueError:
            messagebox.showerror("Ошибка", "Числовые параметры должны быть корректными значениями.")
            return

        if (
            time_steps <= 0
            or label_max <= 0
            or frames_per_step <= 0
            or mel_bins <= 0
            or frame_duration <= 0
            or (phoneme_max is not None and phoneme_max <= 0)
        ):
            messagebox.showerror("Ошибка", "Числовые параметры должны быть больше нуля.")
            return

        if not root_dir or not os.path.isdir(root_dir):
            messagebox.showerror("Ошибка", "Укажите существующую корневую папку LibriSpeech.")
            return
        if not output_dir:
            messagebox.showerror("Ошибка", "Укажите папку для сохранения данных.")
            return

        os.makedirs(output_dir, exist_ok=True)
        self._cleanup_output_dir(output_dir)
        self._save_settings()

        dataset_config = DatasetConfig(
            root_dir=root_dir,
            output_dir=output_dir,
            time_steps=time_steps,
            label_max_len=label_max,
            mel_bins=mel_bins,
            frame_duration_ms=frame_duration,
            frames_per_step=frames_per_step,
            shuffle=self.shuffle_var.get(),
        )

        phoneme_config = (
            PhonemeConfig(dataset_dir=output_dir, max_phoneme_len=phoneme_max or 2000)
            if self.convert_phoneme_var.get()
            else None
        )

        self.start_button.configure(state="disabled")
        self.progress_bar.start(10)
        self._log("Запуск обработки...")

        def worker() -> None:
            try:
                prepare_ctc_dataset(dataset_config, log=self._log)
                if phoneme_config:
                    self._log("Старт конвертации в фонемы...")
                    convert_dataset_to_phonemes(phoneme_config, log=self._log)
                self.log_queue.put(lambda path=output_dir: self._load_dataset(path, silent=True))
                self._log("Обработка завершена успешно.")
            except DatasetPreparationError as exc:
                self._log(f"Ошибка: {exc}")
                self.log_queue.put(lambda: messagebox.showerror("Ошибка обработки", str(exc)))
            except Exception as exc:  # pragma: no cover - unexpected
                message = str(exc)
                self._log(f"Непредвиденная ошибка: {message}")
                self.log_queue.put(lambda msg=message: messagebox.showerror("Критическая ошибка", msg))
            finally:
                self.log_queue.put(self._on_worker_finished)

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _on_worker_finished(self) -> None:
        self.progress_bar.stop()
        self.start_button.configure(state="normal")
        self.worker = None


def main() -> None:
    app = DatasetApp()
    app.mainloop()


if __name__ == "__main__":
    main()
