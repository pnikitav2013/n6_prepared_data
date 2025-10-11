from __future__ import annotations

import glob
import os
import re
import sys
import time
from typing import Tuple

import librosa
import numpy as np
import sounddevice as sd


def _resolve_spectrogram_params(
    sr: int,
    frame_duration_ms: float | None,
    mel_bins: int,
) -> Tuple[int | None, int | None, int | None]:
    """Calculate hop_length, win_length, and n_fft for librosa based on frame duration."""

    if not frame_duration_ms or frame_duration_ms <= 0:
        return None, None, None

    hop_length = max(1, int(round(sr * frame_duration_ms / 1000.0)))

    # Use a window roughly twice the hop (≈50% overlap) but keep it reasonable.
    win_length = min(sr, max(hop_length * 2, hop_length))

    n_fft = 1
    while n_fft < win_length:
        n_fft <<= 1

    # Ensure there are enough frequency bins to support the requested mel resolution.
    min_fft = max(512, mel_bins * 2)
    while n_fft < min_fft:
        n_fft <<= 1

    return hop_length, win_length, n_fft


def process_audio_and_text_nornalize(
    audio_path,
    sentence,
    mel_bins: int = 128,
    frame_duration_ms: float | None = None,
    return_params: bool = False,
):
    """Load audio, compute mel spectrogram, tokenize transcript, and normalize."""

    audio, sr = librosa.load(audio_path, sr=None)

    hop_length, win_length, n_fft = _resolve_spectrogram_params(sr, frame_duration_ms, mel_bins)
    spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=mel_bins,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        center=False,
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram = np.nan_to_num(spectrogram, nan=-80.0, neginf=-80.0, posinf=0.0)

    tokenized_sentence = tokenize_text_with_space(sentence)

    spectrogram = spectrogram.transpose(1, 0)
    spec_min = spectrogram.min()
    spec_max = spectrogram.max()
    if np.isclose(spec_max, spec_min):
        spectrogram = np.zeros_like(spectrogram)
    else:
        spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)

    if return_params:
        params: dict[str, float] = {}
        if hop_length:
            params["hop_length"] = float(hop_length)
            params["effective_frame_duration_ms"] = float(hop_length / sr * 1000.0)
        if win_length:
            params["win_length"] = float(win_length)
            params["window_duration_ms"] = float(win_length / sr * 1000.0)
        if n_fft:
            params["n_fft"] = float(n_fft)
        if frame_duration_ms is not None:
            params["requested_frame_duration_ms"] = float(frame_duration_ms)
        return spectrogram, tokenized_sentence, params

    return spectrogram, tokenized_sentence


def process_audio_and_record():
    duration = 1  # seconds
    sr = 16000  # Sample rate
    while True:
        # Запись аудио с микрофона
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
        sd.wait()  # Wait until recording is finished

        # Преобразование аудио в спектрограмму
        spectrogram = librosa.feature.melspectrogram(y=np.squeeze(audio), sr=sr, center=False)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram = np.nan_to_num(spectrogram, nan=-80.0, neginf=-80.0, posinf=0.0)
        spectrogram = spectrogram.transpose(1, 0)
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        if np.isclose(spec_max, spec_min):
            spectrogram = np.zeros_like(spectrogram)
        else:
            spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)

        yield spectrogram

        # Задержка в 0.5 секунды
        time.sleep(1)


def extract_pairs_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    pattern = re.compile(r'(\d+-\d+-\d{4}) (.*)')
    pairs = pattern.findall(text)
    return pairs

def tokenize_text_with_space(text):
    # Словарь для отображения символов и пробела в индексы
    char_to_index = {char: index for index, char in enumerate(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'", 1)}
    
    # Токенизация текста
    tokenized_text = []
    for char in text:
        if char in char_to_index:
            tokenized_text.append(char_to_index[char])
        else:
            print(f"Error: Character '{char}' is not in the list")
            sys.exit(1)
    
    return tokenized_text


def detokenize_text_with_space(tokenized_text):
    # Словарь для отображения индексов в символы и пробел
    index_to_char = {index: char for index, char in enumerate(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'", 1)}
    
    # Детокенизация текста
    detokenized_text = []
    for subarray in tokenized_text:
        for token in subarray:
            # Преобразование numpy.ndarray в обычное число
            token = int(token)
            if token in index_to_char:
                detokenized_text.append(index_to_char[token])
            else:
                print(f"Error: Index '{token}' is not in the list")
                sys.exit(1)
    
    return ''.join(detokenized_text)

def detokenize_text_with_space_train(tokenized_text):
    # Dictionary for mapping indices to characters and space
    index_to_char = {index: char for index, char in enumerate(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'", 1)}
    
    # Detokenize the text
    detokenized_text = []
    for token in tokenized_text:
        # Convert numpy.ndarray to a regular number
        token = int(token)
        if token == 0:
            break
        elif token in index_to_char:
            detokenized_text.append(index_to_char[token])
        else:
            print(f"Error: Index '{token}' is not in the list")
            sys.exit(1)
    
    return ''.join(detokenized_text).lower()

class DataTrain:

    def __init__(self):
        self.x_data = []
        self.max_length = 0
        self.x_input_length = []
        self.max_label_length = 0
        self.y_label_length = []
        self.y_labels = []


    def add_data(self, spectrogram, tokenized_sentence):
        
        
        self.x_input_length.append(spectrogram.shape[0])
        
        if len(self.x_data) == 0:
            self.x_data = [spectrogram]
            self.max_length = spectrogram.shape[0]
        else:
            new_max_length = max(self.max_length, spectrogram.shape[0])
            if new_max_length > self.max_length:
                # Расширяем все предыдущие спектрограммы до новой максимальной длины
                self.x_data = [np.pad(x, ((0, new_max_length - x.shape[0]), (0, 0)), 'constant') for x in self.x_data]
                self.max_length = new_max_length

            # Теперь расширяем текущую спектрограмму, если нужно
            if spectrogram.shape[0] < self.max_length:
                spectrogram = np.pad(spectrogram, ((0, self.max_length - spectrogram.shape[0]), (0, 0)), 'constant')

            self.x_data.append(spectrogram)
            
        # print(np.array(self.x_data).shape)
        
        self.y_label_length.append(len(tokenized_sentence))
        if len(self.y_labels) == 0:
            self.y_labels = [tokenized_sentence]
            self.max_label_length = len(tokenized_sentence)
        else:
            new_max_label_length = max(self.max_label_length, len(tokenized_sentence))
            if new_max_label_length > self.max_label_length:
                self.y_labels = [np.pad(y, (0, new_max_label_length - len(y)), 'constant') for y in self.y_labels]
                self.max_label_length = new_max_label_length
            if len(tokenized_sentence) < self.max_label_length:
                tokenized_sentence = np.pad(tokenized_sentence, (0, self.max_label_length - len(tokenized_sentence)), 'constant')
            self.y_labels.append(tokenized_sentence)

        # print(np.array(self.x_data).shape, np.array(self.y_labels).shape)
        
        
        
    def get_x_input_length(self):
        return self.x_input_length
    
    def get_y_label_length(self):
        return self.y_label_length
    
    def get_x_data(self):
        return self.x_data
    
    def get_y_labels(self):
        return self.y_labels   
    
    def __del__(self):
        print("DataTrain object is being deleted")
        
        

# x_data = np.random.uniform(0.999, 1.0, size=(32, 1900, 128))  # Входные аудиодорожки
# y_labels = np.random.randint(8, 9, size=(32, 50))


# x_input_length = np.full((32, 1), 1900)  # Все аудиодорожки имеют длину 540
# y_label_length = np.random.randint(40, 45, size=(32, 1))  # Длины меток