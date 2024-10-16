import torchaudio
from sklearn.model_selection import train_test_split
from utils import get_speaker_id_from_file
import random
from data_loader import *

# Функция для извлечения мел-спектрограмм
def extract_mel_spectrogram(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=80)(waveform)
    return mel_spectrogram


# Создание триплетов (Anchor, Positive, Negative)
def create_triplets(mel_spectrograms, speaker_labels):
    triplets = []
    speakers = list(set(speaker_labels))  # Уникальные спикеры

    for i in range(len(mel_spectrograms)):
        anchor = mel_spectrograms[i]
        anchor_label = speaker_labels[i]

        # Найти "positive" — другой образец того же спикера
        positive_indices = [idx for idx, label in enumerate(speaker_labels) if label == anchor_label and idx != i]
        if positive_indices:
            positive = mel_spectrograms[random.choice(positive_indices)]
        else:
            continue  # Пропустить, если нет других записей для этого спикера

        # Найти "negative" — образец от другого спикера
        negative_indices = [idx for idx, label in enumerate(speaker_labels) if label != anchor_label]
        negative = mel_spectrograms[random.choice(negative_indices)]

        # Добавляем триплет (anchor, positive, negative)
        triplets.append((anchor, positive, negative))

    return triplets


# Основная функция для подготовки данных
def process_and_split_data(wav_path):
    mel_spectrograms = []
    speaker_labels = []

    audio_files = load_audio_files(wav_path)  # Загрузка аудиофайлов
    
    for audio_file in audio_files:
        mel_spectrogram = extract_mel_spectrogram(audio_file)
        speaker_id = get_speaker_id_from_file(audio_file)
        mel_spectrograms.append(mel_spectrogram)
        speaker_labels.append(speaker_id)

    triplets = create_triplets(mel_spectrograms, speaker_labels)

    # Разделение на обучающий и тестовый наборы
    train_triplets, test_triplets = train_test_split(triplets, test_size=0.2, random_state=42)

    return train_triplets, test_triplets
