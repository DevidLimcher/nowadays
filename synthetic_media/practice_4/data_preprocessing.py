import os
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram

# Пути к данным и директории
DATA_DIR = "/home/davidlimcher/projects/nowadays/synthetic_media/practice_4/VCTK-Corpus"
AUDIO_DIR = os.path.join(DATA_DIR, "wav48")
MEL_DIR = os.path.join(DATA_DIR, "mel")

# Создание директории для мел-спектрограмм
os.makedirs(MEL_DIR, exist_ok=True)

def extract_mel_spectrogram(file_path, sample_rate=16000, n_mels=64):
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resample(waveform)
        mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
        return mel_spectrogram.squeeze(0)
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return None

def process_dataset(audio_dir, mel_dir):
    speakers = os.listdir(audio_dir)
    data = []
    for speaker in speakers:
        speaker_path = os.path.join(audio_dir, speaker)
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                audio_file = os.path.join(speaker_path, file)
                mel = extract_mel_spectrogram(audio_file)
                if mel is not None:
                    mel_file = os.path.join(mel_dir, f"{speaker}_{file.split('.')[0]}.pt")
                    torch.save(mel, mel_file)
                    data.append((mel_file, speaker))
    return data
