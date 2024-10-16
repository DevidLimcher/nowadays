import os
import pandas as pd

# Загрузка информации о спикерах
def get_speaker_info(speaker_info_path):
    speaker_info = pd.read_csv(speaker_info_path, delim_whitespace=True)
    return speaker_info

# Загрузка всех аудиофайлов из структуры папок
def load_audio_files(wav_path):
    audio_files = []
    for speaker_dir in os.listdir(wav_path):
        speaker_dir_path = os.path.join(wav_path, speaker_dir)
        if os.path.isdir(speaker_dir_path):
            for audio_file in os.listdir(speaker_dir_path):
                if audio_file.endswith('.wav'):
                    audio_files.append(os.path.join(speaker_dir_path, audio_file))
    return audio_files
