import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import torch
from torch.utils.data import Dataset

import os
import torch
from torch.utils.data import Dataset
import random

class SpeakerDataset(Dataset):
    def __init__(self, data, max_mel_length=512):
        self.data = data
        self.max_mel_length = max_mel_length
        self.speaker_dict = self._group_by_speaker()

    def _group_by_speaker(self):
        speaker_dict = {}
        for mel_file, speaker in self.data:
            if speaker not in speaker_dict:
                speaker_dict[speaker] = []
            speaker_dict[speaker].append(mel_file)
        return speaker_dict

    def _load_mel(self, mel_file):
        mel = torch.load(mel_file, weights_only=True)
        if mel.shape[1] > self.max_mel_length:
            mel = mel[:, :self.max_mel_length]
        return mel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the anchor sample
        anchor_file, anchor_speaker = self.data[index]
        anchor_mel = self._load_mel(anchor_file)

        # Get a positive sample (same speaker)
        positive_file = random.choice(self.speaker_dict[anchor_speaker])
        positive_mel = self._load_mel(positive_file)

        # Get a negative sample (different speaker)
        negative_speaker = random.choice([spk for spk in self.speaker_dict if spk != anchor_speaker])
        negative_file = random.choice(self.speaker_dict[negative_speaker])
        negative_mel = self._load_mel(negative_file)

        return anchor_mel, positive_mel, negative_mel


def pad_collate(batch):
    """Collate function to pad mel spectrograms to the same length."""
    mel_spectrograms, speakers = zip(*batch)
    max_len = max([mel.shape[1] for mel in mel_spectrograms])
    padded_mels = []
    
    for mel in mel_spectrograms:
        # Add padding to make all spectrograms the same length
        padding = max_len - mel.shape[1]
        if padding > 0:
            mel = torch.nn.functional.pad(mel, (0, padding), "constant", 0)
        padded_mels.append(mel)
    
    return torch.stack(padded_mels), speakers


def process_dataset(audio_dir, mel_dir):
    data = []
    speakers = os.listdir(audio_dir)
    for speaker in speakers:
        speaker_dir = os.path.join(audio_dir, speaker)
        if os.path.isdir(speaker_dir):
            for file_name in os.listdir(speaker_dir):
                if file_name.endswith(".wav"):
                    mel_path = os.path.join(mel_dir, f"{speaker}_{file_name.split('.')[0]}.pt")
                    if os.path.exists(mel_path):
                        data.append((mel_path, speaker))
    if not data:
        print("Нет данных для обработки. Проверьте директории аудио и мел-спектрограмм.")
    return data

import torch

def collate_fn(batch):
    # Определить максимальную длину среди мел-спектрограмм в батче
    max_len = max(item.shape[1] for trio in batch for item in trio)
    
    def pad_tensor(tensor, length):
        if tensor.shape[1] < length:
            padding = torch.zeros(tensor.shape[0], length - tensor.shape[1])
            return torch.cat((tensor, padding), dim=1)
        return tensor

    # Применить padding к каждому из anchor, positive, negative в батче
    padded_batch = []
    for anchor, positive, negative in batch:
        anchor = pad_tensor(anchor, max_len)
        positive = pad_tensor(positive, max_len)
        negative = pad_tensor(negative, max_len)
        padded_batch.append((anchor, positive, negative))

    anchors, positives, negatives = zip(*padded_batch)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
