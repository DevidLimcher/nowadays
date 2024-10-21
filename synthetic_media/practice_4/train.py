import os
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import SpeakerDataset, process_dataset, collate_fn
from model import CNNEncoder
from triplet_utils import triplet_loss, cosine_similarity
import wandb

# Initialize Weights & Biases
wandb.init(project="speaker-verification", entity="makmafi-south-ural-state-university")

# Paths
DATA_DIR = "/home/davidlimcher/projects/nowadays/synthetic_media/practice_4/VCTK-Corpus"
AUDIO_DIR = os.path.join(DATA_DIR, "wav48")
MEL_DIR = os.path.join(DATA_DIR, "mel")
os.makedirs(MEL_DIR, exist_ok=True)

# Prepare data
data = process_dataset(AUDIO_DIR, MEL_DIR)
labels = [label for _, label in data]
train_data, test_data = train_test_split(data, test_size=0.2, stratify=labels)

# Training parameters
batch_size = 64
epochs = 20
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, optimizer, scaler
model = CNNEncoder().to(device)
model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler()

# Prepare Dataset & DataLoader
train_dataset = SpeakerDataset(train_data, max_mel_length=512)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=collate_fn)

def train_model(model, train_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_cos_sim_positive = 0
        total_cos_sim_negative = 0
        batch_count = 0

        for anchor, positive, negative in train_loader:
            batch_count += 1
            anchor, positive, negative = (
                anchor.unsqueeze(1).to(device),
                positive.unsqueeze(1).to(device),
                negative.unsqueeze(1).to(device),
            )
            optimizer.zero_grad()
            
            with autocast():
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                loss = triplet_loss(anchor_out, positive_out, negative_out)

                # Calculate cosine similarities
                cos_sim_pos = cosine_similarity(anchor_out, positive_out)
                cos_sim_neg = cosine_similarity(anchor_out, negative_out)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_cos_sim_positive += cos_sim_pos.item()
            total_cos_sim_negative += cos_sim_neg.item()

        avg_loss = total_loss / len(train_loader)
        avg_cos_sim_pos = total_cos_sim_positive / batch_count
        avg_cos_sim_neg = total_cos_sim_negative / batch_count

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "cosine_similarity_positive": avg_cos_sim_pos,
            "cosine_similarity_negative": avg_cos_sim_neg,
            "learning_rate": learning_rate
        })

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Cosine Sim (Pos): {avg_cos_sim_pos:.4f}, Cosine Sim (Neg): {avg_cos_sim_neg:.4f}")

# Training
print(f"Используем устройство: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
train_model(model, train_loader, epochs)
