import torch
from model import CNNEncoder
from preprocess import process_and_split_data
from triplet_loss import triplet_loss


def train_model(encoder, train_data, num_epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    for epoch in range(num_epochs):
        total_loss = 0
        for anchor, positive, negative in train_data:
            optimizer.zero_grad()
            anchor_out = encoder(anchor)
            positive_out = encoder(positive)
            negative_out = encoder(negative)

            loss = triplet_loss(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')


if __name__ == "__main__":
    wav_path = "/home/davidlimcher/projects/nowadays/synthetic_media/practice_4/data/VCTK-Corpus/wav48"
    # Генерация триплетов и разделение данных
    train_triplets, test_triplets = process_and_split_data(wav_path)

    # Печать нескольких примеров
    for i in range(3):
        print(f"Триплет {i}: {train_triplets[i]}")

    # Проверьте количество триплетов
    print(f"Количество триплетов для обучения: {len(train_triplets)}")
    print(f"Пример триплета: {train_triplets[0]}")

    
    cnn_encoder = CNNEncoder()
    train_model(cnn_encoder, train_triplets)
