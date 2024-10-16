import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.fc = None  # Инициализация будет выполнена динамически

    def forward(self, x):
        # Пропуск через сверточные слои
        x = self.pool(F.relu(self.conv1(x)))
        print(f"Размер после первого сверточного слоя: {x.shape}")  # Отладка
        
        x = self.pool(F.relu(self.conv2(x)))
        print(f"Размер после второго сверточного слоя: {x.shape}")  # Отладка

        # Преобразование тензора в плоский вектор
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        print(f"Размер после выравнивания: {x.shape}")  # Отладка

        # Динамическое создание полносвязного слоя на основе текущего размера
        if self.fc is None or self.fc.in_features != x.size(1):
            print(f"Переинициализация полносвязного слоя для размера: {x.size(1)}")
            self.fc = nn.Linear(x.size(1), 128)  # Инициализация линейного слоя

        # Пропуск через полносвязный слой
        x = self.fc(x)
        return x







class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out[:, -1, :]
