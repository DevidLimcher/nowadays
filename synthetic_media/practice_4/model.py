import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Используем AdaptiveAvgPool2d для унификации выходных данных
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        
        # Убираем фиксированное количество входов и делаем динамическое определение
        self.fc = nn.Linear(64 * 16 * 16, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Унифицируем размер с помощью адаптивного пуллинга
        x = self.adaptive_pool(x)
        
        # Временный вывод формы тензора
        #print("Shape before FC:", x.shape)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
