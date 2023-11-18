import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.datasets import get_loader, get_data
from utils.run import train, test


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 输入假设为3通道图像
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = x.view(-1, 512 * 2 * 2)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        x = self.fc3(x)
        return x


data_path_A = r"../Datasets/PneumoniaMNIST/pneumoniamnist.npz"

data_a = np.load(data_path_A)
x_train, x_val, x_test, y_train, y_val, y_test = get_data(data_a)

num_epochs = 100
batch_size = 64
learning_rate = 0.001
patience = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader = get_loader(x_train, y_train, batch_size=batch_size, flag='Train')
val_loader = get_loader(x_val, y_val, batch_size=batch_size, flag='Val')
test_loader = get_loader(x_test, y_test, batch_size=batch_size, flag='Test')

model = Net().to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_model, train_loss_history, valid_loss_history = train(model, train_loader, val_loader, loss_function, optimizer, device, num_epochs, patience)
test_accuracy = test(best_model, test_loader, device)
print("Test accuracy:", test_accuracy)
