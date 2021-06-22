import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torchaudio

import os


class MusicDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.labels = [ x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) ]
        print("LABELS", self.labels)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        l = idx // 100
        x = idx % 100

        label = self.labels[l]
        waveform, _ = torchaudio.load(os.path.join(self.path, label, '{}.{:05d}.wav'.format(label, x)), 30)

        waveform = torch.split(waveform, 660000, 1)[0]

        return waveform, l


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential( nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)) 
        self.layer2 = nn.Sequential( nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2), 
            nn.ReLU(), nn.MaxPool1d(kernel_size=2, stride=2)) 
        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(7 * 7 * 64, 1000) 
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x): 
        out = self.layer1(x) 
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1) 
        out = self.drop_out(out)

        out = self.fc1(out) 
        out = self.fc2(out)

        return out


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = MusicDataset('Data/genres_original')
dataloader = DataLoader(data, batch_size=64, shuffle=True)

total_step = 0
num_epochs = 10

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Отслеживание точности
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))