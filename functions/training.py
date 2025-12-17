from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import time



class TwoLayerNet(nn.Module):
    def __init__(self, hidden_size=128):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ThreeLayerNet(nn.Module):
    def __init__(self, hidden_size1=128, hidden_size2=64):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(hidden_size, train_loader, test_loader, epochs=3, lr=0.001,model_name="model"):
    if len(hidden_size) == 1:
        model = TwoLayerNet(*hidden_size)
    elif len(hidden_size) == 2:
        model = ThreeLayerNet(*hidden_size)
    else:
        print("Invalid len of hidden sizes")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    train_time = time.time() - start_time

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    torch.save(model.state_dict(), f"data/models/{model_name}_params.pth")

    return train_time, acc, cm