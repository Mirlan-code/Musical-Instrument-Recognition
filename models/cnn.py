import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.utils as utils
from torch.autograd import Variable
import matplotlib.pyplot as plt

import itertools 
from models.evaluate_model import evaluate_model
from dataset_preparation.preprocess_data import Dataset
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

LABELS_TO_NUMBERS = {
    "gac" : 0,
    "pia" : 1,
    "sax" : 2
}


class Net(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=16, out_channels=, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)

        log_probs = torch.nn.functional.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output.permute(1, 0, 2) #original output dimensions are batchSizex1x10 
        loss = F.nll_loss(output[0], target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        output = output.permute(1, 0, 2)
        pred = output.max(2)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum().item()

    return correct / len(test_loader.dataset)


class CNNDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.X[idx], LABELS_TO_NUMBERS[self.Y[idx]])
        if self.transform:
            sample = self.transform(sample)
        return sample


def cnn(dataset, features):
    X, Y = dataset(features)
    print(X.shape)
    print(Y.shape)

    # Transformations
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2.0/10)

    epochs = 8
    momentum = 0.5
    batch_size = 32
    learning_rate = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CNNDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = CNNDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Net()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)

    for epoch in range(1, epochs + 1):
        print("Epoch number is " + str(epoch))
        train(model, device, train_loader, optimizer)

    return test(model, device, test_loader)
    


def findsubsets(s, n): 
    return [set(i) for i in itertools.combinations(s, n)]

if __name__ == "__main__":
    dataset = Dataset(path="dataset", reinitialize=False)
    s = [
        "chroma_stft",
        "chroma_cqt",
        "chroma_cens",
        "melspectogram",
        # "rms",
        # "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
        "spectral_flatness",
        # "spectral_rolloff",
        # "poly_features",
        "tonnetz",
        "zero_crossing_rate",

        "tempogram",
        # "fourier_tempogram"
    ]
    answer = []
    for i in range(len(s), len(s) + 1):
        lists = findsubsets(s, i)
        for l in lists:
            print(l)
            res = cnn(dataset, l)
            answer.append([l, res])
    answer = sorted(answer, key=lambda res: res[1], reverse=True)
    for i in answer:
        print(i[0])
        print(str(i[1]))