import torch
import numpy as np
import sys
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F  # All functions that don't have any parameters
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import Dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,), )])
# loadingSets
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=True, transform=transform)
# Preparing for validation test
indices = list(range(len(trainset)))
np.random.shuffle(indices)
# get 20% of the train set
split = int(np.floor(0.2 * len(trainset)))
train_sample = SubsetRandomSampler(indices[split:])
valid_sample = SubsetRandomSampler(indices[:split])
# Date Loader
trainLoader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64)
validLoader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64)


class BestNet(nn.Module):
    # setting up the network
    def __init__(self, image_size):
        super(BestNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn0 = nn.BatchNorm1d(100)
        self.bn1 = nn.BatchNorm1d(50)
        self.bn2 = nn.BatchNorm1d(10)

    # forward propagation
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return F.log_softmax(x, dim=1)


# setting up the model
model = BestNet(image_size=28 * 28)
criterion = nn.NLLLoss()  # negative log likelihood
optimizer = optim.SGD(model.parameters(), lr=0.095)
validLossMin = np.inf
epochs = 10
model.train()
train_losses, valid_losses, train_acc, valid_acc = [], [], [], []
# train the model for 10 epochs
for epoch in range(epochs):
    runningLoss = 0
    validationLoss = 0
    hits = 0
    val_hits = 0
    for images, labels in trainLoader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        hits += pred.eq(labels.view_as(pred)).cpu().sum()
        runningLoss += loss.item() * images.size(0)
    # calculating validation set loss ...
    for images, labels in validLoader:
        validationOutput = model(images)
        loss = criterion(validationOutput, labels)
        pred1 = validationOutput.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        val_hits += pred1.eq(labels.view_as(pred1)).cpu().sum()
        validationLoss += loss.item() * images.size(0)

    runningLoss /= len(trainLoader.sampler)
    validationLoss /= len(validLoader.sampler)
    hits = float(hits / len(trainLoader.sampler))
    val_hits = float(val_hits / len(validLoader.sampler))

    train_losses.append(runningLoss)
    valid_losses.append(validationLoss)
    train_acc.append(hits)
    valid_acc.append(val_hits)


def test(model, test_loader):
    model.eval()
    outputFile = open("test_y", "w")
    length = len(test_loader.sampler)
    counter = 0
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for prediction in pred:
                if counter != length - 1:
                    outputFile.write(str(prediction.numpy()[0]) + "\n")
                    counter += 1
                else:
                    outputFile.write(str(prediction.numpy()[0]))
    outputFile.close()


# creating a dataset for the given "test_x" file
class TestSet(Dataset):
    # initialize the dataset
    def __init__(self, transform=None):
        self.test_set = np.loadtxt(sys.argv[3]) / 255
        self.test_set = self.test_set.astype(np.float32)
        self.transform = transform

    # return an element from the test set based on an index
    def __getitem__(self, index):
        sample = self.test_set[index]
        sample = sample.reshape(1, 784)
        if self.transform:
            sample = self.transform(sample)
        return sample

    # len of the test set
    def __len__(self):
        return len(self.test_set)


# testing our model
dataset = TestSet(transform=transform)
testLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
test(model, testLoader)
