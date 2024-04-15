import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from architectures import *

import matplotlib.pyplot as plt
import numpy as np
from params import Params


def train(name, loader, device, net):
    net.to(device)
    running_loss = 0.0
    for i, (images, labels) in enumerate(loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = images.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 100:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0




if __name__ == '__main__':
    params = Params()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.1307,), (0.3081,))]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params.net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(params.epochs):  # loop over the dataset multiple times
        train(params.name, trainloader, params.device, params.net)

    print('Finished Training')

    PATH = f'./{params.name}.pth'
    torch.save(params.net.state_dict(), PATH)
    print(f'Saving {PATH}')