import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from params import Params


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test(net, loader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on all test images: {100 * correct // total} %')
    print('\n')


def testclasses(net, loader, classes):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for {classname} is {accuracy} %')


if __name__ == '__main__':
    params = Params()

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))]))

    testloader = torch.utils.data.DataLoader(testset, batch_size=params.batch_size,
                                             shuffle=False, num_workers=0)

    PATH = f'./{params.name}.pth'
    params.net.load_state_dict(torch.load(PATH))

    test(params.net, testloader)  # Test all images

    testclasses(params.net, testloader, params.classes)  # Test accuracy for each class
