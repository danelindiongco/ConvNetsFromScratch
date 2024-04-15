import torch
from architectures import *

class Params:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.name = "SimpleCNN"
        self.batch_size = 100
        self.epochs = 10
        self.net = Net()