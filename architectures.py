import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(28*28, 100)
        self.linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.final(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # First Layer
        first_conv_out = self.conv1(x)
        first_relu_out = F.relu(first_conv_out)
        first_pool_out = self.pool(first_relu_out)

        second_conv_out = self.conv2(first_pool_out)
        second_relu_out = F.relu(second_conv_out)
        second_pool_out = self.pool(second_relu_out)

        x = torch.flatten(second_pool_out, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # First fully connected layer
        x = F.relu(self.fc2(x)) # Second fully connected layer
        x = self.fc3(x) # Third fully connected layer
        
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        x = F.log_softmax(x)
        
        return x


if __name__ == '__main__':
    x = 1