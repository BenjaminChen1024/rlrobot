# LeNet5 Convolution Neural Network Modules
import torch.nn as nn

class LeNet5Layer(nn.Module):
    def __init__(self, input_channel, padding):
        super(LeNet5Layer, self).__init__()
        self.flatten = nn.Flatten()
        active_func = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel,6,5, padding=padding),
            active_func,
            nn.MaxPool2d(2,2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            active_func,
            nn.MaxPool2d(2,2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            active_func,
            nn.Linear(120, 84),
            active_func,
            nn.Linear(84, 10),
        )

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = LeNet5Layer(1, 2)
    print(model)