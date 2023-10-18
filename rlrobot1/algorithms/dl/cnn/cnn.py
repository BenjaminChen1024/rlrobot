# Convolution Neural Network Modules
import torch.nn as nn

class CNNLayer(nn.Module):
    def __init__(self, input_shape, hidden_size, layer_N, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()
        self.layer_N = layer_N - 1
        active_func = nn.ReLU()

        input_channel = input_shape[0]
        input_width = input_shape[1]
        input_height = input_shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride),
            active_func,
            nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size),
            active_func,
            nn.Linear(hidden_size, hidden_size), active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x
