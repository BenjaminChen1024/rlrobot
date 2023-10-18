# Muti-Layer Perceptron Modules
import torch.nn as nn

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, layer_N):
        super(MLPLayer, self).__init__()
        self.layer_N = layer_N - 1
        self.flatten = nn.Flatten()
        active_func = nn.ReLU()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, hidden_size), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size), active_func, nn.LayerNorm(hidden_size)) for i in range(self.layer_N)])
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        for i in range(self.layer_N):
            x = self.fc2[i](x)
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    model=MLPLayer(2, 1, 128, 3); 
    print(model)