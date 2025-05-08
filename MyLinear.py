import torch
from torch import nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_size, out_size))
        self.bias = nn.Parameter(torch.randn(out_size,))
        
    def forward(self, x):
        linear = torch.matmul(x, self.weight.data) + self.bias.data
        return F.relu(linear)

if __name__ == '__main__':
    linear = MyLinear(5, 3)
    net = nn.Sequential(MyLinear(64,8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))