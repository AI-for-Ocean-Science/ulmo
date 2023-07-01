'''Linear Model in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(dim_input, dim_output)
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
class MultiLayer(nn.Module):
    def __init__(self, dim_input, dim_output, hidden_structure):
        super(MultiLayer, self).__init__()
        linear_layer_list = self._make_dense_layers(dim_input, dim_output, hidden_structure)
        self.linear_layer = nn.ModuleList(linear_layer_list)
        
    def _make_dense_layers(self, dim_input, dim_output, hidden_structure):
        linear_layer_list = []
        for num_neurons in hidden_structure:
            linear = nn.Linear(dim_input, num_neurons)
            linear_layer_list.append(linear)
            dim_input = num_neurons
        linear = nn.Linear(dim_input, dim_output)
        linear_layer_list.append(linear)
        return linear_layer_list
        
    def forward(self, x):
        output = x
        for layer in self.linear_layer[:-1]:
            output = layer(output)
            output = torch.relu(output)
        output = self.linear_layer[-1](output)
        return output   

def linear_deproj(dim_input, dim_output):
    return LinearModel(dim_input, dim_output)

def multilayer_deproj(dim_input, dim_output, hidden_structure):
    return MultiLayer(dim_input, dim_output, hidden_structure) 

def test_linear():
    net = linear_deproj(10, 10)
    x = torch.randn(10, 10)
    y = net(x)
    print(y)
    
def test_multilayer():
    net = MultiLayer(10, 10, [20, 30, 40])
    x = torch.randn(10, 10)
    y = net(x)
    print(y)

if __name__ == '__main__':
    #test_linear()
    test_multilayer()