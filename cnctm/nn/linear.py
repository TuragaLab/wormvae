import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module

class NeuronWiseAffine(Module):
    def __init__(self, in_features, bias=True, nonlinearity=F.softplus):
        super(NeuronWiseAffine, self).__init__()

        self.in_features = in_features
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.scale = Parameter(torch.rand(1))

        if bias:
            self.shift = Parameter(torch.rand(in_features))

    def forward(self, input):
        output = self.scale * input
        if self.bias:
            output += self.shift[None,:,None]
        return self.nonlinearity(input)

    def extra_repr(self):
        return 'in_features={}, bias={}, nonlinearity={}'.format(
            self.in_features, self.bias, self.nonlinearity
        )

class NeuronWiseLinear(Module):
    def __init__(self, in_features, bias=True):
        super(NeuronWiseLinear, self).__init__()

        self.in_features = in_features
        self.bias = bias
        self.scale = Parameter(torch.empty(in_features).uniform_(0, 0.2))

        if bias:
            self.shift = Parameter(torch.rand(in_features))

    def forward(self, input):
        output = self.scale[None,:,None] * input
        if self.bias:
            output += self.shift[None,:,None]
        return output

    def extra_repr(self):
        return 'in_features={}, bias={}'.format(self.in_features, self.bias)
