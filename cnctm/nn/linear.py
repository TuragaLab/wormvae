import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
from .init import positive_kaiming_uniform_, _no_grad_signed_uniform_
import pdb


def decomposed_linear(input, sparsity, magnitudes, signs, bias=None):
    assert  magnitudes.min() >= 0.
    mag_signs = torch.mul(magnitudes, signs)  # sign_vector is broadcasted and multiplied with every row of `magnitudes`
    weight = torch.mul(sparsity, mag_signs)

    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, input, weight.t())
    else:
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dales_law=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if dales_law:
            self.signs = Parameter(torch.Tensor(in_features))
        else:
            self.signs = Parameter(torch.Tensor(out_features, in_features))

        self.magnitudes = Parameter(torch.Tensor(out_features, in_features))
        self.sparsity = Parameter(torch.ones_like(self.magnitudes),
                                   requires_grad=False)  # not trainable
        # initialize weights
        _no_grad_signed_uniform_(self.signs)
        positive_kaiming_uniform_(self.magnitudes, nonlinearity='relu')

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))  # trainable
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def __repr__(self):
        return 'Constrained' + super(Linear, self).__repr__()

    def reset_parameters(self):
        if self.bias is not None:  # no changes made to bias initializations
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.magnitudes)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        return decomposed_linear(input, self.sparsity, self.magnitudes, self.signs, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class NeuronWiseAffine(Module):
    def __init__(self, in_features, bias=True, nonlinearity=F.softplus):
        super(NeuronWiseAffine, self).__init__()

        self.in_features = in_features
        self.bias = bias
        self.nonlinearity = nonlinearity
        #self.scale = Parameter(0.1 * torch.rand(in_features))
        #self.scale = Parameter(torch.empty(in_features).uniform_(0, 0.1))
        # same across neurons
        self.scale = Parameter(torch.rand(1))

        if bias:
            self.shift = Parameter(torch.rand(in_features))
            #self.shift = Parameter(torch.rand(1))

    def forward(self, input):
        #same across neurons
        #pdb.set_trace()
        output = self.scale * input
        #output = self.scale[None,:,None] * input
        if self.bias:
            output += self.shift[None,:,None]
            #output += self.shift
        #return self.nonlinearity(output)
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
        #self.scale = Parameter(0.2 * torch.rand(in_features))
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
