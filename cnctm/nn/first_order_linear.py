import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np


class FirstOrderLinear(Module):
    def __init__(self, n, dt, bias_init=None, tau_init=None):
        super(FirstOrderLinear, self).__init__()
        self.n = n
        self.dt = dt

        if tau_init is None:
            self.tau = Parameter(torch.empty(self.n, device = device).uniform_(self.dt, 0.5))
        else:
            self.tau = Parameter(tau_init)
        if bias_init is None:
            self.bias = Parameter(torch.zeros(self.n, device = device))
        else:
            self.bias = Parameter(bias_init)

    def forward(self, input, custom_init=None):
        with torch.no_grad():
            self.tau.data.clamp_(min = 0)

        if custom_init is None:
            custom_init = input[0, :]
        hidden_states, output = [custom_init], []
        tau_clamp = self.tau.clamp(min=self.dt)
        timesteps = input.shape[0]

        for t in range(timesteps):
            s = hidden_states[-1]
            s_tp1 = s + (self.dt / tau_clamp) * (input[t, :] - s)
            hidden_states.append(s_tp1)
            output.append(s_tp1 + self.bias)
        return torch.cat(output, dim=0)
