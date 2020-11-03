from __future__ import division

import math

import torch
import torch.nn.init as init

def positive_kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = init._calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(0., bound)

def _no_grad_signed_uniform_(tensor):
    with torch.no_grad():
        tensor.random_(0, 2)
        tensor.sub_(0.5)
        return tensor.sign_()

def positive_no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        tensor.normal_(mean, std)
        return tensor.mul_(torch.sign(tensor))

def positive_kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = init._calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return positive_no_grad_normal_(tensor, 0., std)

def positive_xavier_normal_(tensor, gain=1.):
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return positive_no_grad_normal_(tensor, 0., std)

# for student nets
def positive_xavier_normal_sparsity(tensor, sparsity, gain=1.):
    with torch.no_grad():
        input_fmaps = sparsity.sum(axis=1)
        fan_out = tensor.size(0)
        fan_in_orig = tensor.size(1)
        receptive_field_size = tensor[0][0].numel()

        feature_map_stds = []
        for inp_fmap in input_fmaps:
            fan_in = inp_fmap * receptive_field_size
            fmap_std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            feature_map_stds.append(fmap_std)

        final_tensor = []
        for std in feature_map_stds:  # number of fmaps
            fmap_filters = positive_no_grad_normal_(torch.empty_like(tensor[0:1, ...]), 0., std)
            final_tensor.append(fmap_filters)
        final_tensor = torch.cat(final_tensor, axis=0)
        assert tensor.shape == final_tensor.shape
        return final_tensor

def bias_sparsity_init(bias_tensor, sparsity):
    with torch.no_grad():
        input_fmaps = sparsity.sum(axis=1)
        bounds = []
        for fan_in in input_fmaps:
            if fan_in == 0:
                fan_in = max(input_fmaps)  # doesn't matter
            bound = 1. / math.sqrt(fan_in)
            bounds.append(bound)

        final_tensor = []
        for bound in bounds:
            bias_init = init.uniform_(torch.empty_like(bias_tensor[0:1, ...]), -bound, bound)
            final_tensor.append(bias_init)

        final_tensor = torch.cat(final_tensor, axis=0)
        assert final_tensor.shape == bias_tensor.shape

        return final_tensor
