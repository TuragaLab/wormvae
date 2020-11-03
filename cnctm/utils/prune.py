import torch
from torch.nn.parameter import Parameter
import numpy as np

# based on https://github.com/google-research/lottery-ticket-hypothesis/blob/master/foundations/pruning.py

# for ConstrainedLinear
def refactorize_weights(connectome_layer):
    with torch.no_grad():
        connectome_layer.magnitudes.mul_(torch.abs(connectome_layer.signs))
        connectome_layer.signs.sign_()

def prune_by_percent(connectome_layer, percent):
    refactorize_weights(connectome_layer)
    if percent > 0.:
        with torch.no_grad():
            mask = connectome_layer.sparsity.cpu().numpy()
            final_weight = connectome_layer.magnitudes.cpu().numpy()
            assert final_weight.min() >= 0.
            sorted_weights = np.sort(final_weight[mask == 1])
            # Determine the cutoff for weights to be pruned.
            cutoff_index = np.round(percent * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index]
            # TODO: add exception for edge condtion; cutoff == 0
            if cutoff == 0.:
                print('cutoff index == 0.')
                rows, cols = np.where(mask == 1.)
                num_ones = rows.shape[0]
                assert num_ones == cols.shape[0]
                rand_one_indices = np.random.randint(low=0, high=num_ones, size=cutoff_index)
                rand_ones = zip(rows[rand_one_indices], cols[rand_one_indices])
                for t in rand_ones:
                    mask[t] = 0.
                connectome_layer.sparsity = Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
            else:
                connectome_layer.sparsity = Parameter(torch.tensor(np.where(np.abs(final_weight) <= cutoff, np.zeros_like(mask), mask), dtype=torch.float32), requires_grad=False)


# for ConstrainedConv2d
def refactorize_filter_weights(connectome_layer):
    with torch.no_grad():
        connectome_layer.magnitudes.mul_(torch.abs(connectome_layer.signs[None, :, None, None]))
        connectome_layer.signs.sign_()

def prune_filter_by_percent(connectome_layer, percent):
    refactorize_filter_weights(connectome_layer)
    if percent > 0.:
        with torch.no_grad():
            mask = connectome_layer.sparsity.cpu().numpy()
            final_weight = connectome_layer.magnitudes.detach().cpu().numpy()
            assert final_weight.min() >= 0.
            filter_norms = np.linalg.norm(final_weight, axis=(2, 3))
            assert filter_norms.shape == mask.shape
            sorted_weights = np.sort(filter_norms[mask == 1])
            # Determine the cutoff for filters to be pruned.
            cutoff_index = np.round(percent * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index]
            # edge condition; if cutoff == 0, don't prune all zeros
            if cutoff == 0.:
                print('cutoff index == 0.')
                rows, cols = np.where(mask == 1.)
                num_ones = rows.shape[0]
                assert num_ones == cols.shape[0]
                rand_one_indices = np.random.randint(low=0, high=num_ones, size=cutoff_index)
                rand_ones = zip(rows[rand_one_indices], cols[rand_one_indices])
                for t in rand_ones:
                    mask[t] = 0.
                connectome_layer.sparsity = Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
            else:
                connectome_layer.sparsity = Parameter(torch.tensor(np.where(np.abs(filter_norms) <= cutoff, np.zeros_like(mask), mask), dtype=torch.float32), requires_grad=False)
