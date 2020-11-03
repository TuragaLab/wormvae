import torch
from torch.utils.data import Dataset
import pdb

# connectome_dict = {'sparsity_c': None,
#                     'sparsity_e': None,
#                     'signs_c': None,
#                     'magnitudes_c': None,
#                     'magnitudes_e': None,
#                     'n': None}

class ConnectomeConstructor:
    def __init__(self, connectome_dict):
        for key, value in connectome_dict.items():
            if value is not None:
                setattr(self, key, value)

class TimeSeriesDataloader(Dataset):
    def __init__(self, data, window_size): # trace.shape = (tpoints)
        super(TimeSeriesDataloader, self).__init__()
        self.data = data 
        self.window_size = window_size
        self.max_index = self.data[0].shape[0] // self.window_size

    def __getitem__(self, index):
        res = [self.data[i][index*self.window_size[i]:(index+1)*self.window_size[i], :] for i in range(len(self.data))]
        return res

    def __len__(self):
        return self.max_index
