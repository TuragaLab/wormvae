import numpy as np 
import torch 

class Data_Loader():
    def __init__(self, net_input, target, out_dim):
        '''
        Parameters:
            net_input - stimulus input into the network
            target - ground truth of network prediction
            out_dim - full dimensionality of the output
            missing_input - mask for missing input data
            masked_target - ground truth with mask concatenated to it
            total_input - net_input concatenated to the missing_input
        '''
        self.device = net_input.device
        self.input = net_input
        self.target = target
        self.out_dim = out_dim
        self.total_input = self.__generate_masked_input()
        self.masked_target, self.missing_target = self.__generate_target_mask()
        self.start_index = 0
        self.shift = 0
        

    def generate_random_batch(self, interp_factor, batch_size):
        '''
        Generates a batch of size 'batch_size' starting from a random position in the input array. 

        Parameters:
            interp_factor - float indicating multiplicative factor by which interpolation has upsampled input data
            batch_size 
        Returns:
            batch_input
            batch_target
            batch_missing_target
        '''
        start_index = np.random.uniform(low = 0, high = (self.target.shape[0] - batch_size)) 
        batch_total_in = self.total_input[int(start_index)*interp_factor: (int(start_index) + batch_size)*interp_factor, :]
        batch_target = self.masked_target[int(start_index):int(start_index) + batch_size,:]
        batch_missing_target = self.missing_target[int(start_index): int(start_index) + batch_size,:]
        return batch_total_in, batch_target, batch_missing_target

    def generate_next_batch(self, interp_factor, batch_size):
        '''
        Generates a batch starting from self.start_index + self.shift which are initialized to 0,0 with the class. 
        Each successive call will generate a batch from the next section of the array in order. If the entire array
        is iterated through, generate_next_batch will generate a batch from the beginning again with a shift updated
        with a number drawn from a random uniform distribution in [0, batch_size]

        Parameters
            interp_factor - float indicating multiplicative factor by which interpolation has upsampled input data
            batch_size
        Returns:
            batch_input
            batch_target
            batch_missing_target
        '''
        start_index = self.start_index + self.shift
        if start_index < self.target.shape[0] - batch_size: 
            batch_total_in = self.total_input[int(start_index)*interp_factor: (int(start_index) + batch_size)*interp_factor, :]
            batch_target = self.masked_target[int(start_index):int(start_index) + batch_size,:]
            batch_missing_target = self.missing_target[int(start_index): int(start_index) + batch_size,:]
            self.start_index += batch_size
        else:
            self.start_index = 0
            self.shift = int(np.random.uniform(low = 0, high = batch_size))
            batch_total_in, batch_target, batch_missing_target = self.generate_next_batch(interp_factor, batch_size)
        return batch_total_in, batch_target, batch_missing_target

    def __generate_masked_input(self):
        '''
        Returns input with missing tensor concatenated to it.
        '''
        input_stim = self.input
        missing_input = torch.zeros(self.input.shape, device = self.device).to(torch.float32)
        missing_input[np.isnan(input_stim.cpu().numpy())] = 1
        input_stim[np.isnan(input_stim.cpu().numpy())] = 0
        masked_input = torch.cat((input_stim, missing_input), dim = 1)
        return masked_input

    def __generate_target_mask(self):
        '''
        Returns the target with extra dimensions concatenated as zero columns and the
        missing data vector: (target, missing_target)
        '''
        full_missing_target = torch.zeros(size = (self.target.shape), device = self.device)
        full_missing_target[torch.isnan(self.target)] = 1
        full_target = self.target
        full_target[torch.isnan(self.target)] = 0
        return full_target, full_missing_target

    def reset(self):
        self.start_index = 0
        self.shift = 0
    
if __name__ == "__main__":
    worm_id = 0
    dt = 0.04
    basepath = Path.home()/'Samuel_Susoy_mating/'
    net = 'synaptic_partners_v1.csv'
    neuron = 'transmitters_and_annotations.csv'
    calcium_folder = 'normalized_activity_id_not_filtered/'
    mdata = Mating_Data(basepath, net, neuron, calcium_folder)
    subnet_df = mdata.extract_subnet()
    subneuron_df = mdata.extract_subneuron()
    connectome = Connectome(subnet_df, subneuron_df, mdata.neurons)
