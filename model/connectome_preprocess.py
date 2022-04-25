import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import torch


__all__ = ['Connectome','Neuron']

class Neuron:
    def __init__(self, name, index, sex_spec, shared, type):
        self.name = name
        self.index = index
        self.sex_spec = sex_spec
        self.shared = shared
        self.type = type

class WhiteConnectomeData:
    def __init__(self, cnctmpath, device = 'cuda'):
        basepath = cnctmpath
        self.__device = device
        self.__chemdf = pd.read_csv(basepath+'chem.csv', skipinitialspace = True, quotechar = '"').drop('Unnamed: 0', axis = 1)
        self.__gjsym = pd.read_csv(basepath+'elec_symmetric.csv', skipinitialspace = True, quotechar = '"').drop('Unnamed: 0', axis = 1)
        self.__gjassym = pd.read_csv(basepath+'elec_asymmetric.csv', skipinitialspace = True, quotechar = '"').drop('Unnamed: 0', axis = 1)
        self.__spec_neurondf = pd.read_csv(basepath + 'herm_specific_cells.csv', skipinitialspace = True, quotechar = '"')
        self.__spec_neurondf.index = list(self.__spec_neurondf['name'])
        self.__spec_neurondf.drop('name', axis = 1)
        self.__chemdf.index = list(self.__chemdf.columns)
        self.__gjsym.index = list(self.__gjsym.columns)
        self.__gjassym.index = list(self.__gjassym.columns)   
        self.__shared_neurondf = pd.read_csv(basepath + 'sex_shared_cells.csv', skipinitialspace = True, quotechar = '"')           
        self.__shared_neurondf.index = list(self.__shared_neurondf['name'])       
        self.__shared_neurondf.drop('name', axis = 1)
        self.white_connectomedf = pd.read_csv(basepath + 'white_neuron_connect.csv', skipinitialspace = True)
        self.white_neurontypesdf = pd.read_csv(basepath + 'white_neuron_types.csv', skipinitialspace = True)
        self.neuron_list = self.__generate_neuron_list()
        self.name_neuron_dict = {neuron.name:neuron for neuron in self.neuron_list}
        self.__overwrite_neuron_types()
        self.neuron_mask_dict = self.__generate_neuron_masks()  
        self.synapses_dict = self.__generate_weight_matrices()
        self.N = len(self.neuron_list)

    def __generate_neuron_list(self):
        neuron_list = []
        for i, neuron_name in enumerate(list(self.__chemdf.columns)):
            if neuron_name in list(self.__shared_neurondf['name']):
                neuron_list.append(Neuron(name = neuron_name,
                                          index = i,
                                          sex_spec = False,
                                          shared = True,
                                          type = None))
            if neuron_name in list(self.__spec_neurondf['name']):
                neuron_list.append(Neuron(name = neuron_name,
                                          index = i,
                                          sex_spec = True,
                                          shared = False,
                                          type = None))
        return neuron_list

    def __overwrite_neuron_types(self):
        for index,row in self.white_neurontypesdf.iterrows():
            neuron = self.name_neuron_dict[row['Neuron']]
            if row['Type'] == 'sensory':
                neuron.type = 'sensory'
            elif row['Type'] == 'motor':
                neuron.type = 'motor'
            elif row['Type'] == 'interneuron':
                neuron.type = 'inter'
            else:
                neuron.type = 'pharyngeal'              

    def __generate_neuron_masks(self):
        sex_spec_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        shared_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        motor_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        sensory_mask = torch.zeros(size = (1, len(self.neuron_list)), device = self.__device)
        for i, neuron in enumerate(self.neuron_list):
            if neuron.sex_spec:
                sex_spec_mask[:,i] = 1
            if neuron.shared:
                shared_mask[:,i] = 1
            if neuron.type == 'motor':
                motor_mask[:,i] = 1
            if neuron.type == 'sensory':
                sensory_mask[:,i]= 1
        return {'sex_spec': sex_spec_mask, 'shared': shared_mask, 'motor': motor_mask, 'sensory': sensory_mask}

    def __generate_weight_matrices(self):
        chem_weights = torch.zeros(torch.from_numpy(self.__chemdf.values).shape).float().to(self.__device)
        eassym_weights = torch.zeros(torch.from_numpy(self.__gjsym.values).shape).float().to(self.__device)
        chem_sparsity = torch.zeros(torch.from_numpy(self.__chemdf.values).shape).float().to(self.__device)
        esym_sparsity = torch.zeros(torch.from_numpy(self.__gjsym.values).shape).float().to(self.__device)
        for index, row in self.white_connectomedf.iterrows():
            if row['Neuron 1'] in self.name_neuron_dict and row['Neuron 2'] in self.name_neuron_dict:
                index1 = self.name_neuron_dict[row['Neuron 1']].index
                index2 = self.name_neuron_dict[row['Neuron 2']].index
                if row['Type'] == 'EJ':
                    eassym_weights[index1, index2] = int(row['Nbr'])
                    esym_sparsity[index1, index2] = 1
                elif row['Type'] == 'S' or row['Type'] == 'SP':
                    chem_weights[index2,index1] = int(row['Nbr'])
                    chem_sparsity[index2,index1] = 1
                else: 
                    chem_weights[index1,index2] = int(row['Nbr'])
                    chem_sparsity[index1,index2] = 1                
        
        return {'chem_weights': chem_weights, 'eassym_weights': eassym_weights,'chem_adj': chem_sparsity, 'esym_adj': esym_sparsity}
   
    def save_connectome_as_pickle(self, filename):
        '''
        Saves connectome information in dict[dict[torch.Tensor]] with format
        {'connectivity': {'chem_weights': chem_weights,
                         'eassym_weights': eassym_weights,
                         'chem_adj': chem_sparsity,
                         'esym_adj': esym_sparsity}
         'neuron_types': {'sex_spec':sex_spec_mask,
                          'shared' : shared_mask,
                           'motor' : motor_mask,
                           'sensory': sensory_mask}}
        '''

        with open(filename,'wb') as handle:
            pickle.dump({'connectivity': self.synapses_dict, 
                          'neuron_types': self.neuron_mask_dict},
                          handle,
                          protocol = pickle.HIGHEST_PROTOCOL)
                          
    def make_full_connectome_activity(self, activity_data, activity_neuron_list, holdout_list = []):
        '''
        Takes partial neural activity data (B, N_s, T) and maps it into a tensor (B, N, T) where N includes all neurons in this connectome.
        All missing values are NaN. For this connectome N = 302. 

        Parameters:
            activity_data: tensor of calcium traces from a single worm (B, N_s, T)
            activity_neuron_list: list of neuron labels associated with calcium traces

        Return:
            all_activity: tensor (B, N, T)
        '''   
        all_activity = np.NaN*torch.ones(size = (activity_data.shape[0], len(self.neuron_list), activity_data.shape[2]), device = self.__device)
        for i in range(activity_data.shape[1]):
            if activity_neuron_list[i] not in holdout_list:
                mask_neuron_index = self.name_neuron_dict[activity_neuron_list[i]].index # map neural activity to full connectome tensor index
                all_activity[:,mask_neuron_index,:] = activity_data[:,i,:] # fill in neural activity to full neuron (B, N, T) activity vector
        return all_activity

if __name__ == "__main__":
    c = WhiteConnectomeData('../herm_cnctm_data/')
    c.save_connectome_as_pickle('../herm_cnctm_data/white_connectome.pickle')