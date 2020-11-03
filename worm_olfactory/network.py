import torch
import torch.nn as nn

class Network(nn.Module):
	def __init__(self, connectome, device = 'cpu', random_seed = None, dt = 0.2, remove_synapses = False, dales_rule = True, nonlinearity = 'relu'): 
		super(Network, self).__init__()
		if random_seed != None:
			torch.random.manual_seed(random_seed)
		self.ct = connectome
		self.N = len(self.ct.neuron_list)
		chem_weight_scale, elec_weight_scale = 1e-6, 1e-6
		self.chem_weight_scale = torch.tensor([chem_weight_scale], device = device).to(torch.float32).unsqueeze(1)
		self.elec_weight_scale = torch.tensor([elec_weight_scale], device = device).to(torch.float32).unsqueeze(1)
		self.chem_weight_scale.requires_grad = False
		self.elec_weight_scale.requires_grad = False
		chem_rand = chem_weight_scale*torch.rand(self.ct.synapses_dict['chem_weights'].shape, dtype = torch.float32, device = device) 
		elec_rand = elec_weight_scale*torch.rand(self.ct.synapses_dict['eassym_weights'].shape, dtype = torch.float32, device = device) 
		self.chem_weights = torch.nn.Parameter(self.chem_weight_scale * self.ct.synapses_dict['chem_weights'] + chem_rand)
		self.elec_weights = torch.nn.Parameter(self.elec_weight_scale * self.ct.synapses_dict['eassym_weights'] + elec_rand)
		if remove_synapses:
			self.chem_weights = torch.zeros(shape = self.ct.chem_weight_matrix.shape).to(torch.float32).to(device)
			self.elec_weights = torch.zeros(shape = self.ct.esym_weight_matrix.shape).to(torch.float32).to(device)
			self.chem_weights.requires_grad = False
			self.esym_weights.requires_grad = False
		self.chem_adj = self.ct.synapses_dict['chem_adj']
		self.elec_adj = self.ct.synapses_dict['esym_adj']
		self.chem_adj.requires_grad = False
		self.elec_adj.requires_grad = False
		self.ones = torch.ones((1, self.N), device = device)
		self.neuron_signs = torch.ones((1,self.N), dtype = torch.float32, device = device)
		if dales_rule:
			self.neuron_signs = torch.nn.Parameter(0.01*torch.rand((1,self.N), dtype = torch.float32, device = device) - 0.002)
		self.neuron_tau = torch.nn.Parameter(0.05*torch.rand((self.N,1), dtype = torch.float32, device = device)) 
		self.calcium_tau = torch.nn.Parameter(0.2*torch.rand((self.N,1), dtype = torch.float32, device = device)) 
		self.calcium_shift = torch.nn.Parameter(0.01*torch.rand((self.N,1), dtype = torch.float32, device = device) - 0.005)
		self.neuron_bias = torch.nn.Parameter(1e-1*torch.rand(self.N, device = device).unsqueeze(1))
		#self.calcium_scale = torch.nn.Parameter(torch.ones((self.N,1), dtype = torch.float32, device = device))
		self.initialization =  torch.nn.Parameter(torch.zeros((self.N,1), dtype = torch.float32, device = device))
		self.dt = torch.tensor(dt , dtype = torch.float32, device = device)
		self.calcium_rate = torch.tensor(0.2, dtype = torch.float32, device = device)	
		if nonlinearity == 'relu':
			self.neuron_nonlinearity = torch.nn.ReLU()
		elif nonlinearity == 'softplus':
			self.neuron_nonlinearity = torch.nn.Softplus(threshold = 5)
		elif nonlinearity == 'leakyrelu':
			self.neuron_nonlinearity = torch.nn.LeakyReLU(negative_slope = 1e-2)
		self.loss = self.__regularized_MSEloss

	def forward(self, s, x0 = None):
		if x0 is None:
			x0 = self.initialization
		x_int, f_int, x_slow, f_slow, neural_in, sensory_in = [x0], [x0], [x0], [x0], [], []
		for t in range(int(s.shape[0])):
			chem_in = torch.mm((self.neuron_signs.repeat(self.N,1) * self.chem_weights * self.chem_adj), x_int[t])
			gap_potentials = torch.mm(x_int[t], self.ones) - torch.mm(self.ones.transpose(0,1), x_int[t].transpose(0,1)) # = x1^T - 1x^T
			elec_in = torch.sum((self.elec_weights + self.elec_weights.transpose(0,1)) * self.elec_adj * gap_potentials, dim = 1).unsqueeze(1)

			####### ------------------------------------
			neuron_tau = self.neuron_tau.clamp(min = self.dt)
			calcium_tau = self.calcium_tau.clamp(min = self.dt)

			xt = x_int[t] + (self.dt/neuron_tau)*(self.neuron_nonlinearity(chem_in + elec_in + self.neuron_bias +  s[t,:].unsqueeze(1))-x_int[t])
			ft = f_int[t] + (self.dt/calcium_tau)*(x_int[t] - f_int[t]) 

			x_int.append(xt)
			f_int.append(ft)
			if t% int(self.calcium_rate/self.dt) == 0:
				neural_in.append(chem_in + elec_in)
				sensory_in.append(s[t,:].unsqueeze(1))
				x_slow.append(xt)	
				f_slow.append(ft + self.calcium_shift)

		return x_slow[1:], f_slow[1:], neural_in, sensory_in

	def nosign_forward(self, s, neuron_signs, x0 = None):
		if x0 is None:
			x0 = self.initialization
		x_int, f_int, x_slow, f_slow, neural_in, sensory_in = [x0], [x0], [x0], [x0], [], []
		for t in range(int(s.shape[0])):
			chem_in = torch.mm((neuron_signs * self.chem_weights * self.chem_adj), x_int[t])
			gap_potentials = torch.mm(x_int[t], self.ones) - torch.mm(self.ones.transpose(0,1), x_int[t].transpose(0,1)) # = x1^T - 1x^T
			elec_in = torch.sum((self.elec_weights + self.elec_weights.transpose(0,1)) * self.elec_adj * gap_potentials, dim = 1).unsqueeze(1)
			neuron_tau = self.neuron_tau.clamp(min = self.dt)
			calcium_tau = self.calcium_tau.clamp(min = self.calcium_rate)
			xt = x_int[t] + (self.dt/neuron_tau)*(self.neuron_nonlinearity(chem_in + elec_in + self.neuron_bias +  s[t,:].unsqueeze(1))-x_int[t])
			ft = f_int[t] + (self.dt/calcium_tau)*(x_int[t] - f_int[t])
			x_int.append(xt)
			f_int.append(ft)
			if t% int(self.calcium_rate/self.dt) == 0:
				neural_in.append(chem_in + elec_in)
				sensory_in.append(s[t,:].unsqueeze(1))
				x_slow.append(xt)	
				f_slow.append(ft + self.calcium_shift)
		return x_slow[1:], f_slow[1:], neural_in, sensory_in

	def __regularized_MSEloss(self,x,target, missing_target, male_reg = 0, shared_reg = 0, weight_reg = 0):
		msefit = torch.mean((1-missing_target)*(x - target)**2) 
		reg_male =  male_reg*(torch.sum(torch.abs(x * self.ct.neuron_mask_dict['male'].repeat(x.shape[0],1))))
		reg_shared = shared_reg*(torch.sum(torch.abs(x * self.ct.neuron_mask_dict['shared'].repeat(x.shape[0],1))))
		weight_decay = weight_reg*(torch.norm(self.chem_weights) + torch.norm(self.elec_weights))
		return msefit + reg_male + reg_shared + weight_decay

if __name__ == "__main__":
	basepath = Path.home()/'Samuel_Susoy_mating/'
	net = 'synaptic_partners_v1.csv'
	neuron = 'transmitters_and_annotations.csv'
	calcium_folder = 'normalized_activity_id_not_filtered/'
	from susoy_connectome import Connectome
	from data import Mating_Data as md
	mdata = md(basepath, net, neuron, calcium_folder)
	subnet_df = mdata.extract_subnet()
	subneuron_df = mdata.extract_subneuron()
	connectome = Connectome(subnet_df, subneuron_df, mdata.neurons)
	network = Network(connectome)


		


