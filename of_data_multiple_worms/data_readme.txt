fps = frames/second (all traces have been re-sampled to the same constant frame rate)
Note: heads are at 4Hz & tails are at 3.87 Hz (the max possible without upsampling)
is_AWCL_OFF = for each file, is AWCL the OFF neuron
Note: traces are either length=21 for single-sided neurons or, 42 for L/R neurons.
When length=42, neurons are ordered 21 L then 21 R, in file order.
is_L = for each neuron trace, is it L (1), R (0), or neither (nan) redundant to the above
max_traces = the max(dF/F0) per trace
neurons = the names of the neurons (L/R are combined into a single class)
neurons_i = indices for the L/R neurons in a separate file (available if you want)
samples = the number of real-valued traces available per neuron
Note: traces contain empty ({}) when the neuron was unavailable in the indexed file
stim_names = a list of the 3 stimuli we give
stim_times = times when stimuli are turned on & off
stims = the stimulus order per file
traces = the traces for each neuron (note the empty {} place holders)
Note: all traces are ordered by file but, for L/R neurons, its 21 L then 21 R
norm_traces = the normalized traces for each neuron (note the empty {} place holders)
I removed non-neuronal cells like AMSO, PHSO, & HMC.
Note: dF/F0 has negative values & I used Erdem�s simple histogram matching without the nonnegative constraint (as requested). I can kick it back in as needed.