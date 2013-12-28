from Network import Network as Network
import analysis as postdoc
import os, visualization
import numpy as np


N = {'memories':10,'neurons':100}
ru = {}

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

mixing_fractions = np.linspace(0,1,num=3)
simulation = Network(N=N,duration=1000,downsampling=1, mixing_fraction=mixing_fractions,r_schema=r_schema,u_schema=u_schema)

active_directory = simulation.basedir
results = [filename for filename in os.listdir(active_directory) if 'results' in filename]
accuracy = np.zeros((len(mixing_fractions),N['memories']))
energies = np.zeros((len(mixing_fractions),N['memories']))

for i,(results_filename,fraction) in enumerate(zip(results,mixing_fractions)):
	data = postdoc.load_data(os.path.join(active_directory,results_filename))
	accuracy[i,:] = postdoc.accuracy_figure(data,savename=os.path.join(active_directory,'accuracy-%s')%str(int(fraction*10)))
	# energies[i,:] = postdoc.energy_figure(data,savename=os.path.join(active_directory,'energy-%s')%str(int(fraction*10)))
	#postdoc.correlation_visualization(data,savename =os.path.join(active_directory,'correlations-%s')%str(int(fraction*10)))
	visualization.track_matrices(data['M'],savename=os.path.join(active_directory,'M-change-%s')%str(int(fraction*10)))
	visualization.memory_stability(data['memory_stability'],savename=os.path.join(active_directory,'M-stability-%s')%str(int(fraction*10)))
	visualization.network_stability(data['network_stability'],savename=os.path.join(active_directory,'network-stability-%s')%str(int(fraction*10)))
	#Don't forget about this.
correl = postdoc.sensitivities(mixing_fractions,accuracy.transpose(), savename = os.path.join(active_directory,'sensitivities'))
#Transpose so that the x-axis contains mixing fraction and y-axis accuracy
#print correl

'''
	TODO: 
		1. Exploring sensitvity of recall to changes in M and initial conditions (mixing parameter) 

		2. What values of Qru denote addictive states?

		3. What values of Quu and Qvu are associated with different values of Qru?

		4. Abstract so can run with different r,u combinations; analyze qvu | quu
'''

'''
	Figures:

	 1. Heat map of the maximum accuracy (q) the network can obtain for each memory (row) as a function of interference (column).
	 2. Heat map of the maximum accruacy (q) the network can obtain for each memory (row) as a function of stimulus-reward correlation (column).
	 3. Heat map of the stability (energy) of each memory (row) as a function of stimulus-reward correlation (column).
	 		--Three schemes of stimulus-reward correlation: 
				 		"Exposure" (a pulse)
				 		"Continued use" (series of pulses), 
				 		"Cessation" (truncated series of pulses)

	 4. Time series of correlations 
	 5. Plot of asymptotic network stability (E(v_inf)) for different values of Qru. 
	 6. Time series of M,W tracking changes in the matrix   
'''