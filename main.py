from Network import Network as Network
import analysis as postdoc
import os
import numpy as np



ru = {}
ru['idem'] = {'means':np.array([0,0]),
			  'covariances': np.ones((2,2))}

mixing_fractions = np.linspace(0,1,num=3)
simulation = Network(duration=1000,downsampling=1, ru_correl_matrix=ru['idem'], mixing_fraction=mixing_fractions)



#pass right version of results to analysis
#create graphics the use all iterations of results in a simulation

active_directory = simulation.basedir

results = [filename for filename in os.listdir(active_directory) if 'results' in filename]
asymp_accuracy = np.zeros(size=mixing_fractions.shape)
for i,(results_filename,fraction) in enumerate(zip(results,mixing_fractions)):
	print os.path.join(active_directory,results_filename)
	data = postdoc.load_data(os.path.join(active_directory,results_filename))
	asymp_accuracy[i] = postdoc.accuracy_figure(data,savename=os.path.join(active_directory,'accuracy-%s')%str(int(fraction*10)))
	postdoc.correlation_visualization(data,savename =os.path.join(active_directory,'correlations-%s')%str(int(fraction*10)))

postdoc.sensitivity(mixing_fractions,asymp_accuracy, savename = os.path.join(active_directory,'sensitivity'))


'''
	TODO: 
		1. Exploring sensitvity of recall to changes in M and initial conditions (mixing parameter) 

		2. What values of Qru denote addictive states?

		3. What values of Quu and Qvu are associated with different values of Qru?

		4. Abstract so can run with different r,u combinations; analyze qvu | quu
'''