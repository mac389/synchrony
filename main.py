from Network import Network as Network
import analysis as postdoc
import os
import numpy as np



ru = {}
ru['idem'] = {'means':np.array([0,0]),
			  'covariances': np.ones((2,2))}

simulation = Network(duration=1000,downsampling=1, ru_correl=ru['idem'], mixing_fraction=np.linspace(0,1,num=11))

data = postdoc.load_data(simulation.writename)
path,_ = os.path.split(simulation.writename)
postdoc.accuracy_figure(data,savename=os.path.join(path,'accuracy'))
postdoc.correlation_visualization(data,savename =os.path.join(path,'correlations'))


'''
	TODO: 
		1. Exploring sensitvity of recall to changes in M and initial conditions (mixing parameter) 

		2. What values of Qru denote addictive states?

		3. What values of Quu and Qvu are associated with different values of Qru?

		4. Abstract so can run with different r,u combinations; analyze qvu | quu
'''