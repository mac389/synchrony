from Network import Network as Network
from progress.bar import Bar

import analysis as postdoc
import os, visualization
import numpy as np


N = {'memories':10,'neurons':100}
duration = 5000
ru = {}

r_schema = ['susceptible']#,'resilient']
u_schema = ['exposure']#,'chronic','cessation']


basedir=None
mixing_fractions = np.linspace(0,1,num=21)
bar = Bar('Running simulation', max=len(r_schema)*len(u_schema)*len(mixing_fractions))
bar.next()
for reward in r_schema:
    print ''
    print reward
    for stimulus in u_schema:
        moniker = '%s-%s'%(reward,stimulus)
        print '\t',stimulus
        simulation = Network(N=N,duration=duration,downsampling=1, mixing_fraction=mixing_fractions,
                            r_schema=reward,u_schema=stimulus,basename=basedir)

        active_directory = simulation.basedir
        results = [filename for filename in os.listdir(active_directory) if 'all-results' in filename]
        accuracy = np.zeros((len(mixing_fractions),N['memories']))
        energies = np.zeros_like(accuracy)

        for i,fraction in enumerate(mixing_fractions):
            print '\t\t',fraction,
            data = simulation.results
            accuracy[i,:] = postdoc.accuracy_figure(data,
                    savename=os.path.join(active_directory,'accuracy-%s-%s')%(str(int(fraction*100)),moniker))
            visualization.track_matrices(data['M'],
                    savename=os.path.join(active_directory,'M-change-%s-%s')%(str(int(fraction*100)),moniker))
            visualization.memory_stability(data['memory_stability'],
                    savename=os.path.join(active_directory,'M-stability-%s-%s')%(str(int(fraction*100)),moniker))
            visualization.network_stability(data['network_stability'],
                    savename=os.path.join(active_directory,'network-stability-%s-%s')%(str(int(fraction*100)),moniker))
            bar.next()

        correl = postdoc.sensitivities(mixing_fractions,accuracy.transpose(), savename = os.path.join(active_directory,'sensitivities-%s'%moniker))
        #Transpose so that the x-axis contains mixing fraction and y-axis accuracy
        basedir = simulation.basedir
        del simulation
        bar.next()
    bar.next()
bar.finish()
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
