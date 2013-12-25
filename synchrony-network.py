from brian import *
from brian.library.synapses import alpha_synapse
from brian.library.IF import leaky_IF

from numpy import random

import matplotlib.pyplot as plt

N = 20


V = {'reset':-60*mV,'threshold':-40*mV}
tau = {'membrane':10*ms,'synapses':10*ms}


W = .05*(1.62*mV + 0.5*random.random_sample(size=(N['pre'],N['post']))) 
#Synaptic efficacies

model = leaky_IF(tau=tau['membrane'],El=V['reset']) + Current('I=ge:mV') +\
			 alpha_synapse(input='I_in',tau=tau['synapses'],unit=mV,output='ge')

brain = NeuronGroup(N=N,model=model,reset=V['reset'], threshold=V['threshold'])

recurrent = '''
				
			'''

feedforward = '''
	
			'''

afferents = Connection(inputs,brain,'ge',weight=W)
intrinsic = Synapses(brain,model=recurrent,pre='')

#Initialization
brain.v = (V['threshold']-V['reset'])*random.random_sample(size=(N['post'],)) + V['reset']

#Monitors
raster = {'pre':SpikeMonitor(inputs),'post':SpikeMonitor(brain)}
voltage_traces = StateMonitor(brain,'vm',record=True)
synaptic_potentials = StateMonitor(brain,'ge',record=True)

run(.5*second)

fig,axs = plt.subplots(nrows=3,ncols=1,sharex=True)

axs[0].plot(voltage_traces.times/ms,voltage_traces[0]/mV,'k',linewidth=2)
axs[0].set_xlabel('$\mathrm{Time \; (ms)} $')
axs[0].set_ylabel('$\mathrm{Membrane \; voltage \; (mV)} $')


axs[1].plot(synaptic_potentials.times/ms,synaptic_potentials[0]/mV,'k',linewidth=2) #Current misnamed
axs[1].set_xlabel('$\mathrm{Time \; (ms)}$')
axs[1].set_ylabel('$\mathrm{Synaptic \; potential \; (mV)}$')


raster_plot(raster['pre'],figure=fig)
axs[2].set_xlabel('$\mathrm{Time \; (ms)}$')
axs[2].set_ylabel('$\mathrm{Neuron \; no.}$')
plt.tight_layout()
plt.show()