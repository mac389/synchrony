from brian import *
from brian.library.synapses import alpha_synapse
from brian.library.IF import leaky_IF

from numpy import random

N = {'pre':25,'post':1}


V = {'reset':-60*mV,'threshold':=40*mV}
tau = {'membrane':10 * ms}

model = Equations('''
	dv/dt = -(v-Vr)/tau :volt
	'''
	)

brain = NeuronGroup(N=N,model=model,reset=Vr, threshold=Vt)

#Initialization
brain.v = (Vt-Vr)*random.random_sample(size=(N,)) + Vr


#Monitors
raster = SpikeMonitor(brain)
voltage_trace = StateMonitor(brain,'v',record=True)

run(2*second)

raster_plot(raster)

figure()
imshow(voltage_trace.values/mV,aspect='auto',interpolation='nearest')
colorbar()
show()