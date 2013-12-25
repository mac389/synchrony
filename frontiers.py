import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import Graphics as artist

#Make initial connection matrix
from numpy.random import random_sample, random_integers, random
from matplotlib import rcParams

rcParams['text.usetex'] = True

timesteps = 100
n = 100

v = np.zeros((n,timesteps))
energies = np.zeros((timesteps,))
dt = 0.001

#create memory fixed points
nMem = n/10

memories = random_integers(0,high=1,size=(n,nMem))
#this means alpha = 0.5

M = 4/float(nMem)*sum([np.outer(memory-0.5*np.ones_like(memory),
						memory-0.5*np.ones_like(memory))
						 for memory in memories.T]) - 2/float(nMem)


u = random_sample((n,))
W = 2*random_sample(size=(u.shape[0],n))-1
F  = lambda x: 1./(1+np.exp(-x))

def energy(v,u,M,W):
	return -(np.dot(u,np.dot(W,v)) + 0.5*np.dot(v,np.dot(M,v)))
	
#initial conditions
v[:,0] = random_integers(0,high=1,size=v[:,0].shape)

for t in range(1,timesteps):
	v[:,t] = F(v[:,t-1] + dt*(-v[:,t-1] + np.dot(M,v[:,t-1]) + np.dot(W,u)))>random()
	energies[t] = energy(u,v[:,t],M,W)


fig,axs = plt.subplots(nrows=1,ncols=2)
axs[0].imshow(memories,interpolation='nearest',aspect='auto',cmap=plt.cm.binary)
cax = axs[1].imshow(v,interpolation='nearest',aspect='auto',cmap=plt.cm.binary)
plt.colorbar(cax)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(energies)


plt.tight_layout()
plt.show()