import numpy as np
import matplotlib.pyplot as plt

N = 100
duration = 1000 #Assume dt= 0.1 ms

v = np.zeros((N,duration))
u = np.zeros_like(v)
r = np.zeros_like(v)

M = np.zeros((N,N,duration))
W = np.zeros_like(M)

Quu = np.zeros((N,N,duration))
Qru = np.zeros_like(Quu)
Qvu = np.zeros_like(Quu)

v[:,0] = np.random.random_sample(size=N)
M[:,:,0] = 0.1*(2*np.random.random_sample(size=(N,N))-1)
W[:,:,0] = np.random.random_sample(size=(N,N)) #Assume same number of inputs for now
r = np.random.random_sample(size=r.shape)
xcorr = lambda data: np.outer(data,data)

epsilon = 0.01 #ratio of tau to timestep
for t in range(1,duration):
	
	K = np.linalg.inv(np.eye(N)-M[:,:,t-1])
	Quu[:,:,t] = np.outer(u[:,t-1],u[:,t-1])
	Qru[:,:,t] = np.outer(r[:,t-1],u[:,t-1])
	Qvu[:,:,t] = np.outer(r[:,t-1],u[:,t-1])

	v[:,t] = -v[:,t-1] + epsilon*(v[:,t-1] + M[:,:,t].dot(v[:,t-1]) )
	M[:,:,t] = M[:,:,t-1] + epsilon/10.*(np.eye(N)-M[:,:,t-1] - np.outer(W[:,:,0].dot(u[:,t]),v[:,t]))
	W[:,:,t] = W[:,:,t-1] + epsilon/100.*(K.dot(W[:,:,t-1]).dot(Quu[:,:,t-1]).dot(Qru[:,:,t-1]-Qvu[:,:,t-1]))


fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(v,interpolation='nearest',aspect='auto')
plt.colorbar(cax)
plt.tight_layout()
plt.show()