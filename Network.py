import os, cPickle

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from time import time

READ = 'rb'
WRITE = 'wb'

class Network(object): #later make this inherit brian classes

	def __init__(self,N=None,M=None,W=None, duration=1000, memories=None,downsampling=10, mixing_fraction=0,
					ru_correl_matrix=None): #consider **kwargs
 		self.N = N if N else  {'neurons':100,'memories':10}
		self.duration = duration
		self.mixing_fraction = mixing_fraction

		self.mixing_fraction = mixing_fraction if mixing_fraction is np.ndarray or list else list(mixing_fraction)
		
		self.basename = self.timestamp()

		self.memories = cPickle.load(open(memories,READ)) if memories \
					else np.random.random_integers(0,high=1,size=(self.N['neurons'],self.N['memories']))
		
		for fraction in self.mixing_fraction:
			self.initialize(ru_params = ru_correl_matrix, mixing_fraction = fraction)
			self.run()
			self.save(downsample = downsampling, suffix = str(int(fraction*10)), basename = self.basename)

	#Everything belongs to self, don't need to pass so many arguments!

	def initialize(self, ru_params, mixing_fraction):
	
		#**kwargs will clean this up
		#These functions are getting spaghetti-like

		self.v = np.zeros((self.N['neurons'],self.duration),dtype=np.float16)
		self.u = np.zeros_like(self.v,dtype=np.float16)
		self.r = np.zeros_like(self.v,dtype=np.float16)

		self.M = np.zeros((self.N['neurons'],self.N['neurons'],self.duration),dtype=np.float16)
		self.W = np.zeros_like(self.M,dtype=np.float16)

		self.Quu = np.zeros((self.N['neurons'],self.N['neurons'],self.duration),dtype=np.float16)
		self.Qru = np.zeros_like(self.Quu,dtype=np.float16)
		self.Qvu = np.zeros_like(self.Quu,dtype=np.float16)

		arrs = np.random.multivariate_normal(ru_params['means'],ru_params['covariances'],
			size=(self.N['neurons'],self.duration)).astype(np.float16)

		r = arrs[:,:,0]
		u = arrs[:,:,1]

		self.M[:,:,0] = 4/float(self.N['memories'])*sum([np.outer(memory-0.5*np.ones_like(memory),
								memory-0.5*np.ones_like(memory))
					for memory in self.memories.T]) - 2/float(self.N['memories'])

		self.v[:,0] = (1-mixing_fraction)*self.memories[:,0] +\
							 mixing_fraction*(np.random.random_integers(0,high=1,size=(self.N['neurons'],)))

		self.W[:,:,0] = np.random.random_sample(size=(self.N['neurons'],self.N['neurons'])) #Assume same number of inputs for now
		self.r = np.random.random_sample(size=self.r.shape)
		self.u = np.random.random_sample(size=self.u.shape)

		self.u[:,400:600] = np.tile(self.memories[:,0],(200,1)).transpose()
		self.r[:,400:600] = np.ones((self.N['neurons'],200))

		self.I = np.eye(self.N['neurons'])

		self.epsilon = 0.001 #ratio of membrane time constant to timestep

	def run(self):
		for t in range(1,self.duration):
			
			self.K = np.linalg.inv(self.I-self.M[:,:,t-1])
			self.Quu[:,:,t] = np.outer(self.u[:,t-1],self.u[:,t-1])
			self.Qru[:,:,t] = np.outer(self.r[:,t-1],self.u[:,t-1])
			self.Qvu[:,:,t] = np.outer(self.r[:,t-1],self.u[:,t-1])

			self.v[:,t] = self.v[:,t-1] + self.epsilon*(-self.v[:,t-1] + self.M[:,:,t].dot(self.v[:,t-1]) )
			self.M[:,:,t] = self.M[:,:,t-1] + self.epsilon/10*(self.I-self.M[:,:,t-1] - np.outer(self.W[:,:,t-1].dot(self.u[:,t]),self.v[:,t]))
			self.W[:,:,t] = self.W[:,:,t-1] + self.epsilon/100.*(self.K.dot(self.W[:,:,t-1]).dot(self.Quu[:,:,t-1]).dot(self.Qru[:,:,t-1]-self.Qvu[:,:,t-1]))	

	def timestamp(self):
		return datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')

	def save(self,downsample=10, prefix='/Volumes/My Book/synchrony-data',suffix='',basename=None):

		self.results = {'v':self.v[:,::downsample],'M':self.M[:,:,::(downsample*10)],'W':self.W[:,:,::(downsample*10)],
						'Qru':self.Qru[:,:,::downsample],'Quu':self.Quu[:,:,::downsample],'Qvu':self.Qvu[:,:,::downsample],
						'r':self.r[:,::downsample],'u':self.u[:,::downsample],'memories':self.memories}

		self.basedir= os.path.join(prefix,basename if basename else self.timestamp())

		if not os.path.isdir(self.basedir):
			os.makedirs(self.basedir)

		self.writename = os.path.join(self.basedir,'results-%s.pkl'%(suffix))

		with open(self.writename,WRITE) as f:
			cPickle.dump(self.results,f)

	def quick_view(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.imshow(self.v,interpolation='nearest',aspect='auto')
		plt.colorbar(cax)
		plt.tight_layout()
		plt.show()
