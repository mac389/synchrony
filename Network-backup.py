import os, cPickle, random, itertools

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from time import time
from scipy import signal

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
					else 2*np.random.random_integers(0,high=1,size=(self.N['neurons'],self.N['memories']))-1

		for fraction in self.mixing_fraction:
			self.initialize(ru_params = ru_correl_matrix, mixing_fraction = fraction)
			self.run()
			#self.quick_view()
			self.save(downsample = downsampling, suffix = str(int(fraction*10)), basename = self.basename)

	#Everything belongs to self, don't need to pass so many arguments!


	def F(self,vector):
		answer = vector.copy()

		answer[answer>=0]=1
		answer[answer<0] = -1

		return answer

	def Fs(self,scalar):
		return -scalar if random.random() < 1/(1+np.exp(-scalar)) else scalar


	def gauss(other,n=200,sigma=50):
		xs = range(-int(n/2),int(n/2)+1)
		kern = np.array([1/(sigma*np.sqrt(2*np.pi))*np.exp(-float(x)**2/(2*sigma**2)) for x in xs])
		return kern

	def loggauss(other,n=200,sigma=.5):
		xs = np.linspace(0.001,3,num=n+1)
		kern = np.array([1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-np.log(float(x))**2/(2*sigma**2)) for x in xs])
		return kern/30.


	def initialize(self, ru_params, mixing_fraction):
	
		t = np.array(range(self.duration))

		#These functions are getting spaghetti-like
		self.ugen = {}
		self.ugen['frequency'] = 0.01 # Hz
		self.ugen['fill'] = -1
		self.ugen['buffer'] = 10
		self.ugen['chronic'] = lambda timepoints: signal.square(2*np.pi*self.ugen['frequency']*timepoints)
		self.ugen['exposure'] = lambda timepoints: np.lib.pad(self.ugen['chronic'](t)[:int(1/self.ugen['frequency'])],
													  (u['buffer'],len(timepoints)-int(1/self.ugen['frequency']+self.ugen['buffer'])),
													  'constant',constant_values=(self.ugen['fill'],self.ugen['fill']))
		self.ugen['cessation'] = lambda timepoints: np.lib.pad(self.ugen['chronic'](t)[:5*int(1/self.ugen['frequency'])],
													  (self.ugen['buffer'],len(timepoints)-5*int(1/self.ugen['frequency']+self.ugen['buffer'])),
													  'constant',constant_values=(self.ugen['fill'],self.ugen['fill']))

		self.rgen = {}
		self.rgen['susceptible'] = self.loggauss()
		self.rgen['resilient'] = self.gauss()

		self.v = np.zeros((self.N['neurons'],self.duration),dtype=np.float16)

		#NEXT ABSTRACT OVER SCHEMES
		self.u = np.tile(self.ugen['chronic'](t),(self.N['neurons'],1))


		#self.u = np.zeros_like(self.v,dtype=np.float16)
		self.r = np.zeros_like(self.v,dtype=np.float16)

		self.M = np.zeros((self.N['neurons'],self.N['neurons'],self.duration),dtype=np.float16)
		self.W = np.zeros_like(self.M,dtype=np.float16)

		self.Quu = np.zeros((self.N['neurons'],self.N['neurons'],self.duration),dtype=np.float16)
		self.Qru = np.zeros_like(self.Quu,dtype=np.float16)
		self.Qvu = np.zeros_like(self.Quu,dtype=np.float16)

		self.M[:,:,0] = np.array([np.outer(one,one) for one in self.memories.T]).sum(axis=0)
		self.M[:,:,0][np.diag_indices(self.N['neurons'])] = 0

		self.memory_stability = np.zeros((self.N['memories'],self.duration))

		self.v[:,0] = self.F((1-mixing_fraction)*self.memories[:,0] +\
							 mixing_fraction*(2*(np.random.random_integers(0,high=1,size=(self.N['neurons'],)))-1))

		self.W[:,:,0] = np.random.random_sample(size=(self.N['neurons'],self.N['neurons'])) #Assume same number of inputs for now

		self.I = np.eye(self.N['neurons'])

		self.epsilon = 0.001 #ratio of membrane time constant to timestep

		self.chosen_ones = [random.choice(xrange(self.N['neurons'])) for _ in xrange(1,self.duration)]

	def run(self):
		for t,idx in zip(range(1,self.duration),self.chosen_ones):
			
			self.K = np.linalg.inv(self.I-self.M[:,:,t-1])
			self.Quu[:,:,t] = np.outer(self.u[:,t-1],self.u[:,t-1])
			self.Qru[:,:,t] = np.outer(self.r[:,t-1],self.u[:,t-1])
			self.Qvu[:,:,t] = np.outer(self.r[:,t-1],self.u[:,t-1])

			self.v[:,t] = self.v[:,t-1]
			self.M[:,:,t] = self.M[:,:,t-1]

			I = self.M[idx,:,t].dot(self.v[:,t])

			self.v[idx,t] =  1 if I >=0 else -1

			self.M[:,:,t] = self.M[:,:,t-1] + self.epsilon/10*(self.I-self.M[:,:,t-1] - np.outer(self.W[:,:,t-1].dot(self.u[:,t]),self.v[:,t]))
			self.W[:,:,t] = self.W[:,:,t-1] + self.epsilon/100.*(self.K.dot(self.W[:,:,t-1]).dot(self.Quu[:,:,t-1]).dot(self.Qru[:,:,t-1]-self.Qvu[:,:,t-1]))	

			self.memory_stability[:,t] = -0.5*np.array([memory.dot(self.M[:,:,t]).dot(memory) + memory.dot(self.W[:,:,t]).dot(self.u[:,t]) 
										for memory in self.memories.T])

	def timestamp(self):
		return datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')

	def save(self,downsample=10, prefix='/Volumes/My Book/synchrony-data',suffix='',basename=None):

		self.results = {'v':self.v[:,::downsample],'M':self.M[:,:,::(downsample*10)],'W':self.W[:,:,::(downsample*10)],
						'Qru':self.Qru[:,:,::downsample],'Quu':self.Quu[:,:,::downsample],'Qvu':self.Qvu[:,:,::downsample],
						'r':self.r[:,::downsample],'u':self.u[:,::downsample],'memories':self.memories,
						'memory_stability':self.memory_stability}

		self.basedir= os.path.join(prefix,basename if basename else self.timestamp())

		if not os.path.isdir(self.basedir):
			os.makedirs(self.basedir)

		self.writename = os.path.join(self.basedir,'results-%s.pkl'%(suffix))

		with open(self.writename,WRITE) as f:
			cPickle.dump(self.results,f)

	def quick_view(self):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.imshow(self.v,interpolation='nearest',aspect='auto', cmap=plt.cm.binary)

		fig.colorbar(cax)
		fig.tight_layout()

		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111)
		cax2 = ax2.imshow(self.M[:,:,0],interpolation='nearest',aspect='auto',cmap=plt.cm.binary)
		fig2.colorbar(cax2)
		fig2.tight_layout()
		plt.show()