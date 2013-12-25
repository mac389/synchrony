import numpy as np

class Network(object): #later make this inherit brian classes

	def __init__(self,size,M=None,W=None):
		self.size = size
		
		self.M = M if M else np.random.random_sample(size=(size,size))
		self.W = W if W else np.random.random_sample(size=(size,size))

	def simulate(self,duration):
		self.recording =
