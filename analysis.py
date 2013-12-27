import cPickle, visualization

import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist


def load_data(path_to_data):
	return cPickle.load(open(path_to_data,'rb'))

def accuracy_figure(data,**kwargs):

	voltage = data['v']
	start = voltage[:,0]
	stop = voltage[:,-1]

	target_memory = data['memories'][:,0] #Assume targeting 0th memory

	accuracy = accuracy_trace(voltage, target_memory)
	visualization.accuracy_plot(start,accuracy,stop,target_memory,**kwargs)

	return np.average(accuracy[-200:])

def accuracy_trace(voltage,target):

	return voltage.transpose().dot(target)/float(voltage.shape[0])

dq = lambda data: abs(np.diff(map(abs,map(np.linalg.det,np.rollaxis(data.astype(np.float32),2)))))

def q_star(voltage,target):

	accuracy = accuracy_trace(voltage,target)

	#q_start is the maximum value of the accruacy that is sustained for more than some time
	#starting out with just the accuracy
	return accuracy.max()

def sensitivity(x,y, show=False, savename=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x,y,'k',linewidth=2)

	artist.adjust_spines(ax)
	ax.set_xlabel(r'\Large \textbf{Mixing fraction,} $\; \alpha$')
	ax.set_ylabel(r'\Large textbf{Maximum accuracy,}$\; q_{\max}$')
	plt.tight_layout()
	if savename:
		plt.savefig('%s.png'%savename,dpi=300)

	if show:
		plt.show()

def correlation_visualization(data, show=False,savename=None):
	correlations = ['Quu','Qru','Qvu']
	#Analyze correlations

	fig,axs = plt.subplots(nrows=3,ncols=1,sharex=True)
	for ax,data,label in zip(axs,map(dq,[data[correlation] for correlation in correlations]),correlations):
		ax.plot(data,'k',linewidth=2,label=artist.format(label))

		artist.adjust_spines(ax)

		ax.set_ylabel(r'\Large $\mathbf{\partial \left(\det %s_{%s}\right)}$'%(label[0],label[1:]),rotation='horizontal')
		ax.set_xlabel(artist.format('Time'))

	plt.tight_layout()
	if savename:
		plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()
