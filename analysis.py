import cPickle, visualization

import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist

from scipy.stats import pearsonr

def load_data(path_to_data):
	return cPickle.load(open(path_to_data,'rb'))

def accuracy_figure(data,savename):
	''' Calculate the angle between each memory and the activity pattern at each timestep'''
	voltage = data['v']
	start = voltage[:,0]
	stop = voltage[:,-1]

	N = voltage.shape[0]
	accuracies = np.zeros((data['memories'].shape[1],)) 
	
	for i in xrange(data['memories'].shape[1]):
		memory = data['memories'][:,i]
		accuracy = voltage.transpose().dot(memory)/float(N)	
		accuracies[i] = accuracy[-200:].mean()
		visualization.accuracy_plot(start,accuracy,stop,memory,idx=i, savename=None)
	return accuracies

sn = lambda alpha: (1-alpha)/alpha if alpha > 0 else 0


def sensitivities(x,im,show=False, savename=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	cax = ax.imshow(im,interpolation='nearest',aspect='auto', vmin=-1,vmax=1) 
	cbar = plt.colorbar(cax)

	artist.adjust_spines(ax)
	ax.set_xticks(range(len(x)))
	xlabs = [r'\Large $\mathbf{%.02f}$'%alpha for alpha in map(sn,x)]
	xlabs[0] = r'\Large $\mathrm{All \; signal}$'
	xlabs[-1] = r'\Large $\mathrm{All \; noise}$'
	ax.set_xticklabels(xlabs)
	ax.set_xlabel(r'\Large $\mathrm{\frac{Signal}{Noise}}$')
	ax.set_ylabel(r'\Large $\mathrm{Pattern} $', rotation='horizontal')
	cbar.set_label(r'\Large $\mathrm{Accuracy,} \; q_{\max}$')

	plt.tight_layout()
	if savename:
		plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()
	plt.close()
	return [pearsonr(row,x) for row in im]

def sensitivity(x,y, show=False, savename=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x,y,'ko--',linewidth=2,clip_on=False)

	artist.adjust_spines(ax)
	ax.set_xlabel(r'\Large \textbf{Mixing fraction,} $\; \alpha$')
	ax.set_ylabel(r'\Large \textbf{Maximum accuracy,}$\; q_{\max}$')
	ax.set_ylim((-1.1))
	plt.tight_layout()

	if savename:
		plt.savefig('%s.png'%savename,dpi=300)
	if show:
		plt.show()
	plt.close()
	return pearsonr(x,y)

dq = lambda data: abs(np.diff(map(abs,map(np.linalg.det,np.rollaxis(data.astype(np.float32),2)))))

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
	plt.close()