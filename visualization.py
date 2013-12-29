import Graphics as artist
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

rcParams['text.usetex'] = True


def accuracy_plot(start,accuracy, stop, memory,idx=0,savename=''):

	fig = plt.figure()
	start_plot = plt.subplot2grid((1,5),(0,0))
	accuracy_p = plt.subplot2grid((1,5),(0,1),colspan=2)
	end_plot = plt.subplot2grid((1,5),(0,3))
	target_plot = plt.subplot2grid((1,5),(0,4))
	
	parameters = {'interpolation':'nearest','aspect':'auto', 'cmap':plt.cm.binary}

	start_plot.imshow(start[:,np.newaxis],**parameters)
	accuracy_p.plot(accuracy,'k',linewidth=2)
	end_plot.imshow(stop[:,np.newaxis],**parameters)
	target_plot.imshow(memory[:,np.newaxis],**parameters)

	artist.adjust_spines(accuracy_p)

	for ax in [start_plot,end_plot,target_plot]:
		artist.adjust_spines(ax,['left'])

	accuracy_p.annotate(r'\Large $q\left(t\right) = \frac{1}{N} \mathbf{v}\left(t\right)\cdot \mathbf{v_{target}}$', 
		xy=(.6, .2),  xycoords='axes fraction',horizontalalignment='center', verticalalignment='center')

	accuracy_p.set_ylim((-1,1))

	start_plot.set_xticklabels([])
	start_plot.set_xlabel(artist.format('Start'))

	end_plot.set_xticklabels([])
	end_plot.set_xlabel(artist.format('End'))
	
	target_plot.set_xticklabels([])
	target_plot.set_xlabel(artist.format('Target'))

	fig.tight_layout()

	plt.savefig('%s-memory-%d.png'%(savename,idx),dpi=200)

dq = lambda data: map(np.linalg.det,np.rollaxis(data.astype(np.float32),2))

def memory_stability(mat,savename):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.imshow(mat,interpolation='nearest',aspect='auto')
	
	artist.adjust_spines(ax)
	ax.set_ylabel(artist.format('Memory'))
	ax.set_xlabel(artist.format('Time'))

	cbar = plt.colorbar(cax)
	cbar.set_label(artist.format('Energy'))
	plt.savefig('%s.png'%savename,dpi=200)

def network_stability(energy_trace,savename):

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(energy_trace,'k',linewidth=2)
	artist.adjust_spines(ax)
	ax.set_xlabel(artist.format('Time'))
	ax.set_ylabel(artist.format('Stability (energy)'))

	plt.savefig('%s.png'%savename,dpi=200)

def track_matrices(mat,savename):

	mat = mat.astype(np.float32)
	base = mat[:,:,0]

	base_values,base_vectors = np.linalg.eig(base)

	idx = np.argsort(base_values)[::-1] #Want eigenvectors in descending order

	base_vectors = base_vectors[idx][:10] #Now eigvectors are in descending order, display top 10
	base_norm = np.linalg.norm(base_vectors)
	DURATION = mat.shape[2]
	#Awful magic constant of 10 eigenvectors
	data = np.zeros((10,DURATION))
	for t in range(1,DURATION):
		these_values,these_vectors = np.linalg.eig(mat[:,:,t])
		this_idx = np.argsort(these_values)[::-1]
		these_vectors = these_vectors[this_idx][:10]

		data[:,t] = [a.dot(b.T)/(np.linalg.norm(a)*np.linalg.norm(b)) for a,b in zip(base_vectors,these_vectors)]
		#Scale by eigenvalues?
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax= ax.imshow(data,interpolation='nearest',aspect='auto')
	artist.adjust_spines(ax)
	ax.set_xlabel(artist.format('Time'))
	ax.set_yticks(range(10))
	ax.set_yticklabels([r'\Large $\mathbf{e_%d\left(0\right) \cdot e_%d}$'%(x,x) for x in range(10)])

	ax.set_xticks(range(mat.shape[2])[::10])
	ax.set_xticklabels([10*x for x in range(mat.shape[2])[::10]])

	plt.colorbar(cax)
	fig.tight_layout()
	plt.savefig('%s.png'%savename,dpi=200)