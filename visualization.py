import Graphics as artist
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

rcParams['text.usetex'] = True

def accuracy_plot(start,accuracy, stop, memory, show=False, savename=None):

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

	accuracy_p.annotate(r'\Large $q\left(t\right) = \frac{1}{N} \mathbf{v}\left(t\right)\cdot \mathbf{v_{target}}$', xy=(.6, .2),  
		xycoords='axes fraction',horizontalalignment='center', verticalalignment='center')


	accuracy_p.set_ylim((-1,1))

	start_plot.set_xticklabels([])
	end_plot.set_xticklabels([])
	target_plot.set_xticklabels([])

	plt.tight_layout()

	if savename:
		plt.savefig('%s.png'%savename,dpi=200)
	if show:
		plt.show()

