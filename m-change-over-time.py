import cPickle,os
import numpy as np

import Graphics as artist
import matplotlib.pyplot as plt

from matplotlib import rcParams

rcParams['text.usetex'] = True

basepath = '/Volumes/My Book'
r_schema = ['susceptible','resilient'] #add therapy in
u_schema = ['exposure','chronic','cessation']

#Need to add in Trellis formatting

READ = 'rb'
bins = 100
maxlags = 200
fig,axs = plt.subplots(nrows=len(r_schema),ncols=len(u_schema),sharex=True, sharey=True)
for reward,ax in zip(r_schema,axs):
	print reward
	for stimulus, panel in zip(u_schema,ax):
		print stimulus
		filename = os.path.join(basepath,'heatmap-%s-%s.pkl'%(reward,stimulus))
		with open(filename,READ) as stream:
			data = cPickle.load(stream)

			counts,edges = np.histogram(data[np.tril_indices(data.shape[0],k=-1)],bins=np.linspace(-maxlags,maxlags,num=bins))
			panel.bar(edges[:-1],np.log(counts),color='k')

			artist.adjust_spines(panel)

			panel.set_xlabel(artist.format('Peak of correlation function'))
			panel.set_ylabel(artist.format('Log Count'))
			del data

fig.tight_layout()
plt.show()