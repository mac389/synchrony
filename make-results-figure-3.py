import os,cPickle

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import rcParams

rcParams['text.usetex'] = True

'''
	Figure layout:
														U-schema

									   Exposure			Continued Used			Cessation
					Susceptible
		R-schema

					Resilient

	Each panel of this 2 X 3 figure has three parts:

						   		|
			Raster     Memory   |
					            | Color = activity
					            |_____________
					            	Time	

The data from this are chosen from a simulation where the initial conditions contain no noise. 

'''

#Generate file names to load data
#results--v-0-r-u.pkl
basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}

cmin = -5000
cmax = 5000
#will eventually need subplot2grid because u,r 1/3 space of heat maps

fig = plt.figure()
ncols = 4
nrows = 2
grid = gridspec.GridSpec(nrows,ncols,width_ratios=[4,4,4,1])
for reward,i in zip(r_schema,range(nrows)):
	for stimulus,j in zip(u_schema,range(ncols)):
		if j < (ncols-1):
			panel = plt.Subplot(fig,grid[i,j])

			im = panel.imshow(data['%s-%s'%(reward,stimulus)]['memory_stability'],interpolation = 'nearest',
				aspect='auto',vmin=cmin,vmax=cmax)

			artist.adjust_spines(panel)
			panel.set_xlabel(artist.format('Time, %s'%stimulus.capitalize()))
			panel.set_ylabel(artist.format('Memory, %s'%reward.capitalize()))

			fig.add_subplot(panel)

cax = fig.add_subplot(plt.Subplot(fig,grid[:,ncols-1]))
cbar = plt.colorbar(im,cax=cax)
cbar.set_label(r'\Large \textbf{Stability, } $E\left(\mathrm{Memory}\right)$')

grid.tight_layout(fig)
plt.show()