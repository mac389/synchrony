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
			Raster     Neuron   |
					            | Color = activity
					            |_____________
					            	Time	
			
			R-trace             |____________ 


			U-trace             |____________

The data from this are chosen from a simulation where the initial conditions contain no noise. 

'''

#Generate file names to load data
#results--v-0-r-u.pkl
basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}

#will eventually need subplot2grid because u,r 1/3 space of heat maps
fig,axs = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True)
for reward,row in zip(r_schema,axs):
	for stimulus,col in zip(u_schema,row):
		col.imshow(data['%s-%s'%(stimulus,reward)]['network_stability'],aspect='auto',interpolation='nearest')

		artist.adjust_spines(col
		col.set_xlabel(artist.format('Time'))
		col.set_ylabel(r'\Large $E\left(\mathbf{memory}\right)$')

plt.tight_layout()		
plt.show()