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
fig = plt.figure(figsize=(10,8))
outer_grid = gridspec.GridSpec(2,3)
for reward,i in zip(r_schema,range(2)):
	for stimulus,j in zip(u_schema,range(3)):
		inner_grid = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=outer_grid[i*3+j],height_ratios=[2,1,1])
		col = plt.Subplot(fig,inner_grid[0])
		col.imshow(data['%s-%s'%(reward,stimulus)]['v'],interpolation='nearest',aspect='auto',cmap=plt.cm.binary)

		artist.adjust_spines(col,['left'])
		col.set_ylabel(artist.format(reward.capitalize()))
		fig.add_subplot(col)

		utrace = plt.Subplot(fig,inner_grid[1])
		utrace.plot(data['%s-%s'%(reward,stimulus)]['u'][:5000][::50],'k',linewidth=2)

		artist.adjust_spines(utrace,['left'])
		utrace.set_yticks([-1,1])
		utrace.set_yticklabels(map(artist.format,['Not using','Using']))

		fig.add_subplot(utrace)
		rtrace = plt.Subplot(fig,inner_grid[2])
		rtrace.plot(data['%s-%s'%(reward,stimulus)]['r'][:5000],'k',linewidth=2)

		artist.adjust_spines(rtrace)
		rtrace.set_xlabel(artist.format('Time, %s'%stimulus.capitalize()))
		rtrace.set_yticks([-1,0,1])
		rtrace.set_ylim((-2.3,1))
		rtrace.set_yticklabels(map(artist.format,['Harm','Neutral','Reward']))

		fig.add_subplot(rtrace)
plt.tight_layout()		
plt.show()