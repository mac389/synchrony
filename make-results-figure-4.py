import os,cPickle

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import analysis as postdoc

from matplotlib import rcParams
from scipy.stats import pearsonr

rcParams['text.usetex'] = True


'''
	Figure layout:
														U-schema

									   Exposure			Continued Used			Cessation
					Susceptible
		R-schema

					Resilient

	Each panel of this 2 X 3 figure has three parts:

						   		|   3 traces
						   		|  {Exposure, Chronic,
			Raster     q(v)     |       Cessation}
					            | 
					            |_____________
					            	Alpha	
			

The data from this are chosen from a simulation where the initial conditions contain no noise. 

'''

#Generate file names to load data
#results--v-0-r-u.pkl
basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}

formats = dict(zip(u_schema,['k','k--','k-.']))
sn = lambda alpha: (1-alpha)/alpha if alpha > 0 else 0

mixing_fractions = np.linspace(0,1,num=11)

accuracies = {}

fig,axs = plt.subplots(ncols=len(r_schema))
for reward,panel in zip(r_schema,axs):
	for stimulus in u_schema:

		accuracy = postdoc.accuracy_figure(data['%s-%s'%(reward,stimulus)],savename=None)
		panel.plot(artist.smooth(accuracy,beta=2),formats[stimulus],linewidth=2,label=artist.format(stimulus.capitalize()))

		accuracies['%s-%s'%(reward,stimulus)] = accuracy

		artist.adjust_spines(panel)

		panel.set_xlabel(r'\Large $\mathrm{\frac{Signal}{Noise}}$')
		panel.set_ylabel(r'\Large \textbf{%s, } $\mathrm{Accuracy, q\left(\mathbf{v}\right)\Bigg|_{\mathbf{v^0}}} $'%reward.capitalize())	

		panel.set_ylim((-1,1))
		xlabs = [r'\Large $\mathbf{%.02f}$'%alpha for alpha in map(sn,mixing_fractions[:-1])]
		xlabs[0] = r'\Large $\mathrm{All \; signal}$'
		xlabs[-1] = r'\Large $\mathrm{All \; noise}$'

		panel.set_xticklabels(xlabs, rotation='vertical')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()		


keys = accuracies.keys()

for one in keys:
	for two in keys:

		r,p = pearsonr(accuracies[one],accuracies[two])
		print one,two, r if p < 0.05 else 0