import os,cPickle

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import rcParams

rcParams['text.usetex'] = True

'''
	Figure layout


		|  3 traces								| 3 traces
	E(v)|  {Exposure, Chronic,				E(v)|  {Exposure, Chronic
		|     Cessation}						|     Cessation}
		|___________ 							|______________________
			t 											t

			Susceptible								 Resilient
'''

basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}

formats = dict(zip(u_schema,['k','k--','k-.']))

fig,axs = plt.subplots(nrows=1,ncols=2,sharex=True,sharey=True)
for reward,ax in zip(r_schema,axs):
	for stimulus in u_schema:
		ax.plot(artist.smooth(data['%s-%s'%(reward,stimulus)]['network_stability']),formats[stimulus],
			linewidth=2,label=artist.format(stimulus.capitalize()))
		plt.hold(True)

	ax.annotate(artist.format(reward.capitalize()), xy=(.2, .7),  xycoords='axes fraction',
    	horizontalalignment='center', verticalalignment='center')

	artist.adjust_spines(ax)
	ax.set_xlabel(artist.format('Time'))
	ax.set_ylabel(r'\Large $E\left(\mathbf{v}\right)$')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()