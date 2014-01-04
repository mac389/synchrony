import os,cPickle

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import analysis as postdoc

from matplotlib import rcParams
from scipy.stats import pearsonr,scoreatpercentile


def compute_angles(mat):

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


		data[:,t] = np.arccos([np.real(a.dot(np.conjugate(b)))/(np.linalg.norm(a)*np.linalg.norm(b)) for a,b in zip(base_vectors,these_vectors)])
		#Scale by eigenvalues?
	return data

rcParams['text.usetex'] = True

basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}

fig,axs = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True)
for reward,ax in zip(r_schema,axs):
	for stimulus,panel in zip(u_schema,ax):
		v = data['%s-%s'%(reward,stimulus)]['v'].mean(axis=0)
		panel.acorr(v,usevlines=True, maxlags=150, linewidth=2)
		panel.axhline(y=0,color='k',linewidth=2)

		artist.adjust_spines(panel)
		panel.set_xlabel(artist.format('Lags, %s'%(stimulus.capitalize())))
		panel.set_ylabel(artist.format(reward.capitalize()))
		panel.set_xlim(-150,150)
		panel.grid(True)

plt.tight_layout()
plt.show()