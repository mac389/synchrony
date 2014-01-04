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

angles = {key:compute_angles(value['M']) for key,value in data.iteritems()}
def circular_variance(angles):
	n = len(angles)
	return 1-np.sqrt(np.square(np.cos(angles)).sum()+np.square(np.sin(angles)).sum())/float(n)

z= angles['susceptible-exposure']
'''
cvs = np.zeros((10,len(r_schema)+len(u_schema)))
iqr = lambda data: 0.5*(scoreatpercentile(data,75)-scoreatpercentile(data,25))
for i,reward in enumerate(r_schema):
	for j,stimulus in enumerate(u_schema):
		cvs[:,(i*len(r_schema)+j)] = np.array([circular_variance(row) for row in angles['%s-%s'%(reward,stimulus)]])
print cvs
'''
'''
y = np.fft.fft(z[0,:])/float(len(z[0,:])) #Normalization
y = 10*np.absolute(y/y[0])
x = np.fft.fftfreq(len(y),1)
cutoff = scoreatpercentile(y,95)

fig = plt.figure()
ax = fig.add_subplot(111)
markerline, stemlines, baseline = ax.stem(x[:len(x)/2],y[:len(y)/2],linefmt='k',
	markerfmt='ko', basefmt='k-')

for line,point, in zip(stemlines,y[:len(y)/2]):
	color = 'k' if point > cutoff else '0.5'

ax.axhline(y=cutoff,color='r',linewidth=2,linestyle='--')
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Frequency'))
ax.set_ylabel(artist.format('Power (dB)'))
plt.show()
'''
fig,axs = plt.subplots(nrows=2,ncols=3)
for reward,ax in zip(r_schema,axs):
	for stimulus,col in zip(u_schema,ax):
		res = angles['%s-%s'%(reward,stimulus)][-1,:]

		col.acorr(res, maxlags=50,lw=2,usevlines=True)
		col.axhline(0,color='k',lw=2)

		col.set_ylabel(artist.format(stimulus.capitalize()))
		col.set_xlabel(artist.format('Lags, %s'%reward.capitalize()))

plt.tight_layout()
plt.show()