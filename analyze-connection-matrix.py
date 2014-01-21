import os,cPickle, itertools

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import analysis as postdoc

from matplotlib import rcParams
from scipy.stats import pearsonr
from scipy.signal import detrend,fftconvolve
from scipy.ndimage import filters

from progress.bar import Bar
from time import time
rcParams['text.usetex'] = True


'''


'''

#Generate file names to load data
basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient'] #add therapy in
u_schema = ['exposure','chronic','cessation']


filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}


def get_peak_shift(correl_function):
	return len(correl_function)/2-np.argmax(np.absolute(correl_function))

#Iterate using itertools?
#How to flatten the third dimension? Start witha  simple example, I don't really know what's going on with roll axis



x = data['susceptible-exposure']['M']
z = np.array([detrend(row) for sheet in x for row in sheet])

CCF_peaks = {}
nops = len(data)*len(z)*(len(z)-1)/2
bar = Bar("Calculating correlations",max = nops)
start = time()
for k,condition in enumerate(['susceptible-cessation']):#enumerate(data.keys()):
	y = np.array([detrend(row) for sheet in data[condition]['M'] for row in sheet])
	heatmap = np.zeros((len(y),len(y)),dtype=np.int8)
	for i in range(len(y)):
		for j in range(i-1):
			heatmap[i,j] = get_peak_shift(fftconvolve(y[i],y[j][::-1],mode='same'))
			bar.next()
		bar.next() 
	print 'ETA %.02f hours'%((time()-start)/(3600.)*len(data)*len(y)*(len(y)-1)/2/((k+i+1)*len(y)))
	bar.next()
	CCF_peaks[condition] = heatmap
	cPickle.dump(heatmap,open('/Volumes/My Book/heatmap-%s.pkl'%condition,'wb'))
	del heatmap
bar.finish()

cPickle.dump(CCF_peaks,open('/Volumes/My Book/heatmaps-figure-4.pkl','wb'))
'''
fig  = plt.figure()
ax = fig.add_subplot(111)
ax.hist(heatmap,color='k', bins=100, range=[-150,150])
plt.show()
'''

'''
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(heatmap,interpolation='nearest',aspect='auto')
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Synaptic weight'))
ax.set_ylabel(artist.format('Synaptic weight'))

cbar = plt.colorbar(cax)
cbar.set_label(artist.format('Shift in peak'))
plt.savefig('/Volumes/My Book/M-synchrony.tif',dpi=300)
plt.close()
'''
'''
fig = plt.figure()
ax = fig.add_subplot(111)
lag = 300
dd = np.correlate(detrend(x[1,2,:]),detrend(x[1,1,:]),mode='same')
peak = lag-np.argmax(np.absolute(dd))
ax.xcorr(detrend(x[1,2,:]),detrend(x[1,1,:]),color='k',linewidth=2,maxlags=lag)
ax.axvline(x=0)
plt.show()
'''