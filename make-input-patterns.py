import cPickle

import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist

from scipy import signal, ndimage
from matplotlib import rcParams

rcParams['text.usetex'] = True


def gauss(n=200,sigma=50):
	xs = range(-int(n/2),int(n/2)+1)
	kern = np.array([1/(sigma*np.sqrt(2*np.pi))*np.exp(-float(x)**2/(2*sigma**2)) for x in xs])
	return kern

def loggauss(n=200,sigma=.5):
	xs = np.linspace(0.001,3,num=n+1)
	kern = np.array([1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-np.log(float(x))**2/(2*sigma**2)) for x in xs])
	return kern/30.



u = {}
u['frequency'] = 0.01 # Hz
u['fill'] = -1
u['buffer'] = 10
u['chronic'] = lambda timepoints: signal.square(2*np.pi*u['frequency']*timepoints)
u['exposure'] = lambda timepoints: np.lib.pad(u['chronic'](t)[:int(1/u['frequency'])],
											  (u['buffer'],len(timepoints)-int(1/u['frequency']+u['buffer'])),
											  'constant',constant_values=(u['fill'],u['fill']))
u['cessation'] = lambda timepoints: np.lib.pad(u['chronic'](t)[:5*int(1/u['frequency'])],
											  (u['buffer'],len(timepoints)-5*int(1/u['frequency']+u['buffer'])),
											  'constant',constant_values=(u['fill'],u['fill']))

r = {}
r['susceptible'] = loggauss()
r['resilient'] = gauss()

u['therapy'] = lambda timepoints: 2*random.randint(1,size=len(timepoints))-1
r['therapy'] = lambda timepoints: 2*random.randint(1,size=len(timepoints))-1
#cPickle.dump({'r':r,'u':u},open('ru.pkl','wb'))
t = np.linspace(0,1000,num=1001)
'''
fig,axs = plt.subplots(nrows=2,ncols=3)
for j,(r_schema,row) in enumerate(zip(['susceptible','resilient'],axs)):
	for i,(u_schema, col) in enumerate(zip(['exposure','chronic','cessation'],row)):
		col.plot(np.convolve(u[u_schema](t),r[r_schema]),'k',linewidth=2)
		artist.adjust_spines(col)

		if i ==0:
			col.set_ylabel(artist.format(r_schema.capitalize()))

		if j ==1:
			col.set_xlabel(artist.format(u_schema.capitalize()))
		col.set_xticklabels([])
'''
'''
fig,axs = plt.subplots(ncols=2,nrows=1)
for scheme,ax in zip(['susceptible','resilient'],axs):
	ax.plot(r[scheme],'k',linewidth=2)
	artist.adjust_spines(ax)

	ax.set_xticklabels([])
	ax.set_xticks([])


	ax.annotate(artist.format(scheme.capitalize()), xy=(.8, .5),  xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center')
'''
'''
fig,axs = plt.subplots(nrows=3,ncols=1)
for scheme,ax in zip(['exposure','chronic','cessation'],axs):
	ax.plot(u[scheme](t),'k',linewidth=2)
	artist.adjust_spines(ax)

	ax.set_xlabel(r'\Large \textbf{%s, Time}'%scheme.capitalize())
	ax.set_yticks([-1,1])
	ax.set_ylim((-1.5,1.5))
	ax.set_yticklabels(map(artist.format,['not using','using']))
	ax.set_xticklabels([])
'''
'''
fig,axs = plt.subplots(nrows=3,ncols=2)
for reward,ax in zip(r,axs):
	for stimulus,panel in zip(['exposure','chronic','cessation'],ax):
		panel.xcorr(r[reward],u[stimulus](t),usevlines=True,normed=True,linewidth=2,maxlags=100)
		panel.axhline(y=0,color='k',linewidth=2)

fig.tight_layout()
plt.show()
'''