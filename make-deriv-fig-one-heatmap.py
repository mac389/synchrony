import cPickle

import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist 

from pprint import pprint
from matplotlib import rcParams
from matplotlib import cm as CM
from scipy.stats import ks_2samp

rcParams['text.usetex'] = True

data = cPickle.load(open('figure-1-deriv-data-cricular.pkl','rb'))

settings = {'interpolation':'nearest','aspect':'auto', 'vmax':1}

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

monikers = ['%s-%s'%(reward,stimulus) for reward in r_schema for stimulus in u_schema]
img = np.array([[data['%s:%s'%(one,two)][0] for one in monikers] for two in monikers])


susceps = [data[x][0] for x in data if 'susceptible' in x]
resil = [data[x][0] for x in data if 'resilient' in x]

print ks_2samp(susceps,resil),'all'

img[np.triu_indices(img.shape[0],k=1)] = None
img[np.diag_indices_from(img)]=1
fig = plt.figure()
ax = fig.add_subplot(111)

cmap = CM.get_cmap('jet')
cmap.set_bad('w')
settings['cmap'] = cmap

cax = ax.imshow(np.tril(img),**settings)

artist.adjust_spines(ax)
ax.set_xticks(range(6))
ax.set_yticks(range(6))

ax.set_xticklabels(map(artist.format,[x.capitalize() for x in u_schema*2]),range(6),rotation=60)
ax.set_yticklabels(map(artist.format,[x.capitalize() for x in u_schema*2]),range(6))

cbar = plt.colorbar(cax)
cbar.set_label(r'\Large $ \langle \cos \theta \rangle$', rotation='horizontal')
fig.tight_layout()
plt.show()
