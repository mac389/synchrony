import os,cPickle

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import rcParams
from numpy.core.umath_tests import inner1d
rcParams['text.usetex'] = True


basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation','therapy']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb'))['M'] for r in r_schema for u in u_schema}

#and then W
keys = data.keys() # to freeze the order of the keys

comps = {'%s:%s'%(one,two):inner1d(data[one]['v'],data[two]['v']) for one in keys for two in keys}

