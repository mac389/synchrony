import os,cPickle

import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import rcParams
from numpy.core.umath_tests import inner1d
from pprint import pprint
from scipy.stats import scoreatpercentile

rcParams['text.usetex'] = True


basedir = '/Volumes/My Book/synchrony-data/2013-12-29-13-26-12'

r_schema = ['susceptible','resilient']
u_schema = ['exposure','chronic','cessation']

filename = lambda ru: os.path.join(basedir,'all-results-0-%s-%s.pkl'%(ru[0],ru[1]))
data = {'%s-%s'%(r,u):cPickle.load(open(filename((r,u)),'rb')) for r in r_schema for u in u_schema}


iqr = lambda data: 0.5*(scoreatpercentile(data,75)-scoreatpercentile(data,25))

def angle(one,two):
	return np.absolute(one.dot(two)/(np.linalg.norm(one)*np.linalg.norm(two)))

def circ_mean(data):
	return np.arctan((np.sin(data).sum() + np.cos(data).sum())/float(len(data)))

def circ_var(data):
	return 1- np.sqrt((np.cos(data).sum()**2 + np.sin(data).sum()**2))/float(len(data))

def summarize(data):
	#Mean and SEM
	return (circ_mean(data),circ_var(data),len(data))

def angles(first,second):
	return summarize(np.array([angle(one,two) for one in first for two in second]))

#and then W
keys = data.keys() # to freeze the order of the keys

comps = {'%s:%s'%(one,two): angles(data[one]['v'],data[two]['v']) for one in keys for two in keys}
cPickle.dump(comps,open('figure-1-deriv-data-cricular.pkl','wb'))
pprint(comps)
