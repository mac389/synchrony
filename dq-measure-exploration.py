import numpy as np

#exploring my q measure

dq = lambda data: abs(np.diff(map(abs,map(np.linalg.det,np.rollaxis(data,2)))))

series = np.random.random_sample(size=(3,3,3))

print dq(series)


'''
	Expectations for measurements. 

	Distance measure:

	1. d(x,y) = d(y,x) Symmetric
	2. d(x,x) = 0 Consistent 
	3. d(x,y) > 0 Positive

'''

#Baseline for distance measurements is the distance between random matrices.
#INCREASING the correlation between two variables will DECREASE the dq