import numpy as np

from scipy.stats import pearsonr

r = 0.6 
x1 = np.array([0,0,0,1,1,1,0,0,0,1]).astype(float)
x1 -= x1.mean()

x2 = np.random.random_sample(size=x1.shape)


x3 = x2-x2.dot(x1) #Orthogonal component is the vector rejection


x1 /= np.linalg.norm(x1)
x3 /= np.linalg.norm(x2)

x3 = x3 + 1/np.tan(r)*x1

print np.corrcoef(x1,x3)