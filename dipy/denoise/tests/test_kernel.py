from dipy.denoise.kernel import EnhancementKernel

import numpy as np

D33=1.0
D44=0.04
t=1
k = EnhancementKernel(D33,D44,t)

x = np.array([0,0,0],dtype=np.float64)
y = np.array([0,0,0],dtype=np.float64)
r = np.array([0,0,.8],dtype=np.float64)
v = np.array([0,0,1],dtype=np.float64)
print k.k2(x,y,r,v)
# should print: 0.00932114

x = np.array([1,0,0],dtype=np.float64)
y = np.array([0,0,0],dtype=np.float64)
r = np.array([0,0,1],dtype=np.float64)
v = np.array([0,0,1],dtype=np.float64)
print k.k2(x,y,r,v)
# should print: 0.0355297

