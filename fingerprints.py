import numpy as np
import math
from scipy import optimize, stats
import sys, os, time
from neural_data import data_agg

#normalizing parameter
def norm(n):
    return math.sqrt(math.sqrt(2*math.pi)*math.factorial(n))
#Hermite Polynomials
def h0(x):
    return 1/norm(0)
def h1(x):
    return x/norm(1)
def h2(x):
    return (x**2 - 1)/norm(2)
def h3(x):
    return (x**3 - 3*(x))/norm(3)
def h4(x):
    return (x**4 - 6*(x**2) + 3)/norm(4)

#Basis set
basis_set = [h0, h1, h2, h3, h4]

#Fingerprint
fingerprint = np.zeros([1,len(basis_set)])

#Voxelwise fingerprints

#Define subject
sys.argv = [sys.argv[0], sys.argv[1]]
sub = str(sys.argv[1])
subject = 'sub-' + sub
print('sub: ', sub)

#load subject data
ffa, gm = data_agg(subject)

#fingerprint data structure
fingerprint = np.zeros([gm.shape[1],len(basis_set)])

#for each GM voxel
voxels = gm.shape[1]

for vox in range(voxels):
    vox_data = gm[:,vox]
    for i, coef in enumerate(basis_set):
        basis_func = coef
        def pred(c):
            return sum((vox_data - c * basis_func(ffa))**2)
        ci = optimize.minimize(pred, x0=0).x
        fingerprint[vox,i] = ci
np.save('/gsfs0/data/poskanzc/FingerPrint_Project/FingerPrints/avg_FingerPrints/'+sub+'_zscored_fingerprint', fingerprint) 


