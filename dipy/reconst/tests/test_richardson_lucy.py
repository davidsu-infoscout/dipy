import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_data, get_sphere
from dipy.sims.voxel import multi_tensor, multi_tensor_odf, single_tensor, all_tensor_evecs
from dipy.viz import fvtk



SNR = 100
S0 = 1

fdwi, fbvals, fbvecs = get_data('small_64D')

bvals = np.load(fbvals)
bvecs = np.load(fbvecs)

gtab = gradient_table(bvals, bvecs)
mevals = np.array(([0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]))

angles = [(0, 0), (60, 0)]

S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                         fractions=[50, 50], snr=SNR)

#sphere = get_sphere('symmetric362')
sphere = get_sphere('symmetric724')
#sphere = sphere.subdivide(2)

odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

ren = fvtk.ren()

fvtk.add(ren, fvtk.sphere_funcs(odf_gt, sphere))

fvtk.show(ren)

# Dictionary creation

num_dir = 64
num_dir_fod = 724
response = (np.array([0.0017, 0.0002, 0.0002]), S0)

H = np.zeros((num_dir, num_dir_fod))


for i in range(num_dir_fod):

    evals = response[0]
    evecs = all_tensor_evecs(sphere.vertices[i])

    signal_tmp = single_tensor(gtab, S0, evals, evecs)

    H[:, i] = signal_tmp[1:]

print(H.shape)

H_t = H.T


H_t_s = np.dot(H_t, S[1:])

print(H_t_s.shape)

fod = np.ones(num_dir_fod)/float(num_dir)

for i in range():


