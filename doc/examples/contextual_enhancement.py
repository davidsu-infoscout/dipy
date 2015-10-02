"""
==============================================
Contextual enhancement
==============================================

Add description here

"""

import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.sims.voxel import add_noise

# read data
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

# add rician noise
data = add_noise(data, 5, 500 , noise_type='rician')

from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

# Estimate response function
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

# For illustration purposes we will fit only a small portion of the data.
data_small = data[20:50, 55:85, 38:39]
csd_fit = csd_model.fit(data_small)
csd_shm = csd_fit.shm_coeff

"""
Create lookup table and perform convolution
"""

from dipy.denoise.kernel import EnhancementKernel
from dipy.denoise.convolution5d import convolve_5d

# Create lookup table
D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t, force_recompute=True)

# Perform convolution
csd_enh = convolve_5d(csd_shm, k)

"""
Visualize the raw and enhanced ODFs
"""

from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import get_sphere

# Convert raw and enhanced data to discrete form
sphere = get_sphere('symmetric724')
csd_odf = sh_to_sf(csd_shm, sphere, sh_order=8)
csd_enh_odf = sh_to_sf(csd_enh, sphere, sh_order=8)

# Visualize the results with VTK
ren = fvtk.ren()

fodf_spheres = fvtk.sphere_funcs(csd_odf, sphere, scale=2, norm=False, radial_scale=True)
fvtk.add(ren, fodf_spheres)

fodf_spheres_enh = fvtk.sphere_funcs(csd_enh_odf*0.2, sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_enh.SetPosition(75, 0, 0)

fvtk.add(ren, fodf_spheres_enh)
fvtk.show(ren, size=(600, 600))
fvtk.record(ren, out_path='enhancements.png', size=(2048, 2048), magnification=2)