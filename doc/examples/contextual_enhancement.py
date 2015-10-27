"""
==============================================
Contextual enhancement
==============================================

A contextual enhancement method is 

We consider Brownian motion on the coupled space of positions and orientations.
The corresponding (Fokker-Planck) diffusion equations are given by

.. math::

    \frac{\partial}{\partial t} W(\vec{y},\vec{n},t) = ((D^{33}(\vec{n} \cdot 
        \nabla)^2 + D^{44} \Delta_{S^2})W)(\vec{y},\vec{n},t)
    W(\vec{y},\vec{n},0) = U(\vec{y},\vec{n})

where :math:`D^{33} \geq 0` and :math:`D^{44} \geq 0` are parameters for the spatial 
(with spatial propagation direction :math:`\vec{n}`) and angular diffusion, 
respectively, for an evolution over time :math:`t \geq 0` with 
position :math:`\vec{y}` \in \mathbb{R}^3` and orientation :math:`\vec{n} \in S^2`. 
See Duits [1] for details on the notations.


.. figure:: beta_histogram.png
   :align: center

   **LiFE streamline weights**


"""
import numpy as np
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.sims.voxel import add_noise

# read data
fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
response2, ratio2 = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model2 = ConstrainedSphericalDeconvModel(gtab, response2)
data2 = data[25:40, 65:80, 35:39]
csd_fit2 = csd_model2.fit(data2)
csd_shm2 = csd_fit2.shm_coeff

# add Rician noise
data = add_noise(data, 2, 500 , noise_type='rician')

from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

# Estimate response function
response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
csd_model = ConstrainedSphericalDeconvModel(gtab, response)

# For illustration purposes we will fit only a small portion of the data.
data = data[25:40, 65:80, 35:39]
#data_small = data[20:50, 55:85, 37:38]
csd_fit = csd_model.fit(data)
csd_shm = csd_fit.shm_coeff

"""
Create lookup table
"""

from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.denoise.convolution5d import convolve_5d

# Create lookup table
D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t)

"""
Visualize the kernel
"""

""" 
Perform convolution
"""

# Perform convolution
csd_enh = convolve_5d(csd_shm, k)

# Sharpen via the Spherical Deconvolution Transform
from dipy.reconst.csdeconv import odf_sh_to_sharp
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
csd_enh_sharp = odf_sh_to_sharp(csd_enh, sphere,  sh_order=8, lambda_=0.1)

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
csd_enh_sharp_odf = sh_to_sf(csd_enh_sharp, sphere, sh_order=8)
csd_org = sh_to_sf(csd_shm2, sphere, sh_order=8)

# Normalize the sharpened ODFs
csd_enh_sharp_odf = csd_enh_sharp_odf * np.amax(csd_enh_odf)/np.amax(csd_enh_sharp_odf)

""" 
Visualize the results with VTK
"""
ren = fvtk.ren()

# original ODF field
fodf_spheres_org = fvtk.sphere_funcs(csd_org[:,:,[3],:], sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_org.SetPosition(0, 35, 0)
fvtk.add(ren, fodf_spheres_org)

# ODF field with added noise
fodf_spheres = fvtk.sphere_funcs(csd_odf[:,:,[3],:], sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres.SetPosition(35, 35, 0)
fvtk.add(ren, fodf_spheres)

# Enhancement of noisy ODF field
fodf_spheres_enh = fvtk.sphere_funcs(csd_enh_odf[:,:,[3],:], sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_enh.SetPosition(0,0, 0)
fvtk.add(ren, fodf_spheres_enh)

# Additional sharpening
fodf_spheres_enh_sharp = fvtk.sphere_funcs(csd_enh_sharp_odf[:,:,[3],:], sphere, scale=2, norm=False, radial_scale=True)
fodf_spheres_enh_sharp.SetPosition(35, 0, 0)
fvtk.add(ren, fodf_spheres_enh_sharp)

#fvtk.show(ren, size=(600, 600))
fvtk.record(ren, out_path='enhancements.png', size=(600, 600))