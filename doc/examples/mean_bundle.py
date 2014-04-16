"""
====================
Tracking Quick Start
====================

This example shows how to perform fiber tracking using Dipy.

We will use Constrained Spherical Deconvolution (CSD) [Tournier07]_ for local
reconstructions and then generate deterministic streamlines using the fiber
directions (peaks) from CSD and fractional anisotropic (FA) as a
stopping criterion.

First, let's load the necessary modules.
"""

from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.peaks import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

"""
Load one of the available datasets with 150 gradients on the sphere and 10 b0s
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

data = img.get_data()

maskdata, mask = median_otsu(data, 3, 1, False,
                             vol_idx=range(10, 50), dilate=2)

response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

csd_model = ConstrainedSphericalDeconvModel(gtab, response)


sphere = get_sphere('symmetric724')

csd_peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             mask=mask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)


tensor_model = TensorModel(gtab, fit_method='WLS')
tensor_fit = tensor_model.fit(data, mask)

FA = fractional_anisotropy(tensor_fit.evals)


stopping_values = np.zeros(csd_peaks.peak_values.shape)
stopping_values[:] = FA[..., None]

ren = fvtk.ren()

streamline_generator = EuDX(stopping_values,#[..., :5],
                            csd_peaks.peak_indices,#[..., :5],
                            seeds=10**4,
                            odf_vertices=sphere.vertices,
                            a_low=0.1)

streamlines = [streamline for streamline in streamline_generator]

fvtk.clear(ren)

fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))

fvtk.show(ren, size=(900, 900))

from dipy.segment.quickbundles import QuickBundles

qb = QuickBundles(streamlines, dist_thr=10., pts=18)

centroids = qb.centroids

from dipy.tracking.utils import streamline_mapping

# mapping = streamline_mapping(centroids, (1, 1, 1))

mapping = streamline_mapping([centroids[0]], (1, 1, 1))

indices = np.array(mapping.keys())

FA_values = []
for index in indices:
    FA_values.append(FA[tuple(index)])

print(np.mean(FA_values))


"""
.. include:: ../links_names.inc
"""

