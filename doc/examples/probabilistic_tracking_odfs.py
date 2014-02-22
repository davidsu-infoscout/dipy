"""

====================================
Probabilistic Tracking on ODF fields
====================================

In this example we perform probabilistic fiber tracking on fields of ODF peaks.

This example requires importing example `reconst_csa.py`.

"""

import numpy as np
from reconst_csa import *
from dipy.reconst.interpolate import NearestNeighborInterpolator

from dipy.tracking.markov import (BoundaryStepper,
                                  FixedSizeStepper,
                                  ProbabilisticOdfWeightedTracker)

from dipy.tracking.utils import seeds_from_mask


stepper = FixedSizeStepper(1)

"""
Read the voxel size from the image header:
"""

zooms = img.get_header().get_zooms()[:3]


"""
Randomly select some seed points from the mask:
"""

seeds = seeds_from_mask(mask, [1, 1, 1], zooms)
seeds = seeds[:2000]


from dipy.reconst.peaks import peaks_from_model

sphere = get_sphere('symmetric724')

csa_peaks = peaks_from_model(model=csamodel,
                             data=maskdata,
                             sphere=sphere,
                             mask=mask,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)



class FakeModel():
    def __init__(self, B_matrix):
        self.B = B_matrix

    def fit(self, shm_coeff):
        return FakeFit(self.B, shm_coeff)

class FakeFit():
    def __init__(self, B, shm_coeff):
        self.B = B
        self.shm_coeff = shm_coeff

    def odf(self, sphere=None):

        return np.dot(self.B.T, self.shm_coeff)


fmodel = FakeModel(csa_peaks.B)


#interpolator = NearestNeighborInterpolator(maskdata, zooms)
interpolator = NearestNeighborInterpolator(csa_peaks.shm_coeff, zooms)

pwt = ProbabilisticOdfWeightedTracker(fmodel, interpolator, mask,
                                      stepper, 20, seeds, sphere)
csa_streamlines = list(pwt)

"""
Now that we have our streamlines in memory we can save the results to disk.
For this purpose we can use the TrackVis format (``*.trk``). First, we need to
create a header.
"""

import nibabel as nib

hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = (2., 2., 2.)
hdr['voxel_order'] = 'LAS'
hdr['dim'] = csapeaks.gfa.shape[:3]

"""
Save the streamlines.
"""

csa_streamlines_trk = ((sl, None, None) for sl in csa_streamlines)

csa_sl_fname = 'csa_prob_streamline.trk'

nib.trackvis.write(csa_sl_fname, csa_streamlines_trk, hdr)

"""
Visualize the streamlines with fvtk (python vtk is required).
"""

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

r = fvtk.ren()

fvtk.add(r, fvtk.line(csa_streamlines, line_colors(csa_streamlines)))

print('Saving illustration as csa_prob_tracks.png')

fvtk.record(r, n_frames=1, out_path='csa_prob_tracks.png', size=(600, 600))

"""
.. figure:: csa_prob_tracks.png
   :align: center

   **Probabilistic streamlines applied on an ODF field modulated by GFA**.
"""
