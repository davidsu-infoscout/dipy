import numpy as np
import os.path as op
import nibabel as nib
import dipy.core.optimize as opt
if not op.exists('lr-superiorfrontal.trk'):
    from streamline_tools import *
else:
    # We'll need to know where the corpus callosum is from these variables:
    from dipy.data import (read_stanford_labels,
                           fetch_stanford_t1,
                           read_stanford_t1)
    hardi_img, gtab, labels_img = read_stanford_labels()
    labels = labels_img.get_data()
    cc_slice = labels == 2
    fetch_stanford_t1()
    t1 = read_stanford_t1()
    t1_data = t1.get_data()
    data = hardi_img.get_data()
    
# Read the candidates from file in voxel space:
candidate_sl = [s[0] for s in nib.trackvis.read('lr-superiorfrontal.trk',
                                                  points_space='voxel')[0]]

# select 100 fibers
candidate_sl = candidate_sl[1:3]

from dipy.viz.colormap import line_colors
from dipy.viz import fvtk
candidate_streamlines_actor = fvtk.streamtube(candidate_sl,
                                       line_colors(candidate_sl))
cc_ROI_actor = fvtk.contour(cc_slice, levels=[1], colors=[(1., 1., 0.)],
                            opacities=[1.])
vol_actor = fvtk.slicer(t1_data, voxsz=(1.0, 1.0, 1.0), plane_i=[40],
                        plane_j=None, plane_k=[35], outline=False)
# Add display objects to canvas
ren = fvtk.ren()
fvtk.add(ren, candidate_streamlines_actor)
fvtk.add(ren, cc_ROI_actor)
fvtk.add(ren, vol_actor)
fvtk.record(ren, n_frames=1, out_path='fibers_before.png',
            size=(800, 800))

# compute lookup table
from dipy.denoise.kernel import EnhancementKernel
D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t)

# apply fbc measures
from dipy.tracking.fbcmeasures import FBCMeasures
fbc = FBCMeasures(candidate_sl, k)

optimized_sl=np.array(fbc.get_points_rfbc_thresholded(.5))
print 'number of fibers'
print len(optimized_sl)

rfbc = fbc.get_rfbc()
meanrfbc = np.mean(rfbc)
print 'mean rfbc'
print meanrfbc



print candidate_sl
print optimized_sl

ren = fvtk.ren()
fvtk.add(ren, fvtk.streamtube(optimized_sl, line_colors(optimized_sl)))
fvtk.add(ren, cc_ROI_actor)
fvtk.add(ren, vol_actor)
fvtk.record(ren, n_frames=1, out_path='fibers_after.png',
            size=(800, 800))
