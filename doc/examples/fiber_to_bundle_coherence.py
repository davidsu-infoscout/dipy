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


from dipy.denoise.kernel import EnhancementKernel
D33 = 1.0
D44 = 0.02
t = 1
k = EnhancementKernel(D33, D44, t)

from dipy.tracking.fbcmeasures import FBCMeasures
FBCMeasures(candidate_sl, k)