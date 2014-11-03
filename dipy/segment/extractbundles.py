import numpy as np

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import length
from dipy.segment.quickbundles import QuickBundles
from time import time


def whole_brain_registration(streamlines1, streamlines2,
                             rm_small_clusters=50):
    print(len(streamlines1))
    print(len(streamlines2))

    def check_range(streamline, gt=50, lt=250):
        if (length(s) > gt) & (length(s) < lt):
            return True

    streamlines1 = [s for s in streamlines1 if check_range(s)]
    streamlines2 = [s for s in streamlines2 if check_range(s)]

    print(len(streamlines1))
    print(len(streamlines2))

    qb1 = QuickBundles(streamlines1, 20, 20)
    qb1.remove_small_clusters(rm_small_clusters)
    qb_centroids1 = qb1.centroids

    qb2 = QuickBundles(streamlines2, 20, 20)
    qb2.remove_small_clusters(rm_small_clusters)
    qb_centroids2 = qb2.centroids

    slr = StreamlineLinearRegistration(x0='affine')

    t = time()

    slm = slr.optimize(qb_centroids1, qb_centroids2)

    print('QB1', len(qb_centroids1))
    print('QB2', len(qb_centroids2))

    duration = time() - t
    print('SAR done in  %f seconds.' % duration)

    moved_streamlines2 = slm.transform(streamlines2)

    return moved_streamlines2, slm.matrix


class ExtractBundles():
    """ Extract bundles from whole brain streamlines using an initial
    model of the streamlines """

    def extract(self, streamlines, model_streamlines, model_bundle):

        moved_model_streamlines, mat = whole_brain_registration(streamlines,
                                                                model_streamlines)

        return moved_model_streamlines
