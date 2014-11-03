import numpy as np

from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import (length, transform_streamlines,
                                      set_number_of_points)
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking.distances import bundles_distances_mdf
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

    print('QB1 %d' % len(qb_centroids1,))
    print('QB2 %d' % len(qb_centroids2,))

    duration = time() - t
    print('SAR done in  %f seconds.' % (duration, ))

    moved_streamlines2 = slm.transform(streamlines2)

    return moved_streamlines2, slm.matrix, qb1, qb2


def find_bundle(model_bundle_moved, streamlines2, strategy='A', min_thr=10):

    model_bundle_moved = set_number_of_points(model_bundle_moved, 20)
    streamlines2 = set_number_of_points(streamlines2, 20)

    qbm = QuickBundles(model_bundle_moved, 4)

    #D = bundles_distances_mdf(model_bundle_moved, streamlines2)
    D = bundles_distances_mdf(qbm.centroids, streamlines2)

    D[D > min_thr] = np.inf

    print(D.shape)

    if strategy == 'A':
        indices = np.argmin(D, axis=1)

        return [streamlines2[index] for (i, index) in enumerate(indices)
                if D[i, index] != np.inf]

    if strategy == 'B':
        mins = np.min(D, axis=0)
        return [streamlines2[i] for i in np.where(mins != np.inf)[0]]


class ExtractBundles():
    """ Extract bundles from whole brain streamlines using an initial
    model of the streamlines """

    def __init__(self, strategy='A', min_thr=10, verbose=False):
        self.verbose = verbose
        self.qb_model_streamlines = None
        self.qb_streamlines = None
        self.moved_model_streamlines = None
        self.strategy = strategy
        self.min_thr = min_thr

    def extract(self, streamlines, model_streamlines, model_bundle):

        ret = whole_brain_registration(streamlines, model_streamlines)
        moved_model_streams, mat, qb_model_streamlines, qb_streamlines = ret

        self.moved_model_streamlines = moved_model_streams
        self.qb_model_streamlines = qb_model_streamlines
        self.qb_streamlines = qb_streamlines

        moved_model_bundle = transform_streamlines(model_bundle, mat)
        self.moved_model_bundle = moved_model_bundle

        self.extracted_bundle = find_bundle(moved_model_bundle,
                                            streamlines,
                                            self.strategy,
                                            min_thr=self.min_thr)

        return self.extracted_bundle
