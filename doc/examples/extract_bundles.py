"""
===========================
Automatic bundle extraction
===========================
"""
import numpy as np
from os.path import basename
import nibabel.trackvis as tv
from glob import glob
from dipy.viz import fvtk
from time import time
from copy import deepcopy
from dipy.segment.extractbundles import whole_brain_registration
from dipy.tracking.streamline import (transform_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import QuickBundles
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from itertools import chain


def read_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams], hdr


def write_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr_dict = {key: hdr[key] for key in hdr.dtype.names}
        #hdr2 = deepcopy(hdr)
        tv.write(fname, streams, hdr_mapping=hdr_dict, points_space='rasmm')
    else:
        tv.write(fname, streams, points_space='rasmm')


def show_bundles(static, moving,  linewidth=0.15, tubes=False,
                 opacity=1., fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(1900, 1200))
    if fname is not None:
        fvtk.record(ren, size=(1900, 1200), out_path=fname)


def janice_next_subject(dname_whole_streamlines, verbose=False):

    for wb_trk2 in glob(dname_whole_streamlines + '*.trk'):

        wb2, hdr = read_trk(wb_trk2)

        if verbose:
            print(wb_trk2)

        tag = basename(wb_trk2).split('_')[0]

        if verbose:
            print(tag)

        yield (wb2, tag)


# Read janice model streamlines

model_tag = 't0337'

initial_dir = '/home/eleftherios/Data/Hackethon_bdx/'

dname_model_bundles = initial_dir + 'bordeaux_tracts_and_stems/'

model_bundle_trk = dname_model_bundles + \
    model_tag + '/tracts/IFOF_R/' + model_tag + '_IFOF_R_GP.trk'

model_bundle, _ = read_trk(model_bundle_trk)

dname_whole_brain = initial_dir + \
    'bordeaux_whole_brain_DTI/whole_brain_trks_60sj/'

model_streamlines_trk = dname_whole_brain + \
    't0337_dti_mean02_fact-45_splined.trk'

model_streamlines, hdr = read_trk(model_streamlines_trk)


for (streamlines, tag) in janice_next_subject(dname_whole_brain):


    #1. Affine registration

    moved_streamlines, mat, centroids1, centroids2 = whole_brain_registration(model_streamlines,
                                                                              streamlines)

    print(tag)
    write_trk(tag + '_moved_streamlines.trk', moved_streamlines, hdr=hdr)
    print('done')

    #show_bundles(model_streamlines, streamlines[:1000], opacity=0.1)
    #show_bundles(model_streamlines, moved_streamlines[:1000], opacity=0.1)

    #show_bundles(centroids1, centroids2, opacity=1)
    #show_bundles(centroids1, transform_streamlines(centroids2, mat), opacity=1)

    # 2. Centroids of model bundle

    rmodel_bundle = set_number_of_points(model_bundle, 20)
    qb = QuickBundles(threshold=20)
    model_cluster_map = qb.cluster(rmodel_bundle)
    model_centroids = model_cluster_map.centroids

    # 3. Calculate centroids of streamlines

    rstreamlines = set_number_of_points(streamlines, 20)
    cluster_map = qb.cluster(rstreamlines)
    cluster_map.refdata = streamlines

    # 4. Find centroids which are close to the model_centroids

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            cluster_map.centroids)

    centroid_matrix[centroid_matrix > 20] = np.inf

    mins = np.min(centroid_matrix, axis=0)
    close_clusters = [cluster_map[i] for i in np.where(mins != np.inf)[0]]

    close_centroids = [cluster.centroid for cluster in close_clusters]

    close_streamlines = list(chain(*close_clusters))

    show_bundles(model_bundle, close_streamlines)

    # 5. Use SLR to match the model_bundle with the close_streamlines




    break
