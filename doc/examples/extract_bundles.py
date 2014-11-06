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

from dipy.tracking.streamline import (length,
                                      transform_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import QuickBundles
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from itertools import chain, izip
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.viz.axycolor import distinguishable_colormap
from os import mkdir
from os.path import isdir


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


def show_bundles(static, moving, linewidth=0.15, tubes=False,
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


def show_centroids(clusters, colormap=None, cam_pos=None,
                   cam_focal=None, cam_view=None,
                   magnification=1, fname=None, size=(900, 900)):

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    ren = fvtk.ren()
    ren.SetBackground(*bg)

    max_cz = np.max(map(len, clusters))
    for cluster, color in izip(clusters, colormap):
            fvtk.add(ren, fvtk.line(cluster.centroid,
                                    color, linewidth=len(cluster)*10./float(max_cz)))

    fvtk.show(ren, size=size)
    if fname is not None:
        fvtk.record(ren, cam_pos=cam_pos, cam_focal=cam_focal, cam_view=cam_view,
                    out_path=fname, path_numbering=False, n_frames=1, az_ang=10,
                    magnification=magnification, size=size, verbose=True)


def get_bounding_box(streamlines):
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


def show_clusters_grid_view(clusters, colormap=None, makelabel=None,
                            cam_pos=None, cam_focal=None, cam_view=None,
                            magnification=1, fname=None, size=(900, 900)):

    def grid_distribution(N):
        def middle_divisors(n):
            for i in range(int(n ** (0.5)), 2, -1):
                if n % i == 0:
                    return i, n // i

            return middle_divisors(n+1)  # If prime number take next one

        height, width = middle_divisors(N)
        X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), [0])
        return np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    bg = (1, 1, 1)

    #if colormap is None:
    #    colormap = distinguishable_colormap(bg=bg)

    positions = grid_distribution(len(clusters))

    box_min, box_max = get_bounding_box(chain(*clusters))

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    width, height, depth = box_max - box_min
    text_scale = [height*0.1] * 3
    for cluster, color, pos in izip(clusters, colormap, positions):
        offset = pos * (box_max - box_min)
        offset[0] += pos[0] * 4*text_scale[0]
        offset[1] += pos[1] * 4*text_scale[1]

        fvtk.add(ren, fvtk.line([s + offset for s in cluster],
                                [color]*len(cluster)))

        if makelabel is not None:
            label = makelabel(cluster)
            #text_scale = tuple([scale / 50.] * 3)
            text_pos = offset + np.array([0, height+4*text_scale[1], depth])/2.
            text_pos[0] -= len(label) / 2. * text_scale[0]

            fvtk.label(ren, text=label, pos=text_pos, scale=text_scale,
                       color=(0, 0, 0))

    fvtk.show(ren, size=size)


def remove_clusters_by_size(clusters, min_size=0):
    #sizes = np.array(map(len, clusters))
    #mean_size = sizes.mean()
    #std_size = sizes.std()

    by_size = lambda c: len(c) >= min_size
    #and len(c) >= mean_size - alpha * std_size

    # filter returns a list of clusters
    return filter(by_size, clusters)


def whole_brain_registration(streamlines1, streamlines2,
                             rm_small_clusters=50,
                             maxiter=100,
                             verbose=False):

    if verbose:
        print(len(streamlines1))
        print(len(streamlines2))

    def check_range(streamline, gt=50, lt=250):
        if (length(s) > gt) & (length(s) < lt):
            return True

    streamlines1 = [s for s in streamlines1 if check_range(s)]
    streamlines2 = [s for s in streamlines2 if check_range(s)]

    if verbose:
        print(len(streamlines1))
        print(len(streamlines2))

    rstreamlines1 = set_number_of_points(streamlines1, 20)
    qb1 = QuickBundles(threshold=15)
    cluster_map1 = qb1.cluster(rstreamlines1)
    clusters1 = remove_clusters_by_size(cluster_map1, rm_small_clusters)
    qb_centroids1 = [cluster.centroid for cluster in clusters1]

    rstreamlines2 = set_number_of_points(streamlines2, 20)
    qb2 = QuickBundles(threshold=15)
    cluster_map2 = qb2.cluster(rstreamlines2)
    clusters2 = remove_clusters_by_size(cluster_map2, rm_small_clusters)
    qb_centroids2 = [cluster.centroid for cluster in clusters2]

    slr = StreamlineLinearRegistration(x0='affine',
                                       options={'maxiter': maxiter})

    t = time()

    slm = slr.optimize(qb_centroids1, qb_centroids2)

    if verbose:
        print('QB1 %d' % len(qb_centroids1,))
        print('QB2 %d' % len(qb_centroids2,))

    duration = time() - t
    if verbose:
        print('SAR done in  %f seconds.' % (duration, ))

    print('SAR iterations: %d .' % (slm.iterations, ))

    moved_streamlines2 = slm.transform(streamlines2)

    return moved_streamlines2, slm.matrix, qb_centroids1, qb_centroids2


def janice_next_subject(dname_whole_streamlines, verbose=False):

    for wb_trk2 in glob(dname_whole_streamlines + '*.trk'):

        wb2, hdr = read_trk(wb_trk2)

        if verbose:
            print(wb_trk2)

        tag = basename(wb_trk2).split('_')[0]

        if verbose:
            print(tag)

        yield (wb2, tag)


def janice_manual(tag, dname_model_bundles, mat=None):
    trk = dname_model_bundles + \
        tag + '/tracts/IFOF_R/' + tag + '_IFOF_R_GP.trk'
    print('Reading ' + trk)
    streamlines, _ = read_trk(trk)

    if mat is not None:
        return transform_streamlines(streamlines, mat)
    return streamlines


def auto_extract(model_bundle, streamlines,
                 close_centroids_thr=20,
                 clean_thr=7.,
                 local_slr=True,
                 disp=False, verbose=True):

    if verbose:
        print('# Centroids of model bundle')

    t0 = time()

    rmodel_bundle = set_number_of_points(model_bundle, 20)
    qb = QuickBundles(threshold=20)
    model_cluster_map = qb.cluster(rmodel_bundle)
    model_centroids = model_cluster_map.centroids

    if verbose:
        print('Duration %f ' % (time() - t0, ))

    if verbose:
        print('# Calculate centroids of streamlines')

    t = time()

    rstreamlines = set_number_of_points(streamlines, 20)
    cluster_map = qb.cluster(rstreamlines)
    cluster_map.refdata = streamlines

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if verbose:
        print('# Find centroids which are close to the model_centroids')

    t = time()

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            cluster_map.centroids)

    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf

    mins = np.min(centroid_matrix, axis=0)
    close_clusters = [cluster_map[i] for i in np.where(mins != np.inf)[0]]

    #close_centroids = [cluster.centroid for cluster in close_clusters]

    close_streamlines = list(chain(*close_clusters))

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if disp:
        show_bundles(model_bundle, close_streamlines)

    if local_slr:

        if verbose:
            print('# Use SLR to match the model_bundle with the close_streamlines')

        t = time()

        x0 = np.array([0, 0, 0, 0, 0, 0, 1.])
        bounds = [(-30, 30), (-30, 30), (-30, 30),
                  (-45, 45), (-45, 45), (-45, 45), (0.5, 1.5)]

        slr = StreamlineLinearRegistration(x0=x0, bounds=bounds)

        static = select_random_set_of_streamlines(model_bundle, 400)
        moving = select_random_set_of_streamlines(close_streamlines, 600)

        static = set_number_of_points(static, 20)
        moving = set_number_of_points(moving, 20)

        slm = slr.optimize(static, moving)

        closer_streamlines = transform_streamlines(close_streamlines,
                                                   slm.matrix)

        if verbose:
            print('Duration %f ' % (time() - t, ))

        if disp:
            show_bundles(model_bundle, closer_streamlines)

        matrix = slm.matrix
    else:
        closer_streamlines = close_streamlines
        matrix = np.eye(4)

    if verbose:
        print('# Remove unrelated bundles and expand')

    t = time()

    rcloser_streamlines = set_number_of_points(closer_streamlines, 20)

    clean_matrix = bundles_distances_mdf(rmodel_bundle, rcloser_streamlines)

    clean_matrix[clean_matrix > clean_thr] = np.inf

    mins = np.min(clean_matrix, axis=0)
    close_clusters_clean = [closer_streamlines[i]
                            for i in np.where(mins != np.inf)[0]]

    if verbose:
        print('Duration %f ' % (time() - t, ))

    msg = 'Total duration of automatic extraction %0.4f seconds.'
    print(msg % (time() - t0, ))
    if disp:
        show_bundles(model_bundle, close_clusters_clean)

    return close_clusters_clean, matrix


if __name__ == '__main__':

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

    results_dir = initial_dir + 'results_' + model_tag + '/'

    if not isdir(results_dir):
        mkdir(results_dir)

    print(model_tag)

    list_of_all = []

    i = 0

    for (streamlines, tag) in janice_next_subject(dname_whole_brain):

        print(tag)
        print('# Affine registration')
        t = time()

        ret = whole_brain_registration(model_streamlines, streamlines,
                                       maxiter=150)
        moved_streamlines, mat, centroids1, centroids2 = ret

        print('Duration %f ' % (time() - t, ))

        #write_trk(tag + '_moved_streamlines.trk', moved_streamlines, hdr=hdr)

        #show_bundles(centroids1, centroids2)
        #show_bundles(centroids1, transform_streamlines(centroids2, mat))

        close_clusters_clean, mat2 = auto_extract(model_bundle, streamlines,
                                                  close_centroids_thr=20,
                                                  clean_thr=10.,
                                                  local_slr=False,
                                                  disp=False, verbose=True)

        result_trk = results_dir + tag + '_close_clusters_clean.trk'

        print('Writing ' + result_trk)

        write_trk(result_trk, close_clusters_clean, hdr=hdr)

        manual = janice_manual(tag, dname_model_bundles,
                               mat=None)

        slr = StreamlineLinearRegistration(x0='rigid')
        slm = slr.optimize(set_number_of_points(close_clusters_clean, 20),
                           set_number_of_points(manual, 20))

        manual_in_model = slm.transform(manual)

        show_bundles(manual, close_clusters_clean)

        print

        list_of_all.append(close_clusters_clean)

        i += 1
        if i == 2:
            break

    list_of_all.append(model_bundle)

    colormap = np.random.rand(len(list_of_all), 3)
    colormap[-1] = np.array([1., 0, 0])

    show_clusters_grid_view(list_of_all, colormap)
