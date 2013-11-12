import numpy as np
from glob import glob
from os.path import expanduser, join, dirname, realpath, basename
from nibabel import trackvis as tv
import nibabel as nib
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from dipy.bundle.descriptors import (length_distribution,
                                     qb_centroids,
                                     flip_to_source,
                                     avg_streamline,
                                     winding_angles,
                                     dragons_hits)
from dipy.align.streamwarp import (LinearRegistration,
                                   transform_streamlines,
                                   matrix44,
                                   mdf_optimization_sum,
                                   mdf_optimization_min,
                                   center_streamlines)
from dipy.tracking.metrics import downsample


home = expanduser("~")
hcpdname = join(home, 'Data', 'HCP', 'Q1')
subjs = ['100307', '111312', '194140', '865363', '889579']


def bring_bundle_from_all_subjs(name):
    bfname = 'bundles.trk*' + name + '*.trk'
    print(bfname)
    fnames = []
    bundles = []
    for subj in subjs:
        fname = glob(join(hcpdname, subj, bfname))[0]
        print(fname)
        fnames.append(fname)
        bundle = read_bundles(fname)
        bundles.append(bundle)
    return bundles, fnames


def read_bundles(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    streamlines = [s[0] for s in streams]
    return streamlines


def show_all_bundles_fnames(fnames, colors=None):

    ren = fvtk.ren()
    for (i, fname) in enumerate(fnames):
        streamlines = read_bundles(fname)
        if colors is None:
            color = np.random.rand(3)
        else:
            color = colors[i]
        fvtk.add(ren, fvtk.line(streamlines, color))
    fvtk.show(ren)


def show_all_bundles(bundles, colors=None):

    ren = fvtk.ren()
    for (i, bundle) in enumerate(bundles):
        if colors is None:
            color = np.random.rand(3)
        else:
            color = colors[i]
        fvtk.add(ren, fvtk.line(bundle, color))
    fvtk.show(ren)


def clean_bundles(bundles, length_thr):
    new_bundles = []
    for bundle in bundles:
        lengths = length_distribution(bundle)
        new_bundle = [s for (i, s) in enumerate(bundle) if lengths[i] > length_thr]
        new_bundles.append(new_bundle)
    return new_bundles


def qb_bundles(bundles, thr, pts=18):
    new_bundle = []
    for bundle in bundles:
        centroids = qb_centroids(bundle, thr, pts)
        new_bundle.append(centroids)
    return new_bundle


def linear_reg(static, moving):
    static_center, shift = center_streamlines(static)
    lin = LinearRegistration(mdf_optimization_min, 'rigid')
    moving_center = lin.transform(static_center, moving)
    return static_center, moving_center, shift


def viz_vol(vol):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.volume(vol))
    fvtk.show(ren)


def all_descriptors(bundles):
    descr = {}
    descr['lengths'] = []
    descr['avg_streamline'] = []
    descr['winding_angle'] = []
    descr['dragons_hits'] = []
    for bundle in bundles:
        bundle = flip_to_source(bundle, bundle[0][0])
        descr['lengths'].append(length_distribution(bundle))
        avg = avg_streamline(bundle)
        descr['avg_streamline'].append(avg)
        descr['winding_angle'].append(winding_angles(bundle))
        descr['dragons_hits'].append(dragons_hits(bundle, avg))
    return descr


def measure_overlap(static_center, moving_center, show=True, vol_size=(256, 256, 256)):
    static_center = [downsample(s, 100) for s in static_center]
    moving_center = [downsample(s, 100) for s in moving_center]
    vol = np.zeros(vol_size)

    ci, cj, ck = vol_size[0]/2 , vol_size[1]/2 , vol_size[2]/2

    spts = np.concatenate(static_center, axis=0)
    spts = np.round(spts).astype(np.int) + np.array([ci, cj, ck])

    mpts = np.concatenate(moving_center, axis=0)
    mpts = np.round(mpts).astype(np.int) + np.array([ci, cj, ck])

    for index in spts:
        i, j, k = index
        vol[i, j, k] = 1

    vol2 = np.zeros(vol_size)
    for index in mpts:
        i, j, k = index
        vol2[i, j, k] = 1

    vol_and = np.logical_and(vol,vol2)
    overlap = np.sum(vol_and) / float(np.sum(vol2))
    if show: viz_vol(vol_and)
    return 100 * overlap


colors = np.array([fvtk.colors.cyan_white,
                   fvtk.colors.violet_red,
                   fvtk.colors.turquoise_pale,
                   fvtk.colors.steel_blue_light,
                   fvtk.colors.honeydew])
print(colors)

bundles, fnames = bring_bundle_from_all_subjs('cb.right')
bundles = clean_bundles(bundles, 30)

#descr = all_descriptors(bundles)

cbundles = qb_bundles(bundles, thr=10)

show_all_bundles(cbundles, colors=colors)

for i in range(1, 5):

    static = cbundles[0]
    moving = cbundles[i]
    static_center, moving_center, shift = linear_reg(static, moving)
    overlap = measure_overlap(static_center, moving_center)
    print('Overlap %.2f' % overlap)
    rbundles = [static_center, moving_center]
    show_all_bundles(rbundles, colors=colors)
