import numpy as np
from glob import glob
from os.path import expanduser, join, dirname, realpath, basename
from nibabel import trackvis as tv
import nibabel as nib
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from dipy.bundle.descriptors import (length_distribution,
                                     qb_centroids)
from dipy.align.streamwarp import (LinearRegistration,
                                   transform_streamlines,
                                   matrix44,
                                   mdf_optimization_sum,
                                   mdf_optimization_min,
                                   center_streamlines)

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


colors = np.array([fvtk.colors.cyan_white,
                   fvtk.colors.violet_red,
                   fvtk.colors.turquoise_pale,
                   fvtk.colors.steel_blue_light,
                   fvtk.colors.honeydew])
print(colors)


bundles, fnames = bring_bundle_from_all_subjs('cb.right')
bundles = clean_bundles(bundles, 30)
cbundles = qb_bundles(bundles, thr=10)




show_all_bundles(cbundles, colors=colors)
