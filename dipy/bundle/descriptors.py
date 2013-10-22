import numpy as np
from nibabel import trackvis as tv
from dipy.segment.quickbundles import QuickBundles
from dipy.data import get_data
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from dipy.tracking.utils import length
from dipy.tracking.metrics import (downsample, 
                                   winding,
                                   midpoint,
                                   center_of_mass)
from dipy.tracking.distances import cut_plane


def fornix_streamlines():
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    return streamlines


def show_streamlines(streamlines, mpoints=None, cpoints=None):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))

    if mpoints is not None:
        fvtk.add(ren, fvtk.point(mpoints, fvtk.colors.yellow))

    if cpoints is not None:
        fvtk.add(ren, fvtk.point(cpoints, fvtk.colors.white))

    fvtk.show(ren)
    return ren


def length_distribution(streamlines):
    lengths = length(streamlines)
    lengths = list(lengths)
    return lengths


def winding_angles(streamlines):
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    return [winding(s) for s in streamlines]


def avg_streamline(streamlines, pts=18):
    if pts is None:
        return np.mean(streamlines, axis=0)
    else:
        streamlines = [downsample(s, pts) for s in streamlines]
        return np.mean(streamlines, axis=0)


def qb_centroids(streamlines, thr=10, pts=18):
    qb = QuickBundles(streamlines, thr, pts)
    return qb.centroids


def midpoints(streamlines):
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    return np.array([midpoint(s) for s in streamlines])


def centers_of_mass(streamlines):
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    return np.array([center_of_mass(s) for s in streamlines])


def dragons_hits(streamlines, avg_streamline):
    hits = cut_plane(streamlines, avg_streamline)
    xyz = [h[:, :3] for h in hits]
    angles = [h[:, 3:] for h in hits]
    return xyz, angles


def dragons_hits_asarray(streamlines, avg_streamline):
    xyz, angles = dragons_hits(streamlines, avg_streamline)
    return np.concatenate(xyz), np.concatenate(angles)


streamlines = fornix_streamlines()

lengths = length(streamlines)

avg = avg_streamline(streamlines)

centroids = qb_centroids(streamlines, 10)

winds = winding_angles(centroids)

mpoints = midpoints(centroids)

cpoints = centers_of_mass(centroids)

hpoints, hangles = dragons_hits_asarray(centroids, avg)

ren = show_streamlines(centroids, mpoints)

fvtk.add(ren, fvtk.point(hpoints, fvtk.colors.red))

fvtk.show(ren)

