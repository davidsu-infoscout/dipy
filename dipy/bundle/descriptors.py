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
    """ Length distribution

    Parameters
    ----------
    streamlines : list
        List of ndarrays (N, 3)

    Returns
    -------
    lengths : ndarray
    """
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    lengths = length(streamlines)
    lengths = list(lengths)
    return np.array(lengths)


def winding_angles(streamlines):
    """ Total turning angle projected.

    Project space curve to best fitting plane. Calculate the cumulative signed
    angle between each line segment and the previous one.

    Parameters
    ----------
    streamlines : list
        List of arrays representing x,y,z of N points in a track.

    Returns
    -------
    winds : ndarray
        Total turning angles in degrees for every streamline.
    """
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    return np.array([winding(s) for s in streamlines])


def avg_streamline(streamlines, pts=18):
    """ Average streamline in set of streamlines
    
    Parameters
    ----------
    streamlines : list
        List of arrays representing x,y,z of N points in a streamline.

    Returns
    -------
    avg : ndarray        
    """
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    if pts is None:
        return np.mean(streamlines, axis=0)
    else:
        streamlines = [downsample(s, pts) for s in streamlines]
        return np.mean(streamlines, axis=0)


def qb_centroids(streamlines, thr=10, pts=18):
    """ QuickBundles centroids

    See also
    --------
    dipy.segment.quickbundles.QuickBundles
    """
    qb = QuickBundles(streamlines, thr, pts)
    return qb.centroids


def midpoints(streamlines):
    """ Midpoint of each streamline
    
    Parameters
    ----------
    streamlines : list
        List of ndarrays (N, 3)

    Returns
    -------
    mpoints : ndarray
        Array shape (M, 3) where M the number of streamlines
    """
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    return np.array([midpoint(s) for s in streamlines])


def centers_of_mass(streamlines):
    """ Center of mass of each streamline
    
    Parameters
    ----------
    streamlines : list
        List of ndarrays (N, 3)

    Returns
    -------
    cpoints : ndarray
        Array shape (M, 3) where M the number of streamlines
    """
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    return np.array([center_of_mass(s) for s in streamlines])


def _dragons_hits(streamlines, avg_streamline):
    if not isinstance(streamlines, list):
        streamlines = [streamlines]
    hits = cut_plane(streamlines, avg_streamline)
    xyz = [h[:, :3] for h in hits]
    angles = [h[:, 3:] for h in hits]
    return xyz, angles


def dragons_hits(streamlines, ref_streamline):
    """ Intersections of streamlines by planes defined
    along a reference streamline.
    
    Parameters
    ----------
    streamlines : list
        List of ndarrays (N, 3)

    ref_streamline : ndarray
        Array of shape (K, 3)

    Returns
    -------
    hpoints : ndarray
        Array shape (M, 3) where M the number of hit points of the
        streamlines to the planes
    angles : ndarray
        Array shape (M, 2) with the deviation polar angles from each
        streamline to the reference streamline on the planes.
    """
    xyz, angles = _dragons_hits(streamlines, ref_streamline)
    return np.concatenate(xyz), np.concatenate(angles)




