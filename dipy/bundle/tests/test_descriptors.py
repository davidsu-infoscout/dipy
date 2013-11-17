import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_almost_equal,
                           run_module_suite)
from dipy.data import get_data
from nibabel import trackvis as tv
from dipy.bundle.descriptors import (length_distribution,
                                     avg_streamline,
                                     qb_centroids,
                                     winding_angles,
                                     midpoints,
                                     centers_of_mass,
                                     dragons_hits)
from dipy.bundle.descriptors import show_streamlines
from dipy.viz import fvtk


def fornix_streamlines():
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    return streamlines


def test_descriptors_fornix():

    streamlines = fornix_streamlines()

    lengths = length_distribution(streamlines)

    assert_equal(lengths.max() < 100, True)
    assert_equal(lengths.min() > 10, True)

    avg = avg_streamline(streamlines)

    avg_length = length_distribution(avg)

    assert_equal(avg_length < lengths.max(), True)
    assert_equal(avg_length > lengths.min(), True)

    centroids = qb_centroids(streamlines, 10)

    assert_equal(len(centroids), 4)

    winds = winding_angles(centroids)

    assert_equal(np.mean(winds) < 300 and np.mean(winds) > 100, True)

    mpoints = midpoints(centroids)

    assert_equal(len(mpoints), 4)

    cpoints = centers_of_mass(centroids)

    assert_equal(len(cpoints), 4)

    hpoints, hangles = dragons_hits(centroids, avg)

    assert_equal(len(hpoints) > len(avg), True)

    ren = show_streamlines(centroids, mpoints)

    fvtk.add(ren, fvtk.point(hpoints, fvtk.colors.red))

    fvtk.add(ren, fvtk.line(avg, fvtk.colors.tomato))

    fvtk.add(ren, fvtk.point(avg, fvtk.colors.yellow))

    fvtk.show(ren)


def simulated_bundles(no_pts=200):
    t = np.linspace(-10, 10, no_pts)
    fibno = 150

    # helix
    bundle = []
    for i in np.linspace(3, 5, fibno):
        pts = 5 * np.vstack((np.cos(t), np.sin(t), t / i)).T  # helix diverging
        bundle.append(pts)

    # parallel waves
    bundle2 = []
    for i in np.linspace(-5, 5, fibno):
        pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        bundle2.append(pts)

    # spider - diverging in the ends
    bundle3 = []
    for i in np.linspace(-1, 1, fibno):
        pts = np.vstack((i ** 3 * t / 2., t, np.cos(t))).T
        bundle3.append(pts)

    # diverging in the middle
    bundle4 = []
    for i in np.linspace(-1, 1, fibno):
        pts = 2 * \
            np.vstack((0 * t + 2 * i * np.cos(.2 * t), np.cos(.4 * t), t)).T
        bundle4.append(pts)

    return [bundle, bundle2, bundle3, bundle4]


def test_descriptors_sim_bundles():

    sbundles = simulated_bundles()

    helix, parallel, spider, centerdiv = sbundles

    show_streamlines(sbundles[3])


def parametrize_arclength(streamline):
    n_vertex = len(streamline)
    disp = np.diff(streamline, axis=0)
    L2 = np.sqrt(np.sum(disp ** 2, axis=1))

    arc_length = np.sum(L2)
    cum_len = np.cumsum(L2) / float(arc_length)
    para = np.zeros(n_vertex)
    para[1:] = cum_len
    return para


def cosine_series(streamline, para, k=10):
    """
    http://brainimaging.waisman.wisc.edu/~chung/tracts/
    """
    n_vertex = len(para)

    para_even = np.hstack((-para[::-1][:-1], para))

    stream_even = np.vstack((streamline[::-1][:-1], streamline))

    Y = np.zeros((2 * n_vertex - 1, k + 1))

    para_even = np.tile(para_even, (k + 1, 1)).T

    pi_factors = np.tile(np.arange(k+1), (2 * n_vertex - 1, 1)) * np.pi

    Y = np.cos(para_even * pi_factors) * np.sqrt(2)

    beta = np.dot(np.dot(np.linalg.pinv(np.dot(Y.T, Y)), Y.T), stream_even)

    hat = np.dot(Y, beta)

    wfs = hat[n_vertex -1 :]

    return wfs



def test_cosine_series():

    t = np.arange(0, 10.1, 0.1)
    streamline = np.vstack((t*np.sin(t), t*np.cos(t), t)).T
    para = parametrize_arclength(streamline)
    wfs = cosine_series(streamline, para, 10)
    assert_array_almost_equal(wfs[0], np.array([0.5963, 0.6702, 0.6373]), 3)
    assert_array_almost_equal(wfs[-1], np.array([-4.8111, -8.7878, 9.8589]), 3)

    para2 = parametrize_arclength(streamline)
    wfs2 = cosine_series(streamline, para2, k=30)

    helix = simulated_bundles(200)[0]
    streamline = helix[0][:-2]
    para3 = parametrize_arclength(streamline)
    streamline = streamline[::-1]
    para4 = parametrize_arclength(streamline)
    streamline2 = streamline + np.array([[5., 0, 0.]])
    para5 = parametrize_arclength(streamline2)

    wfs3 = cosine_series(streamline, para4, k=20)
    wfs4 = cosine_series(streamline2, para5, k=20)

    from dipy.viz import fvtk

    ren = fvtk.ren()
    fvtk.add(ren, fvtk.line(streamline, fvtk.colors.red))
    fvtk.add(ren, fvtk.line(wfs3, fvtk.colors.green))
    fvtk.add(ren, fvtk.line(streamline2, fvtk.colors.blue))
    fvtk.add(ren, fvtk.line(wfs4, fvtk.colors.cyan))
    fvtk.show(ren)

if __name__ == '__main__':
    # run_module_suite()
    # test_descriptors_fornix()
    test_cosine_series()
