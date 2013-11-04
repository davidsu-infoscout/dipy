import numpy as np
from numpy.testing import (run_module_suite, assert_equal)
from dipy.align.streamwarp import (StreamWarp, 
                                   transform_streamlines, 
                                   matrix44,
                                   mdf_optimization)
from dipy.tracking.metrics import downsample


def simulated_bundle(no_streamlines=10, waves=False, no_pts=12):

    t = np.linspace(-10, 10, 200)
    
    # parallel waves
    bundle = []
    for i in np.linspace(-5, 5, no_streamlines):        
        if waves:
            pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        else:
             pts = np.vstack((np.zeros(t.shape), t, i * np.ones(t.shape))).T           
        pts = downsample(pts, no_pts)

        bundle.append(pts)

    return bundle


def viz(bundle, bundle2):
    from dipy.viz import fvtk
    
    ren = fvtk.ren()    
    fvtk.add(ren, fvtk.line(bundle, fvtk.colors.red))
    fvtk.add(ren, fvtk.line(bundle2, fvtk.colors.cyan))
    fvtk.show(ren)


def test_simulated_bundles():

    bundle = simulated_bundle()
    mat = matrix44([0, 0, 0, 45, 0, 0])
    bundle2 = transform_streamlines(bundle, mat)

    viz(bundle, bundle2)

    sw = StreamWarp(bundle, bundle2, mdf_optimization)
    new_bundle2 = sw.warped()
    print(sw.xopt)

    viz(bundle, new_bundle2)

    1/0

test_simulated_bundles()
