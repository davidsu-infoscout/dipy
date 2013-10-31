from os.path import expanduser, join
from nibabel import trackvis as tv
from glob import glob
from dipy.bundle.descriptor import avg_streamline, flip_to_source, qb_centroids
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors


def strip_fname(fname, tag):
    return fname.split(tag + '_')[1].split('.trk')[0]


def bundles_names(fnames, tag):
    names_ = []
    for fname in fnames:
        names_.append(strip_fname(fname, tag))
    return names_


def read_streamlines(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [s[0] for s in streams]


def show_streamlines(streamlines, avg_streamlines=None, mpoints=None, cpoints=None):

    ren = fvtk.ren()
    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))

    if avg_streamlines is not None:
        fvtk.add(ren, fvtk.line(avg_streamlines, fvtk.colors.yellow, linewidth=4))

    if mpoints is not None:
        fvtk.add(ren, fvtk.point(mpoints, fvtk.colors.yellow))

    if cpoints is not None:
        fvtk.add(ren, fvtk.point(cpoints, fvtk.colors.white))

    fvtk.show(ren)
    return ren


def bundle_descriptors(bundle):
    avg = avg_streamline(bundle)
    qbc = qb_centroids(bundle, thr=100, pts=18)
    
    return (avg, qbc)


home = expanduser('~')
dname = join(home, 'Data', 'bundles_new')

fbun = glob(join(dname, '*_cc_*.trk'))
nbun = bundles_names(fbun, 'bundles_new.trk')

avgs = []
qbcs = []
bundles = []
for fname in fbun:#[:1]:
    print(fname)
    bundle = read_streamlines(fname)
    bundle = flip_to_source(bundle)
    bundles += bundle
    avgs.append(bundle_descriptors(bundle)[0])
    qbcs.append(bundle_descriptors(bundle)[1])

#show_streamlines(bundles, avgs)
show_streamlines(bundles,qbcs)

