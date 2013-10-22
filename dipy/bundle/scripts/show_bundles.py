from os.path import expanduser, join
from nibabel import trackvis as tv
import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath, basename
from dipy.viz import fvtk
from dipy.reconst.dti import TensorModel
from dipy.viz.colormap import line_colors

def show_bundles(fname):

    ren = fvtk.ren()
    streams, hdr = tv.read(fname, points_space='rasmm')
    streamlines = [s[0] for s in streams]
    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))
    fvtk.show(ren, basename(fname).split('bundles.trk_')[1])


def show_all_bundles(fnames):

    ren = fvtk.ren()

    for fname in fnames:
        streams, hdr = tv.read(fname, points_space='rasmm')
        streamlines = [s[0] for s in streams]
        fvtk.add(ren, fvtk.line(streamlines, np.random.rand(3)))

    fvtk.show(ren)


home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef', 'bundles2')

from glob import glob
fnames = glob(join(dname, "*.trk" ))

# for fname in fnames:
#     show_bundles(fname)

show_all_bundles(fnames)