from os.path import expanduser, join
from nibabel import trackvis as tv
import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk
from dipy.reconst.dti import TensorModel

home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
subjid = 'MPI_T1wS0'

from glob import glob
fnames = glob(join(dname, "*.trk" ))

#fname = join(dname, 'test_csd.trk_cst.left.trk')
#fname = join(dname, 'test_csd.trk_cst.right.trk')
#fname = join(dname, 'test_csd.trk_af.right.trk')
#fname = join(dname, 'test_csd.trk_ilf.right.trk')
#fname = join(dname, 'test_csd.trk_ilf.left.trk')
#fname = join(dname, 'csd_streamlines.trk')

streams, hdr = tv.read(fname, points_space='rasmm')
streamlines = [s[0] for s in streams]
streamlines = streamlines[:3000]

from dipy.viz import fvtk

ren = fvtk.ren()

from dipy.viz.colormap import line_colors

fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))

fvtk.show(ren)
