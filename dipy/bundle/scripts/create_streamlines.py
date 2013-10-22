from os.path import expanduser, join
from nibabel import trackvis as tv
import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)


home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
subjid = 'MPI_T1wS0'

fname = join(dname, 'dwi_nlm_1x1x1.nii.gz')
img = nib.load(fname)
data = img.get_data()
affine = img.get_affine()
# this datasets is already masked so we can do
mask = data[..., 0] > 0

fencoding = join(dname, 'encoding.b')
bmat = np.loadtxt(fencoding)

from dipy.core.gradients import gradient_table

gtab = gradient_table(bmat[:, -1], bmat[:, :-1])

print('>>> Calculate FA for tracking threshold')

ten_model = TensorModel(gtab)
ten_fit = ten_model.fit(data, mask)
FA = ten_fit.fa

print('>>> Estimating response...')

response, _ = auto_response(gtab, data, w=20)

print('>>> Constrained Spherical Deconvolution...')

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

print('>>> Find peaks...')

from dipy.reconst.odf import peaks_from_model
peaks = peaks_from_model(model=csd_model,
                         data=data,
                         mask=mask,
                         sphere=sphere,
                         relative_peak_threshold=0.8,
                         min_separation_angle=45,
                         return_odf=False,
                         return_sh=True,
                         normalize_peaks=False,
                         sh_order=8,
                         npeaks=5,
                         parallel=True,
                         nbr_process=6)

print('>>> Start tracking...')

from dipy.tracking.eudx import EuDX

eu = EuDX(FA,
          peaks.peak_indices[..., 0],
          seeds=5*10**6,
          odf_vertices=sphere.vertices,
          a_low=0.1, voxel_origin='corner')

streamlines = [streamline for streamline in eu]

streamlines_trk = ((sl, None, None) for sl in streamlines)
sl_fname = join(dname, 'csd_streamlines.trk')

trk_header = nib.trackvis.empty_header()
nib.trackvis.aff_to_hdr(affine, trk_header, True, True)
trk_header['dim'] = peaks.gfa.shape
trk_header['n_count'] = len(streamlines)
nib.trackvis.write(sl_fname, streamlines_trk, trk_header, points_space='voxel')



