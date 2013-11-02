from os.path import expanduser, join
from nibabel import trackvis as tv
import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.shore import ShoreModel
from subjects import *


print('>>> Loading data...')

fname = join(dname, 'dwi_1x1x1.nii.gz')
img = nib.load(fname)
data = img.get_data()
affine = img.get_affine()

fmask = join(dname, 'dwi_mask_1x1x1.nii.gz')
mask = nib.load(fmask).get_data()

ffa = join(dname, 'dwi_mask_1x1x1.nii.gz')
fa = nib.load(ffa).get_data()

fbvals = join(dname, 'bvals')
fbvecs = join(dname, 'bvecs')

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

gtab = gradient_table(bvals, bvecs, b0_threshold=10)

shore_model = ShoreModel(gtab, radial_order=6, zeta=700,
                         lambdaN=1e-8, lambdaL=1e-8)

from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

print('>>> Find peaks...')

from dipy.reconst.odf import peaks_from_model
peaks = peaks_from_model(model=shore_model,
                         data=data,
                         mask=mask,
                         sphere=sphere,
                         relative_peak_threshold=0.3,
                         min_separation_angle=25,
                         return_odf=False,
                         return_sh=True,
                         normalize_peaks=False,
                         sh_order=8,
                         npeaks=5,
                         parallel=False,
                         nbr_process=6)

print('>>> Save peak indices...')

fpeaks = join(dname, 'dwi_peaks_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(peaks.peak_indices, affine), fpeaks)

fsh = join(dname, 'dwi_sh_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(peaks.shm_coeff, affine), fsh)

print('>>> Start tracking...')

from dipy.tracking.eudx import EuDX

eu = EuDX(fa,
          peaks.peak_indices[..., 0],
          seeds=5*10**6,
          odf_vertices=sphere.vertices,
          a_low=0.1, voxel_origin='corner')

streamlines = [streamline for streamline in eu]

streamlines_trk = ((sl, None, None) for sl in streamlines)
sl_fname = join(dname, 'shore_streamlines.trk')

trk_header = nib.trackvis.empty_header()
nib.trackvis.aff_to_hdr(affine, trk_header, True, True)
trk_header['dim'] = peaks.gfa.shape
trk_header['n_count'] = len(streamlines)
nib.trackvis.write(sl_fname, streamlines_trk, trk_header, points_space='voxel')



