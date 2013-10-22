
import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk
from dipy.reconst.dti import TensorModel, fractional_anisotropy


def record_slice(fname, data, k, show=False):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.slicer(data, plane_k=[k]))
    if show: fvtk.show(ren)
    fvtk.record(ren, out_path=fname, size=(600, 600))


home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
fname = join(dname, 'dwi_nlm.nii.gz')
fencoding = join(dname, 'encoding.b')
subjid = 'MPI'

print('>>> Loading Raw data, b-values and masking background...')

img = nib.load(fname)
data = img.get_data()
affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]

bmat = np.loadtxt(fencoding)

from dipy.core.gradients import gradient_table

gtab = gradient_table(bmat[:, -1], bmat[:, :-1])

b0_index = np.where(gtab.b0s_mask==True)[0][0]

from dipy.segment.mask import median_otsu

data, mask = median_otsu(data, 4, 4)

print(data.shape)
print(affine)
print(nib.aff2axcodes(affine))

fname_slice = 'dwi_nlm_slice.png'
record_slice(fname_slice, data[..., b0_index], k=data.shape[2]/2)

print('>>> Resample data to 1x1x1 mm^3...')

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine,
                          zooms=zooms,
                          new_zooms=(1., 1., 1.))

del data, affine, zooms

print(data2.shape)
print(affine2)
print(nib.aff2axcodes(affine2))

print('>>> Save resampled data, masks and S0...')

fname2 = join(dname, 'dwi_nlm_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(data2, affine2), fname2)

mask2 = data2[..., b0_index] > 0

fname2_mask = join(dname, 'dwi_nlm_mask_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(mask2.astype(np.uint8), affine2), fname2_mask)

fname2_S0 = join(dname, 'dwi_nlm_S0_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(data2[..., b0_index], affine2), fname2_S0)

fname2_slice = 'dwi_nlm_slice_1x1x1.png'
record_slice(fname2_slice, data2[..., b0_index], k=data2.shape[2]/2)

print('>>> Calculate FA...')

ten = TensorModel(gtab)
tenfit = ten.fit(data2, mask2)
fname2_fa = join(dname, 'dwi_nlm_fa_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(tenfit.fa, affine2), fname2_fa)

fname_slice_fa = 'dwi_nlm_slice_FA_1x1x1.png'
record_slice(fname_slice_fa, tenfit.fa, k=data2.shape[2]/2)

