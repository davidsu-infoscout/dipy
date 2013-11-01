import numpy as np
import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.pickles import save_pickle


def record_slice(fname, data, k, show=False):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.slicer(data, plane_k=[k]))
    if show: fvtk.show(ren)
    fvtk.record(ren, out_path=fname, size=(600, 600))

id = 0
base_dirs = ['100307', '111312', '194140', '865363', '889579']
home = expanduser("~")

dname = join(home, 'Data', 'HCP', 'Q1', base_dirs[id])
fname = join(dname, 'data.nii.gz')
fbvals = join(dname, 'bvals')
fbvecs = join(dname, 'bvecs')
fmask = join(dname, 'nodif_brain_mask.nii.gz')

print('>>> Loading Raw data, b-values and masking background...')
print(fname)

img = nib.load(fname)
data = img.get_data()

affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]

bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

from dipy.core.gradients import gradient_table

gtab = gradient_table(bvals, bvecs, b0_threshold=10)

b0_index = np.where(gtab.b0s_mask==True)[0]

mask = nib.load(fmask).get_data()

print(data.shape)
print(affine)
print(nib.aff2axcodes(affine))

print('>>> Resample data to 1x1x1 mm^3...')

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine,
                          zooms=zooms,
                          new_zooms=(1., 1., 1.))

mask2, affine2 = resample(mask, affine,
                          zooms=zooms,
                          new_zooms=(1., 1., 1.))

mask2[mask2 > 0] = 1
data2[mask2 == 0] = np.zeros(data2.shape[-1])

del data, affine, zooms

print(data2.shape)
print(affine2)
print(nib.aff2axcodes(affine2))

print('>>> Save resampled data, masks and S0...')

fname2 = join(dname, 'dwi_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(data2, affine2), fname2)

fname2_mask = join(dname, 'dwi_mask_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(mask2.astype(np.uint8), affine2), fname2_mask)

fname2_S0 = join(dname, 'dwi_S0_1x1x1.nii.gz')

S0s = data2[..., b0_index]
S0 = np.mean(S0s, axis=-1)

nib.save(nib.Nifti1Image(S0, affine2), fname2_S0)

print('>>> Calculate FA...')

ten = TensorModel(gtab)
tenfit = ten.fit(data2, mask2)
fname2_fa = join(dname, 'dwi_fa_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(tenfit.fa, affine2), fname2_fa)

del data2, mask2
