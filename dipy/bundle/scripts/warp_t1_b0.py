
import nibabel as nib
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk


def record_slice(fname, data, k, show=False):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.slicer(data, plane_k=[k]))
    if show: fvtk.show(ren)
    fvtk.record(ren, out_path=fname, size=(600, 600))


home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
fname = join(dname, 'dwi_nlm.nii.gz')

print('>>> 1. Loading Raw data, b-values and masking background...')

img = nib.load(fname)
data = img.get_data()
affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]

fencoding = join(dname, 'encoding.b')
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

print('>>> 2. Resample data to 1x1x1 mm^3...')

from dipy.align.aniso2iso import resample

data2, affine2 = resample(data, affine,
                          zooms=zooms,
                          new_zooms=(1., 1., 1.))

del data, affine, zooms

print(data2.shape)
print(affine2)
print(nib.aff2axcodes(affine2))

print('>>> 3. Save resampled data, masks and S0...')

fname2 = join(dname, 'dwi_nlm_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(data2, affine2), fname2)

mask2 = data2[..., b0_index] > 0

fname2_mask = join(dname, 'dwi_nlm_mask_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(mask2.astype(np.uint8), affine2), fname2_mask)

fname2_S0 = join(dname, 'dwi_nlm_S0_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(data2[..., b0_index], affine2), fname2_S0)

fname2_slice = 'dwi_nlm_slice_1x1x1.png'
record_slice(fname2_slice, data2[..., b0_index], k=data2.shape[2]/2)

print('>>> 4. Calculate FA...')

from dipy.reconst.dti import TensorModel

ten = TensorModel(gtab)
tenfit = ten.fit(data2, mask2)
fname2_fa = join(dname, 'dwi_nlm_fa_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(tenfit.fa, affine2), fname2_fa)

fname_slice_fa = 'dwi_nlm_slice_FA_1x1x1.png'
record_slice(fname_slice_fa, tenfit.fa, k=data2.shape[2]/2)

print('>>> 5. Warp T1 to S0 using ANTS...')

fT1 = join(dname, 't1.nii.gz')
fS0 = join(dname, 'dwi_nlm_S0_1x1x1.nii.gz')
fFA = join(dname, 'dwi_nlm_fa_1x1x1.nii.gz')
fT1wS0 = join(dname, 't1_brain_warped_S0.nii.gz')
fdef = join(dname, 'MultiVar')

img_T1 = nib.load(fT1)
print(img_T1.get_data().shape)
print(img_T1.get_affine())
print(nib.aff2axcodes(img_T1.get_affine()))

del img_T1

from dipy.external.fsl import pipe

# br1 = "[" + fS0 + ", " + fT1 + ", 1, 4]"
# br2 = "[" + fFA + ", " + fT1 + ", 1.5, 4]"

# cmd1 = "ANTS 3 -m CC" + br1 + " -m CC" + br2 + " -o MultiVar -i 75x75x10 -r Gauss[3,0] -t SyN[0.25]"
# cmd2 = "WarpImageMultiTransform 3 " + fT1 + " " + fT1wS0 + " -R " + fS0 + " " + fdef + "Warp.nii.gz " + fdef + "Affine.txt"

# print(cmd1)
# pipe(cmd1)
# print(cmd2)
# pipe(cmd2)

print('>>> 6. Use freesurfer to create the labels...')

