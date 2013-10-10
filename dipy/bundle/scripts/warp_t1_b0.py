
import nibabel as nib
from os.path import expanduser, join, dirname, realpath
from dipy.viz import fvtk
from dipy.reconst.dti import TensorModel, fractional_anisotropy


def record_slice(fname, data, k, show=False):
    ren = fvtk.ren()
    fvtk.add(ren, fvtk.slicer(data, plane_k=[k]))
    if show: fvtk.show(ren)
    fvtk.record(ren, out_path=fname, size=(600, 600))


def estimate_response(data, w=10):

    ci, cj, ck = np.array(data.shape[:3]) / 2

    roi = data[ci - w: ci + w,
               cj - w: cj + w,
               ck - w: ck + w]

    tenfit = ten.fit(roi)

    FA = fractional_anisotropy(tenfit.evals)

    FA[np.isnan(FA)] = 0

    indices = np.where(FA > 0.7)

    lambdas = tenfit.evals[indices][:, :2]

    S0s = roi[indices][:, np.nonzero(gtab.b0s_mask)[0]]

    S0 = np.mean(S0s)

    l01 = np.mean(lambdas, axis=0)

    evals = np.array([l01[0], l01[1], l01[1]])

    response = (evals, S0)

    return response


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

ten = TensorModel(gtab)
tenfit = ten.fit(data2, mask2)
fname2_fa = join(dname, 'dwi_nlm_fa_1x1x1.nii.gz')
nib.save(nib.Nifti1Image(tenfit.fa, affine2), fname2_fa)

fname_slice_fa = 'dwi_nlm_slice_FA_1x1x1.png'
record_slice(fname_slice_fa, tenfit.fa, k=data2.shape[2]/2)




print('>>> 5. Warp T1 to S0 using ANTS...')

fT1 = join(dname, 't1.nii.gz')
fT1_flirt = join(dname, 't1_flirt.nii.gz')
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

flirt_cmd = "flirt -in " + fT1 + " -ref " + fFA + " -out " + fT1_flirt
print(flirt_cmd)
pipe(flirt_cmd)

br1 = "[" + fS0 + ", " + fT1_flirt + ", 1, 4]"
br2 = "[" + fFA + ", " + fT1_flirt + ", 1.5, 4]"

ants_cmd1 = "ANTS 3 -m CC" + br1 + " -m CC" + br2 + " -o MultiVar -i 75x75x10 -r Gauss[3,0] -t SyN[0.25]"
ants_cmd2 = "WarpImageMultiTransform 3 " + fT1_flirt + " " + fT1wS0 + " -R " + fS0 + " " + fdef + "Warp.nii.gz " + fdef + "Affine.txt"

print(ants_cmd1)
pipe(ants_cmd1)
print(ants_cmd2)
pipe(ants_cmd2)

print('>>> 6. Use freesurfer to create the labels...')

freesurfer_cmd = "recon-all -subjid MPI_T1wS0 -i " + fT1wS0 + " -all"
print(freesurfer_cmd)
pipe(freesurfer_cmd)

print('>>> 7. Generate CSD-based streamlines')

response = estimate_response(data)

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

csd_model = ConstrainedSphericalDeconvModel(gtab, response)

from dipy.data import get_sphere
sphere = get_sphere('symmetric362')

from dipy.reconst.odf import peaks_from_model
peaks = peaks_from_model(model=csd_model,
                         data=data_dmri,
                         mask=mask,
                         sphere=sphere,
                         relative_peak_threshold=0.8,
                         min_separation_angle=45,
                         return_odf=False,
                         return_sh=True,
                         normalize_peaks=False,
                         sh_order=8,
                         npeaks=5,
                         parallel=True, nbr_process=6)

from dipy.tracking.eudx import EuDX

eu = EuDX(tenfit.fa,
          peaks.peak_indices[..., 0],
          seeds=10**4,
          odf_vertices=sphere.vertices,
          a_low=0.1)

streamlines = [streamline for streamline in eu]

streamlines_trk = ((sl, None, None) for sl in streamlines)
sl_fname = 'csd_streamlines.trk'
nib.trackvis.write(sl_fname, streamlines_trk, points_space='rasmm')

