import numpy as np
from os.path import join as pjoin
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.align.aniso2iso import resample
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.peaks import peaks_from_model


def separate_multi_shell_data(gtab, data, mask, shell=1):

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    if shell == 1:
        ind1000 = (gtab.bvals < 10) | ((gtab.bvals < 1100) & (gtab.bvals > 900))
        S1000 = data[..., ind1000]
        gtab1000 = gradient_table(bvals[ind1000], bvecs[ind1000, :],
                                  b0_threshold=10)
        return gtab1000, S1000

    if shell == 2:
        ind2000 = (gtab.bvals < 10) | ((gtab.bvals < 2100) & (gtab.bvals > 1900))
        S2000 = data[..., ind2000]
        gtab2000 = gradient_table(bvals[ind2000], bvecs[ind2000, :],
                                  b0_threshold=10)
        return gtab2000, S2000

    if shell == 3:
        ind3000 = (gtab.bvals < 10) | ((gtab.bvals < 3100) & (gtab.bvals > 2900))
        S3000 = data[..., ind3000]
        gtab3000 = gradient_table(bvals[ind3000], bvecs[ind3000, :],
                                  b0_threshold=10)
        return gtab3000, S3000


def load_nifti(fname, verbose=False):
    img = nib.load(fname)
    data = img.get_data()
    affine = img.get_affine()
    if verbose:
        print(fname)
        print(data.shape)
        print(affine)
        print(img.get_header().get_zooms()[:3])
        print(nib.aff2axcodes(affine))
        print
    return data, affine


def load_data(fraw, fmask, fbval, fbvec, verbose=False):
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs, b0_threshold=10)
    data, affine = load_nifti(fraw, verbose)
    mask, _ = load_nifti(fmask)
    return gtab, data, affine, mask


def pfm(model, data, mask, sphere, parallel=True, min_angle=25.0,
        relative_peak_th=0.5, sh_order=8):

    peaks = peaks_from_model(model=model,
                             data=data,
                             mask=mask,
                             sphere=sphere,
                             relative_peak_threshold=relative_peak_th,
                             min_separation_angle=min_angle,
                             return_odf=False,
                             return_sh=True,
                             sh_order=sh_order,
                             npeaks=5,
                             parallel=parallel)
    return peaks


def csd(gtab, data, affine, mask, response, sphere, min_angle=25.0,
        relative_peak_th=0.5, sh_order=8):
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
    peaks = pfm(model, data, mask, sphere, min_angle=min_angle,
                relative_peak_th=relative_peak_th, sh_order=sh_order)
    return peaks


# Load HCP Q3 preprocessesd data

dname = '/home/eleftherios/Data/HCP_Elef/80/100307/T1w/Diffusion'

fbval = pjoin(dname, 'bvals')
fbvec = pjoin(dname, 'bvecs')
fmask = pjoin(dname, 'nodif_brain_mask.nii.gz')
fdwi = pjoin(dname, 'data.nii.gz')

verbose = True
with_resampling = True

first_shell_obtain = False

preserve_memory = True
first_shell_load = True
resolution = 2 # 1.25


# Separate first shell data

if first_shell_obtain:

    gtab, data, affine, mask = load_data(fdwi, fmask, fbval, fbvec, verbose)

    shell_1000 = separate_multi_shell_data(gtab, data, mask, shell=1)
    gtab_1000, data_1000 = shell_1000

    if preserve_memory:
        del data

    if with_resampling:

        data2, affine2 = resample(data_1000, affine, (1.25,) * 3, (2.,) * 3)
        mask2, _ = resample(mask, affine, (1.25,) * 3, (2.,) * 3, order=0)
        nib.save(nib.Nifti1Image(data2, affine2),
                 pjoin(dname, 'data_1000_2.nii.gz'))
        nib.save(nib.Nifti1Image(mask2, affine2),
                 pjoin(dname, 'nodif_brain_mask_2.nii.gz'))

    nib.save(nib.Nifti1Image(data_1000, affine),
             pjoin(dname, 'data_1000.nii.gz'))

    np.savetxt(pjoin(dname, 'bvals_1000'), gtab_1000.bvals)
    np.savetxt(pjoin(dname, 'bvecs_1000'), gtab_1000.bvecs.T)

# Load first shell only

if first_shell_load:

    fbval = pjoin(dname, 'bvals_1000')
    fbvec = pjoin(dname, 'bvecs_1000')

    if resolution == 1.25:
        fdwi = pjoin(dname, 'data_1000.nii.gz')
        fmask = pjoin(dname, 'nodif_brain_mask.nii.gz')
    if resolution == 2:
        fdwi = pjoin(dname, 'data_1000_2.nii.gz')
        fmask = pjoin(dname, 'nodif_brain_mask_2.nii.gz')


gtab, data, affine, mask = load_data(fdwi, fmask, fbval, fbvec, verbose)

# Calculate Constrained Spherical Deconvolution

response, ratio = auto_response(gtab, data, roi_radius=15, fa_thr=0.7)

print('Response function', response)

sphere = get_sphere('symmetric724')

peaks = csd(gtab, data, affine, mask, response, sphere, min_angle=25.0,
            relative_peak_th=0.5, sh_order=8)

if resolution == 1.25:
    nib.save(nib.Nifti1Image(peaks.shm_coeff, affine),
             pjoin(dname, 'sh_1000.nii.gz'))


if resolution == 2:
    nib.save(nib.Nifti1Image(peaks.shm_coeff, affine),
             pjoin(dname, 'sh_1000_2.nii.gz'))

np.savetxt(pjoin(dname, 'B.txt'), peaks.B)
