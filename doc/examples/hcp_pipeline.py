import numpy as np
from os.path import join as pjoin
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.align.aniso2iso import resample
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.dti import TensorModel
from dipy.reconst.peaks import peaks_from_model
from dipy.tracking.local import ThresholdTissueClassifier
from dipy.tracking import utils
from dipy.tracking.local import LocalTracking
from dipy.io.trackvis import save_trk
from dipy.direction.probabilistic_direction_getter import (ProbabilisticDirectionGetter,
                                                           _asarray)
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from scipy.ndimage.filters import convolve


def show_peaks(peaks):
    ren = fvtk.ren()

    peak_dirs = peaks.peak_dirs
    peak_values = peaks.peak_values

    slice_no = peak_values.shape[2] / 2

    fvtk.add(ren, fvtk.peaks(peak_dirs[:, :, slice_no:slice_no + 1],
                             peak_values[:, :, slice_no:slice_no + 1]))

    print('Saving illustration as csd_direction_field.png')
    fvtk.show(ren, size=(900, 900))
    fvtk.record(ren, out_path='csd_direction_field.png', size=(900, 900))


def show_streamlines(streamlines):

    ren = fvtk.ren()

    fvtk.add(ren, fvtk.line(streamlines, line_colors(streamlines)))

    fvtk.show(ren)


class MaximumDeterministicDirectionGetter(ProbabilisticDirectionGetter):
    """Return direction of a sphere with the highest probability mass
    function (pmf).
    """
    def get_direction(self, point, direction):
        """Find direction with the highest pmf to updates ``direction`` array
        with a new direction.
        Parameters
        ----------
        point : memory-view (or ndarray), shape (3,)
            The point in an image at which to lookup tracking directions.
        direction : memory-view (or ndarray), shape (3,)
            Previous tracking direction.
        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        # point and direction are passed in as cython memory views
        pmf = self.pmf_gen.get_pmf(point)
        cdf = self._adj_matrix[tuple(direction)] * pmf
        idx = np.argmax(cdf)

        if pmf[idx] == 0:
            return 1

        newdir = self.vertices[idx]
        # Update direction and return 0 for error
        if np.dot(newdir, _asarray(direction)) > 0:
            direction[:] = newdir
        else:
            direction[:] = -newdir
        return 0


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
        print('Loading...')
        print(fname)
        print(data.shape)
        print(affine)
        print(img.get_header().get_zooms()[:3])
        print(nib.aff2axcodes(affine))
        print
    return data, affine


def save_nifti(fname, data, affine):
    if verbose:
        print('Saving...')
        print(fname)
    nib.save(nib.Nifti1Image(data, affine), fname)


def save_peaks(peaks_dir, peaks, tag=''):

    save_nifti(pjoin(peaks_dir, 'peak_dirs' + tag + '.nii.gz'),
               peaks.peak_dirs)

    save_nifti(pjoin(peaks_dir, 'peak_values' + tag + '.nii.gz'),
               peaks.peak_values)

    save_nifti(pjoin(peaks_dir, 'peak_indices' + tag + '.nii.gz'),
               peaks.peak_indices)

def load_peaks(dname, peaks):

    pass


def load_data(fraw, fmask, fbval, fbvec, verbose=False,
              flip_x=False, flip_y=False, flip_z=False):

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    if flip_x:
        bvecs[:, 0] = - bvecs[:, 0]
    if flip_y:
        bvecs[:, 1] = - bvecs[:, 1]
    if flip_z:
        bvecs[:, 2] = - bvecs[:, 2]

    gtab = gradient_table(bvals, bvecs, b0_threshold=10)
    data, affine = load_nifti(fraw, verbose)
    mask, _ = load_nifti(fmask)
    return gtab, data, affine, mask


def create_wmparc_wm_mask(fwmparc, fwm_mask, resolution):

    data, affine = load_nifti(fwmparc)
    # Label information from
    # http://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    # mask = (data >= 3000) | ((data >= 250) & (data <=255)) | (data == 7)  | (data == 46) | (data == 28) | (data == 60)
    mask = ((data >= 3000) | (data < 1000)) & (data != 224)

    mask = mask.astype('f8')

    mask2, affine2 = resample(mask, affine,
                              (0.7,) * 3, (resolution,) * 3, order=0)

    save_nifti(fwm_mask, mask2, affine2)


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


def estimate_sigma(arr, disable_background_masking=False):
    """Standard deviation estimation from local patches
    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be estimated
    disable_background_masking : bool, default False
        If True, uses all voxels for the estimation, otherwise, only non-zeros voxels are used.
        Useful if the background is masked by the scanner.
    Returns
    -------
    sigma : ndarray
        standard deviation of the noise, one estimation per volume.
    """
    k = np.zeros((3, 3, 3), dtype=np.int8)

    k[0, 1, 1] = 1
    k[2, 1, 1] = 1
    k[1, 0, 1] = 1
    k[1, 2, 1] = 1
    k[1, 1, 0] = 1
    k[1, 1, 2] = 1

    if arr.ndim == 3:
        sigma = np.zeros(1, dtype=np.float32)
        arr = arr[..., None]
    elif arr.ndim == 4:
        sigma = np.zeros(arr.shape[-1], dtype=np.float32)
    else:
        raise ValueError("Array shape is not supported!", arr.shape)

    if disable_background_masking:
        mask = arr[..., 0].astype(np.bool)
    else:
        mask = np.ones_like(arr[..., 0], dtype=np.bool)

    conv_out = np.zeros(arr[...,0].shape, dtype=np.float32)
    for i in range(sigma.size):
        convolve(arr[..., i], k, output=conv_out)
        mean_block = np.sqrt(6/7) * (arr[..., i] - 1/6 * conv_out)
        sigma[i] = np.sqrt(np.mean(mean_block[mask]**2))

    return sigma

# Load HCP Q3 preprocessesd data

dname = '/home/eleftherios/Data/HCP_Elef/80/100307/T1w/Diffusion'

fbval = pjoin(dname, 'bvals')
fbvec = pjoin(dname, 'bvecs')
fmask = pjoin(dname, 'nodif_brain_mask.nii.gz')
fdwi = pjoin(dname, 'data.nii.gz')

verbose = True

first_shell_obtain = False

preserve_memory = True
first_shell_load = True

resolution = 2.0 # 1.25, 0.7

tensor_calculate = False
tensor_load = True

wm_mask_from_t1_calculate = True
wm_mask_from_t1_load = True

csd_calculate = True
csd_load = True

tracking_calculate = True

sphere = get_sphere('symmetric724')

# Separate first shell data

if first_shell_obtain:

    gtab, data, affine, mask = load_data(fdwi, fmask, fbval, fbvec, verbose)

    shell_1000 = separate_multi_shell_data(gtab, data, mask, shell=1)
    gtab_1000, data_1000 = shell_1000

    if preserve_memory:
        del data

    if resolution != 1.25:

        data2, affine2 = resample(data_1000, affine,
                                  (1.25,) * 3, (resolution,) * 3)
        mask2, _ = resample(mask, affine,
                            (1.25,) * 3, (resolution,) * 3, order=0)

        save_nifti(pjoin(dname, 'data_1000_' + str(resolution) + '.nii.gz'),
                   data2, affine2)

        save_nifti(pjoin(dname, 'mask_' + str(resolution) + '.nii.gz'),
                   mask2, affine2)

    else:
        save_nifti(pjoin(dname, 'data_1000_' + str(resolution) + '.nii.gz'),
                   data_1000, affine)
        save_nifti(pjoin(dname, 'mask_' + str(resolution) + '.nii.gz'),
                   mask, affine)

    np.savetxt(pjoin(dname, 'bvals_1000'), gtab_1000.bvals)
    np.savetxt(pjoin(dname, 'bvecs_1000'), gtab_1000.bvecs.T)

# Load first shell only

tag = ''

if first_shell_load:

    fbval = pjoin(dname, 'bvals_1000')
    fbvec = pjoin(dname, 'bvecs_1000')

    fdwi = pjoin(dname, 'data_1000_' + str(resolution) + '.nii.gz')
    fmask = pjoin(dname, 'mask_' + str(resolution) + '.nii.gz')

    tag = '_1000_'


gtab, data, affine, mask = load_data(fdwi, fmask, fbval, fbvec, verbose)


# TODO
# estimate sigma
#
#sigma = estimate_sigma(data, disable_background_masking=False)
#
#print(sigma)


# Calculate Tensors FA and MD

if tensor_calculate:
    ten_model = TensorModel(gtab)
    ten_fit = ten_model.fit(data, mask)

    FA = ten_fit.fa
    MD = ten_fit.md

    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))


    save_nifti(pjoin(dname, 'fa' + tag + str(resolution) + '.nii.gz'),
               FA, affine)
    save_nifti(pjoin(dname, 'md' + tag + str(resolution) + '.nii.gz'),
               MD, affine)
    save_nifti(pjoin(dname, 'wm_mask' + tag + str(resolution) + '.nii.gz'),
               wm_mask.astype('f4'), affine)

if tensor_load:

    FA, _ = load_nifti(pjoin(dname, 'fa' + tag + str(resolution) + '.nii.gz'))
    MD, _ = load_nifti(pjoin(dname, 'md' + tag + str(resolution) + '.nii.gz'))
    wm_mask, _ = load_nifti(pjoin(dname, 'wm_mask' + tag + str(resolution) + '.nii.gz'))


if wm_mask_from_t1_calculate:

    # create_wmparc_wm_mask(pjoin(dname, 'aparc+aseg_0.7.nii.gz'),
    #                      pjoin(dname, 'wm_mask_t1_' + str(resolution) + '.nii.gz'),
    #                      resolution)
    create_wmparc_wm_mask(pjoin(dname, 'wmparc_0.7.nii.gz'),
                          pjoin(dname, 'wm_mask_t1_' + str(resolution) + '.nii.gz'),
                          resolution)



if wm_mask_from_t1_load:
    wm_mask, _ = load_nifti(pjoin(dname, 'wm_mask_t1_' + str(resolution) + '.nii.gz'))

1/0

# Calculate Constrained Spherical Deconvolution

if csd_calculate:

    response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

    print('Response function', response)

    peaks = csd(gtab, data, affine, mask, response, sphere, min_angle=25.0,
                relative_peak_th=0.5, sh_order=8)

    save_nifti(pjoin(dname, 'sh' + tag + str(resolution) + '.nii.gz'),
               peaks.shm_coeff, affine)
    np.savetxt(pjoin(dname, 'B.txt'), peaks.B)


if csd_load:

    sh, affine = load_nifti(pjoin(dname, 'sh' + tag + str(resolution) + '.nii.gz'))

if tracking_calculate:

    # show_peaks(peaks)

    #classifier = ThresholdTissueClassifier(FA, .1)
    classifier = ThresholdTissueClassifier(wm_mask.astype('f8'), .5) # with mask smooth and

    seeds = utils.seeds_from_mask(wm_mask, density=[1, 1, 1], affine=affine)

    # Initialization of LocalTracking. The computation happens in the next step.

    streamlines = LocalTracking(peaks, classifier, seeds, affine, step_size=.5)

    #max_dg = MaximumDeterministicDirectionGetter.from_shcoeff(sh,
    #                                                          max_angle=30.,
    #                                                          sphere=sphere)
    #streamlines = LocalTracking(max_dg, classifier, seeds, affine, step_size=.5)

    # Compute streamlines and store as a list.
    streamlines = list(streamlines)

    # show_streamlines(streamlines[:1000])

    save_trk(pjoin(dname, 'streamlines' + tag + str(resolution) + '_wm_mask_pam.trk'),
             streamlines, affine, FA.shape)
