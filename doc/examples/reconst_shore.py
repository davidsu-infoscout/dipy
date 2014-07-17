import nibabel as nib
from dipy.reconst.shore import ShoreModel
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table, GradientTable
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.peaks import peaks_from_model
from dipy.tracking.eudx import EuDX
from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
from dipy.reconst.csdeconv import auto_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel


def save_nifti(data, affine, dtype, filename):
    nib.save(nib.Nifti1Image(data.astype(dtype), affine), filename)


def save_trk(streamlines, img, filename):
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = img.get_data().shape[:3]

    shore_streamlines_trk = ((sl, None, None) for sl in streamlines)
    shore_sl_fname = filename
    nib.trackvis.write(shore_sl_fname, shore_streamlines_trk, hdr, points_space='voxel')


def fvtk_gradients(gtab, radius, out_path, n_frames=1):

    ren = fvtk.ren()
    fvtk.add(ren, fvtk.point(gtab.gradients, fvtk.colors.red, point_radius=radius))
    fvtk.show(ren)
    fvtk.record(ren, out_path, n_frames=n_frames)


def fvtk_bvecs(bvecs, radius, out_path, n_frames=1):

    ren = fvtk.ren()
    fvtk.add(ren, fvtk.point(bvecs, fvtk.colors.red, point_radius=radius))
    fvtk.show(ren)
    fvtk.record(ren, out_path, n_frames=n_frames)


def dipy_to_fibernav_peaks(peaks):
    peaks_dirs = peaks.peak_dirs
    peaks_dirs = peaks_dirs.reshape(peaks_dirs.shape[:3] + (15,))
    return peaks_dirs


def eudx_streamlines(stopping_volume, stopping_threshold, 
                     peaks, seeds, sphere, final_number):

    stopping_values = np.zeros(peaks.peak_values.shape)
    stopping_values[:] = stopping_volume[..., None]

    print('Generating and saving streamlines ...')
    streamline_generator = EuDX(stopping_values,
                                peaks.peak_indices,
                                seeds = seeds,
                                odf_vertices=sphere.vertices,
                                a_low=stopping_threshold)

    return [streamline for streamline in streamline_generator][:final_number]
   

if __name__ == '__main__':
    
    dname = '/home/eleftherios/Data/Alessandro_Crimi/data/'
    fdwi = dname + 'dwi1_fsl.nii.gz'
    fbval = dname + 'bvals.txt'
    fbvec = dname + 'bvec.txt'
    seeds = 10**6
    stopping_threshold = 0.1
    denoise = True
    reconst_shore = True
    reconst_csd = False

    final_number = 400000
    fname_ending = '_crimi'
    
    print('Loading dataset ...')

    bvecs = np.loadtxt(fbvec)

    gtab = GradientTable(3000 * bvecs)

    img = nib.load(fdwi)
    data = img.get_data()
    affine = img.get_affine()
    
    print('data.shape (%d, %d, %d, %d)' % data.shape)

    _, mask = median_otsu(data, 4, 4)

    save_nifti(mask, affine, np.uint8, 'mask' + fname_ending + '.nii.gz')

    sigma = np.std(data[..., 0][~mask])

    print('Starting nlmeans...')

    if denoise:
        
        data = nlmeans(data, sigma=sigma/3., mask=mask)

    tensor_model = TensorModel(gtab)

    print('Starting Tensor fitting...')

    tensor_fit = tensor_model.fit(data, mask)

    save_nifti(tensor_fit.fa, affine, np.float32, 'FA' + fname_ending + '.nii.gz')

    sphere = get_sphere('symmetric724')

    if reconst_shore:
        print('Starting Shore fitting and peaks...')

        shore_model = ShoreModel(gtab, constrain_e0=True)        

        shore_peaks = peaks_from_model(model=shore_model,
                                       data=data,
                                       mask=mask,
                                       sphere=sphere,
                                       relative_peak_threshold=.5,
                                       min_separation_angle=25,
                                       parallel=True)
       

        save_nifti(dipy_to_fibernav_peaks(shore_peaks), affine, 
                   np.float32, 'shore_dirs_fibernav' + fname_ending + '.nii.gz')

        print('Saving Shore ODFs as SH ...')
        
        save_nifti(shore_peaks.shm_coeff, affine, np.float32, 'SH' + fname_ending + '.nii.gz')

        streamlines = eudx_streamlines(tensor_fit.fa, 0.1, 
                                       shore_peaks, seeds, sphere, final_number)

        save_trk(streamlines, img, 'shore_streamlines' + fname_ending + '.trk')

    if reconst_csd:

        response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
        response = (np.array([0.0015, 0.0003, 0.0003]), 163.)
        ratio = 3/15.

        csd_model = ConstrainedSphericalDeconvModel(gtab, response)

        csd_peaks = peaks_from_model(model=csd_model,
                                    data=data,
                                    mask=mask,
                                    sphere=sphere,
                                    relative_peak_threshold=.5,
                                    min_separation_angle=25,
                                    parallel=True)

        save_nifti(dipy_to_fibernav_peaks(csd_peaks), affine, 
                   np.float32, 'csd_dirs_fibernav' + fname_ending + '.nii.gz')

        print('Saving CSD ODFs as SH ...')
        
        save_nifti(csd_peaks.shm_coeff, affine, np.float32, 'csd_SH' + fname_ending + '.nii.gz')

        streamlines = eudx_streamlines(tensor_fit.fa, 0.1, 
                                       csd_peaks, seeds, sphere, final_number)
        
        save_trk(streamlines, img, 'csd_streamlines' + fname_ending + '.trk')


