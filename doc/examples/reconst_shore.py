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


dname = '/home/eleftherios/Data/Alessandro_Crimi/data/'
fdwi = dname + 'dwi1_fsl.nii.gz'
fbval = dname + 'bvals.txt'
fbvec = dname + 'bvec.txt'

bvecs = np.loadtxt(fbvec)

gtab = GradientTable(3000 * bvecs)

img = nib.load(fdwi)
data = img.get_data()
affine = img.get_affine()

print('data.shape (%d, %d, %d, %d)' % data.shape)

mask = data[..., 0] > 50

tensor_model = TensorModel(gtab)

tensor_fit = tensor_model.fit(data, mask)

radial_order = 6
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
shore_model = ShoreModel(gtab, radial_order=radial_order,
                         zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL,
                         constrain_e0=True)

sphere = get_sphere('symmetric724')

shore_peaks = peaks_from_model(model=shore_model,
                               data=data,
                               mask=mask,
                               sphere=sphere,
                               relative_peak_threshold=.5,
                               min_separation_angle=25,
                               parallel=True)

ren = fvtk.ren()
fvtk.add(ren, fvtk.peaks(shore_peaks.peak_dirs[:, :, 73/2],
         shore_peaks.peak_values[:, :, 73/2], scale=0.5))
fvtk.show(ren)

nib.save(nib.Nifti1Image(tensor_fit.fa, affine), 'FA_map.nii.gz')
nib.save(nib.Nifti1Image(shore_peaks.shm_coeff, affine), 'SH_map.nii.gz')

stopping_values = np.zeros(shore_peaks.peak_values.shape)
stopping_values[:] = tensor_fit.fa[..., None]

streamline_generator = EuDX(stopping_values,
                            shore_peaks.peak_indices,
                            seeds= 10**6,
                            odf_vertices=sphere.vertices,
                            a_low=0.1)

streamlines = [streamline for streamline in streamline_generator]

import nibabel as nib

hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = img.get_header().get_zooms()[:3]
hdr['voxel_order'] = 'LAS'
hdr['dim'] = tensor_fit.fa.shape[:3]

shore_streamlines_trk = ((sl, None, None) for sl in streamlines)

shore_sl_fname = 'shore_streamlines.trk'

nib.trackvis.write(shore_sl_fname, shore_streamlines_trk, hdr, points_space='voxel')

# r = fvtk.ren()
# sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, colormap='jet')
# sfu.RotateX(-90)
# fvtk.add(r, sfu)
# fvtk.record(r, n_frames=1, out_path='odfs.png', size=(600, 600))

