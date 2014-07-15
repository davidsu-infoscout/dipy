import nibabel as nib
from dipy.reconst.shore import ShoreModel
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.peaks import peaks_from_model

dname = '/home/eleftherios/Data/Alessandro_Crimi/data/'
fdwi = dname + 'dwi1_fsl.nii.gz'
fbval = dname + 'bvals.txt'
fbvec = dname + 'bvec.txt'


bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

print('data.shape (%d, %d, %d, %d)' % data.shape)

img = nib.load(fdwi)
data = img.get_data()
affine = img.get_affine()

mask = data[..., 0] > 50

1/0

tensor_model = TensorModel(gtab, mask)

tensor_fit = tensor_model.fit(data, mask)

radial_order = 6
zeta = 700
lambdaN = 1e-8
lambdaL = 1e-8
shore_model = ShoreModel(gtab, radial_order=radial_order,
                         zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)

sphere = get_sphere('symmetric724')



csd_peaks = peaks_from_model(model=csd_model,
                             data=data_small,
                             sphere=sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25,
                             parallel=True)
print('odf.shape (%d, %d, %d)' % odf.shape)

# r = fvtk.ren()
# sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, colormap='jet')
# sfu.RotateX(-90)
# fvtk.add(r, sfu)
# fvtk.record(r, n_frames=1, out_path='odfs.png', size=(600, 600))

