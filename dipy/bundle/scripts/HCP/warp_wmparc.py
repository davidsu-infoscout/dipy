import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe
from subjects import *

fwmparc_nii = join(dname, 'wmparc.nii.gz')
fwmparc_flirt_nii = join(dname, 'wmparc_flirt.nii.gz')
fwmparc_ants_nii = join(dname, 'wmparc_warped_S0.nii.gz')

print('>>> Apply flirt transformation...')
print('---------------------------------')

fS0 = join(dname, 'dwi_S0_1x1x1.nii.gz')
fmat = join(dname, 'flirt_affine.mat')
fdef = join(dname, 'MultiVarNew')

cmd_applyxfm = "flirt -in " + fwmparc_nii + " -ref " + fS0 + " -out " + fwmparc_flirt_nii + " -init " + fmat + " -applyxfm -interp nearestneighbour"
print(cmd_applyxfm)
pipe(cmd_applyxfm)

print('>>> Apply ants transformation...')
print('--------------------------------')

cmd_warp = "WarpImageMultiTransform 3 " + fwmparc_flirt_nii + " " + fwmparc_ants_nii + " -R " +  fS0 + " " + fdef + "Warp.nii.gz " + fdef + "Affine.txt" + " --use-NN"
print(cmd_warp)
pipe(cmd_warp)

