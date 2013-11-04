import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe


home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
subjid = 'MPI'
dname_subjs = environ['SUBJECTS_DIR']


print('>>> Change wmparc.mgz to wmparc.nii.gz...')
print('-----------------------------------------')

fwmparc = join(dname_subjs, subjid, 'mri', 'wmparc.mgz')
fwmparc_nii = join(dname_subjs, subjid, 'mri', 'wmparc.nii.gz')
fwmparc_orig = join(dname_subjs, subjid, 'mri', 'orig', '001.mgz')
fwmparc_flirt_nii = join(dname_subjs, subjid, 'mri', 'wmparc_flirt.nii.gz')
fwmparc_ants_nii = join(dname_subjs, subjid, 'mri', 'wmparc_ants_S0.nii.gz')

cmd_mgz2nii = "mri_convert " + fwmparc + " " + fwmparc_nii + " -rl " + fwmparc_orig + " -rt nearest"
print(cmd_mgz2nii)
pipe(cmd_mgz2nii)

print('>>> Apply flirt transformation...')
print('---------------------------------')

fS0 = join(dname, 'dwi_nlm_S0_1x1x1.nii.gz')
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

