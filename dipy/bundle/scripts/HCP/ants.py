import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe

id = 0
base_dirs = ['100307', '111312', '194140', '865363', '889579']
home = expanduser("~")
dname = join(home, 'Data', 'HCP', 'Q1', base_dirs[id])

print('>>> Warp T1 to S0 using ANTS...')

fT1 = join(dname, 'T1w_acpc_dc_restore_brain.nii.gz')

fT1_flirt = join(dname, 't1_flirt.nii.gz')
fmat = join(dname, 'flirt_affine.mat')
fS0 = join(dname, 'dwi_S0_1x1x1.nii.gz')
fFA = join(dname, 'dwi_fa_1x1x1.nii.gz')
fT1wS0 = join(dname, 't1_warped_S0.nii.gz')
fdef = join(dname, 'MultiVarNew')

img_T1 = nib.load(fT1)
print(img_T1.get_data().shape)
print(img_T1.get_affine())
print(nib.aff2axcodes(img_T1.get_affine()))

del img_T1

flirt_cmd = "flirt -in " + fT1 + " -ref " + fFA + " -out " + fT1_flirt + " -omat " + fmat
print(flirt_cmd)
pipe(flirt_cmd)

br1 = "[" + fS0 + ", " + fT1_flirt + ", 1, 4]"
br2 = "[" + fFA + ", " + fT1_flirt + ", 1.5, 4]"

ants_cmd1 = "ANTS 3 -m CC" + br1 + " -m CC" + br2 + " -o " + fdef + " -i 75x75x10 -r Gauss[3,0] -t SyN[0.25]"
ants_cmd2 = "WarpImageMultiTransform 3 " + fT1_flirt + " " + fT1wS0 + " -R " + fS0 + " " + fdef + "Warp.nii.gz " + fdef + "Affine.txt"

print(ants_cmd1)
pipe(ants_cmd1)
print(ants_cmd2)
pipe(ants_cmd2)


