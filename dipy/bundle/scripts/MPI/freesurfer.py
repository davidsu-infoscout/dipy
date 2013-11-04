import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe


home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
subjid = 'MPI'
dname_subjs = environ['SUBJECTS_DIR']
fT1 = join(dname, 't1.nii.gz')

print('>>> Use freesurfer to create the labels from T1...')

freesurfer_cmd = "recon-all -subjid " + subjid + " -i " + fT1 + " -all"
print(freesurfer_cmd)
pipe(freesurfer_cmd)
