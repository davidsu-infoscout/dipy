import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe


print('>>> Use tract_querier to extract known bundles...')
print('-------------------------------------------------')

home = expanduser("~")
dname = join(home, 'Data', 'MPI_elef')
subjid = 'MPI'
dname_subjs = environ['SUBJECTS_DIR']

dname_subjs = environ['SUBJECTS_DIR']
fwmparc_ants_nii = join(dname_subjs, subjid, 'mri', 'wmparc_ants_S0.nii.gz')
sl_fname = join(dname, 'csd_streamlines.trk')

bundles_base_name = join(dname, 'bundles.trk')
cmd_tq = "tract_querier -t " + sl_fname + " -a " + fwmparc_ants_nii + " -q freesurfer_queries.qry -o " + bundles_base_name
print(cmd_tq)
pipe(cmd_tq)

