import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe


id = 0
base_dirs = ['100307', '111312', '194140', '865363', '889579']
home = expanduser("~")
dname = join(home, 'Data', 'HCP', 'Q1', base_dirs[id])

print('>>> Use tract_querier to extract known bundles...')
print('-------------------------------------------------')

fwmparc_ants_nii = join(dname, 'wmparc_ants_S0.nii.gz')
sl_fname = join(dname, 'shore_streamlines.trk')

bundles_base_name = join(dname, 'bundles.trk')
cmd_tq = "tract_querier -t " + sl_fname + " -a " + fwmparc_ants_nii + " -q freesurfer_queries.qry -o " + bundles_base_name
print(cmd_tq)
pipe(cmd_tq)

