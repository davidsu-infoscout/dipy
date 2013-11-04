import nibabel as nib
from os import environ
from os.path import expanduser, join, dirname, realpath
from dipy.external.fsl import pipe
import sys
#from subjects import *

if __name__ == '__main__':

    dname = sys.argv[1]

    print('>>> Use tract_querier to extract known bundles...')
    print('-------------------------------------------------')

    fwmparc_ants_nii = join(dname, 'wmparc_warped_S0.nii.gz')
    sl_fname = join(dname, 'shore_streamlines.trk')

    bundles_base_name = join(dname, 'bundles.trk')
    cmd_tq = "tract_querier -t " + sl_fname + " -a " + fwmparc_ants_nii + " -q freesurfer_queries.qry -o " + bundles_base_name
    print(cmd_tq)
    pipe(cmd_tq)

