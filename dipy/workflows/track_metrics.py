
from __future__ import division, print_function
import os
from glob import glob
import logging

import numpy as np
import nibabel as nib
import tractconverter

from dipy.tracking.utils import density_map
from dipy.workflows.utils import choose_create_out_dir


def track_density_flow(tractograms, ref_files, up_factor=1.0, out_dir='',
                       out_tdi='tdi.nii.gz'):
    """ Workflow for tract density computation.

    Parameters
    ----------
    tractograms : string
        Path to the tractogram on which to perform tract density computation.
        This path may contain wildcards to process multiple inputs at once.
    ref_files : string
        Path to the reference volumes. This path may contain wildcards to use
        multiple masks at once.
    up_factor : float, optional
        Factor by which to upsample the resulting volume. (default 1.0)
    out_dir : string, optional
        Output directory (default input file directory)
    out_tdi : string, optional
        Tract density file name (default 'tdi.nii.gz')
    """
    for tract_file, ref_file in zip(glob(tractograms), glob(ref_files)):
        logging.info('Computing track density for {0}'.format(tract_file))
        logging.info('Upsampling factor: {0}'.format(up_factor))
        ref = nib.load(ref_file)
        ref_head = ref.get_header()
        pos_factor = ref_head['pixdim'][1:4] / up_factor
        data_shape = np.array(ref.shape) * up_factor
        data_shape = tuple(data_shape.astype('int32'))

        tract_format = tractconverter.detect_format(tract_file)
        tract = tract_format(tract_file, anatFile=ref_file)

        streamlines = [i * up_factor for i in tract]

        # Need to fix scaling
        affine = np.eye(4)
        affine[:3, :3] = ref.get_affine()[:3, :3]
        #affine[:3, :] += 0.5
        # Need to adjust the affine to take upsampling into account
        tdi_map = density_map(streamlines, data_shape, affine=affine)
        affine[0, 0] /= up_factor
        affine[1, 1] /= up_factor
        affine[2, 2] /= up_factor
        affine[:3, :] = (ref.get_affine()[:3, :] * up_factor)

        map_img = nib.Nifti1Image(tdi_map.astype(np.float32), affine)

        if len(tdi_map.shape) > 3:
            pos_factor += [1]

        out_dir_path = choose_create_out_dir(out_dir, tractograms)

        map_img.get_header().set_zooms(pos_factor)
        map_img.get_header().set_qform(ref_head.get_qform())
        map_img.get_header().set_sform(ref_head.get_sform())
        map_img.to_filename(os.path.join(out_dir_path, out_tdi))
        logging.info('Track density map saved as: {0}'.
                     format(os.path.join(out_dir_path, out_tdi)))