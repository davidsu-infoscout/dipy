"""
===============================================
Alternative brain segmentation for DWI datasets
===============================================

This example provides an alternative for when the `median_otsu` function may
not give you a good mask.
"""

from dipy.segment.mask import multi_median, otsu
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
import nibabel as nib

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()
affine = img.get_affine()
zooms = img.get_header().get_zooms()[:3]

"""
The trick here is that we are going to use only a few of the diffusion weighted
volumes and not the S0s to create the mask.
"""

data2 = data[..., [10, 20, 30, 50, 60]]

"""
First we start with a median filter
"""

multi_median(data2, median_radius=3, numpass=1)


"""
Average all volumes together
"""

mask = np.mean(data2, axis=3)

"""
Threshold the `mask` using Otsu's method.
"""

thresh = otsu(mask)

mask = mask > thresh

nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), 'mask.nii.gz')
