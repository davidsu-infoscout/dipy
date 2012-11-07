import nibabel as nib    
from nibabel.orientations import aff2axcodes

fnames = []

dname = '/usr/share/fsl/data/standard/'
fname = dname + 'FMRIB58_FA_1mm.nii.gz'
fnames.append(fname)

dname = '/home/eg309/Data/111104/subj_05/'
#fname = dname + 'data/subj_05/MPRAGE_32/T1_flirt_out.nii.gz'
fname = dname  + 'MPRAGE_32/rawbet.nii.gz'
fnames.append(fname)

fname = dname + '101_32/rawbet.nii.gz'
fnames.append(fname)

fname = dname + '101_32/DTI/fa.nii.gz'
fnames.append(fname)

import numpy as np
np.set_printoptions(2, suppress = True)

for fname in fnames:

    img=nib.load(fname)
    data = img.get_data()
    affine = img.get_affine()

    print fname
    print np.round(affine, 2)
    print aff2axcodes(affine)
