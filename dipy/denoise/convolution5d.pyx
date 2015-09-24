import numpy as np
cimport cython

from dipy.denoise.kernel import EnhancementKernel
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf

def convolve_5d(odfs_sh, kernel):

    #sphere = get_sphere('repulsion100')
    # convert the ODFs from SH basis to DSF
    #odfs_dsf = sh_to_sf(odfs_sh, sphere, sh_order=8, basis_type=None)

    output = perform_convolution(odfs_sh, 
                        kernel.get_lookup_table())
    return output


    
cdef double [:, :, :, ::1] perform_convolution (double [:, :, :, ::1] odfs, 
                                                double [:, :, :, :, ::1] lut):
    
    cdef double [:, :, :, ::1] output = np.array(odfs, copy=True)

    cdef int OR = lut.shape[0]
    cdef int N = lut.shape[2]
    cdef int hn = (N-1)/2
    cdef double totalval

    cdef int nx = odfs.shape[0]
    cdef int ny = odfs.shape[1]
    cdef int nz = odfs.shape[2]

    # loop over ODFs cx,cy,cz
    for corient in range(0,OR):
        for cy in range(0,nx):
            for cz in range(0,nz):
                for cx in range(0,ny):
                    
                    totalval = 0.0
                    # loop over kernel x,y,z
                    for x in range(cx-hn,cx+hn):
                        for y in range(cy-hn,cy+hn):
                            for z in range(cz-hn,cz+hn):
                                for orient in range(0,OR):

                                    if  y < 0 or y >= ny or \
                                        x < 0 or x >= nx or \
                                        z < 0 or z >= nz:
                                        continue
                                    totalval += odfs[x, y, z, orient] * \
                                    lut[corient, orient, x-(cx-hn), y-(cy-hn), z-(cz-hn)]
                    output[cx,cy,cz,corient] = totalval

    return output
