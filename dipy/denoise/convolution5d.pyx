import numpy as np
cimport cython

from dipy.denoise.kernel import EnhancementKernel
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh

def convolve_5d(odfs_sh, kernel, test_mode=False):
    """ Perform the shift-twist convolution with the ODF data and 
        the lookup-table of the kernel.

        Parameters
        ----------
        odfs : array of double
            The ODF data in spherical harmonics format
        lut : array of double
            The 5D lookup table
        test_mode : boolean
            Reduced convolution in one direction only for testing
            
        Returns
        -------
        output : array of double
            The ODF data after convolution enhancement
        """
    # convert the ODFs from SH basis to DSF
    sphere = get_sphere('repulsion100')
    odfs_dsf = sh_to_sf(odfs_sh, sphere, sh_order=8, basis_type=None)

    # perform the convolution
    output = perform_convolution(odfs_dsf, 
                        kernel.get_lookup_table(),
                        test_mode)
    
    # convert back to SH
    output_sh = sf_to_sh(output, sphere, sh_order=8)
    
    return output_sh
    
def convolve_5d_sf(odfs_sf, kernel, test_mode=False):
    """ Perform the shift-twist convolution with the ODF data and 
        the lookup-table of the kernel.

        Parameters
        ----------
        odfs : array of double
            The ODF data sampled on a sphere
        lut : array of double
            The 5D lookup table
        test_mode : boolean
            Reduced convolution in one direction only for testing

        Returns
        -------
        output : array of double
            The ODF data after convolution enhancement
        """
    # perform the convolution
    output = perform_convolution(odfs_sf, 
                        kernel.get_lookup_table(),
                        test_mode)
    return output
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double [:, :, :, ::1] perform_convolution (double [:, :, :, ::1] odfs, 
                                                double [:, :, :, :, ::1] lut,
                                                int test_mode):
    """ Perform the shift-twist convolution with the ODF data 
        and the lookup-table of the kernel.

        Parameters
        ----------
        odfs : array of double
            The ODF data sampled on a sphere
        lut : array of double
            The 5D lookup table
        test_mode : boolean
            Reduced convolution in one direction only for testing
            
        Returns
        -------
        output : array of double
            The ODF data after convolution enhancement
        """
        
    cdef:
        double [:, :, :, ::1] output = np.array(odfs, copy=True)
        int OR1 = lut.shape[0]
        int OR2 = lut.shape[1]
        int N = lut.shape[2]
        int hn = (N-1)/2
        double totalval
        int nx = odfs.shape[0]
        int ny = odfs.shape[1]
        int nz = odfs.shape[2]
        int cx, cy, cz, x, y, z
    
    if test_mode:
        OR2 = 1;
        
    with nogil:
        # loop over ODFs cx,cy,cz
        for corient in range(0,OR1):
            for cy in range(0,ny):
                for cz in range(0,nz):
                    for cx in range(0,nx):
                        
                        totalval = 0.0
                        # loop over kernel x,y,z
                        for x in range(cx-hn,cx+hn+1):
                            for y in range(cy-hn,cy+hn+1):
                                for z in range(cz-hn,cz+hn+1):
                                    for orient in range(0,OR2):

                                        if  y < 0 or y >= ny or \
                                            x < 0 or x >= nx or \
                                            z < 0 or z >= nz:
                                            continue
                                        totalval += odfs[x, y, z, orient] * \
                                        lut[corient, orient, x-(cx-hn), y-(cy-hn), z-(cz-hn)]
                        output[cx, cy, cz, corient] = totalval

    return output
