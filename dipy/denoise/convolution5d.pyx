import numpy as np
cimport numpy as cnp
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp
from cython.parallel import prange

from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh

def convolve_5d(odfs_sh, kernel, test_mode=False, num_threads=None):
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
        
    References
    -------
    [DuitsAndFranken2011] Duits, R. and Franken, E. (2011) Morphological and
                      Linear Scale Spaces for Fiber Enhancement in DWI-MRI.
                      J Math Imaging Vis, 46(3):326-368.
    [Portegies2015] J. Portegies, G. Sanguinetti, S. Meesters, and R. Duits. (2015)
                 New Approximation of a Scale Space Kernel on SE(3) and
                 Applications in Neuroimaging. Fifth International
                 Conference on Scale Space and Variational Methods in
                 Computer Vision
    [Portegies2015b] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters, G.Girard,
                 and R. Duits. (2015) Improving Fiber Alignment in HARDI by 
                 Combining Contextual PDE flow with Constrained Spherical 
                 Deconvolution. PLoS One.
    """
    
    # convert the ODFs from SH basis to DSF
    sphere = kernel.get_sphere()
    odfs_dsf = sh_to_sf(odfs_sh, sphere, sh_order=8, basis_type=None)

    # perform the convolution
    output = perform_convolution(odfs_dsf, 
                        kernel.get_lookup_table(),
                        test_mode,
                        num_threads)

    # normalize the output
    output_norm = output * np.amax(odfs_dsf)/np.amax(output)
    
    # convert back to SH
    output_sh = sf_to_sh(output_norm, sphere, sh_order=8)
    
    return output_sh
    
def convolve_5d_sf(odfs_sf, kernel, test_mode=False, num_threads=None):
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
                        test_mode,
                        num_threads)
    return output
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double [:, :, :, ::1] perform_convolution (double [:, :, :, ::1] odfs, 
                                                double [:, :, :, :, ::1] lut,
                                                int test_mode,
                                                object num_threads=None):
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
        cnp.npy_intp OR1 = lut.shape[0]
        int OR2 = lut.shape[1]
        int N = lut.shape[2]
        int hn = (N-1)/2
        double [:, :, :, :] totalval
        int nx = odfs.shape[0]
        int ny = odfs.shape[1]
        int nz = odfs.shape[2]
        int threads_to_use = -1
        int all_cores
        cnp.npy_intp corient, orient, cx, cy, cz, x, y, z

    totalval = np.zeros((OR1,ny,nz,nx))

    #if have_openmp:
    #    all_cores = openmp.omp_get_num_procs()      <--- crashes here

    # if num_threads is not None:
    #     threads_to_use = num_threads
    # else:
    #     threads_to_use = all_cores

    # if have_openmp:
    #     openmp.omp_set_dynamic(0)
    #     openmp.omp_set_num_threads(threads_to_use)
    
    if test_mode:
        OR2 = 1;
        
    with nogil:
        # loop over ODFs cx,cy,cz,corient
        for corient in prange(OR1, schedule='guided'):
            for cy in range(ny):
                for cz in range(nz):
                    for cx in range(nx):
                        
                        totalval[corient, cy, cz, cx] = 0.0
                        # loop over kernel x,y,z,orient
                        for x in range(cx-hn, cx+hn+1):
                            for y in range(cy-hn, cy+hn+1):
                                for z in range(cz-hn, cz+hn+1):
                                    for orient in range(0, OR2):

                                        if  y < 0 or y >= ny or \
                                            x < 0 or x >= nx or \
                                            z < 0 or z >= nz:
                                            continue
                                        totalval[corient, cy, cz, cx] += odfs[x, y, z, orient] * \
                                        lut[corient, orient, x-(cx-hn), y-(cy-hn), z-(cz-hn)]
                        output[cx, cy, cz, corient] = totalval[corient, cy, cz, cx]

    return output
