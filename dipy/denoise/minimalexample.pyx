import numpy as np
cimport numpy as cnp
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp

from cython.parallel import parallel, prange
from multiprocessing import cpu_count

def testomp(num_threads=None):

    cdef:
        int testrange
        cnp.npy_intp pp
        int all_cores = openmp.omp_get_num_procs()
        int threads_to_use = -1

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = all_cores

    if have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(threads_to_use)

    testrange = 100
    with nogil, parallel():
        for pp in prange(testrange, schedule='guided'):
            pp += 1

    if have_openmp and num_threads is not None:
        openmp.omp_set_num_threads(all_cores)
