import numpy as np
from scipy.spatial import KDTree
from math import sqrt
cimport cython

from dipy.data import get_sphere
from dipy.denoise.kernel import EnhancementKernel
from dipy.core.ndindex import ndindex

cdef class FBCMeasures:

    cdef double [:, :, :] streamline_points
    cdef double [:, :] streamlines_lfbc
    cdef double [:] streamlines_rfbc

    ## Python functions
    
    def __init__(self, streamlines, kernel):
    
        self.compute(streamlines, kernel)
        
    def get_points(self):
        return self.streamline_points
        
    def get_lfbc(self):
        return self.streamlines_lfbc
        
    def get_rfbc(self):
        return self.streamlines_rfbc
        
    def get_points_rfbc_thresholded(self, threshold):
        # select fibers that are above the rfbc threshold
        selectedFibers = np.select([self.streamlines_rfbc>threshold], [self.streamline_points])
        pylist = selectedFibers.tolist()
        for i in range(len(pylist)):
            # remove empty fiber points
            pylist[i] = list(filter(([-1,-1,-1]).__ne__, pylist[i]))
        return pylist
        
    
    ## Cython functions
    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef void compute(self, 
                        py_streamlines,
                        kernel):
    
        cdef:
            int numberOfFibers
            int maxLength
            int dim
            double [:, :, :] streamlines
            int [:] streamlines_length
            double [:, :, :] streamlines_tangent
            int [:, :] streamlines_nearestp
            double [:, :] streamline_scores
            double [:] tangent
            int lineId, pointId
            int lineId2, pointId2
            double score
            int xd, yd, zd
            double [:, :, :, :, ::1] lut
            int N
            int hn
        
        numberOfFibers = len(py_streamlines)
        #numberOfFibers = 100 # temp
        streamlines_length = np.array([len(x) for x in py_streamlines])
        maxLength = max(streamlines_length)
        dim = 3
        
        lut = kernel.get_lookup_table()
        N = lut.shape[2]
        hn = (N-1)/2
        
        streamlines = np.zeros((numberOfFibers, maxLength, dim), dtype=np.float64)-1
        streamlines_tangents = np.zeros((numberOfFibers, maxLength, dim), dtype=np.float64)
        streamlines_nearestp = np.zeros((numberOfFibers, maxLength), dtype=np.int)
        streamline_scores = np.zeros((numberOfFibers, maxLength), dtype=np.float64)-1
        
        # copy python streamlines into c++ buffer
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]):
                for dim in range(3):
                    streamlines[lineId, pointId, dim] = py_streamlines[lineId][pointId][dim]
        self.streamline_points = streamlines
        
        # compute tangents
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                tangent = np.subtract(streamlines[lineId, pointId+1], streamlines[lineId, pointId])
                streamlines_tangents[lineId, pointId] = np.divide(tangent, np.sqrt(np.dot(tangent, tangent)))
        
        # estimate which kernel LUT index corresponds to angles
        tree = KDTree(kernel.get_orientations())
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                streamlines_nearestp[lineId, pointId] = tree.query(streamlines[lineId, pointId])[1]
        
        # compute fiber LFBC measures
        with nogil:
            for lineId in range(numberOfFibers):
                for pointId in range(streamlines_length[lineId]-1):
                    score = 0.0
                    for lineId2 in range(numberOfFibers):
                    
                        # skip lfbc computation with itself
                        if lineId == lineId2:
                            continue
                            
                        for pointId2 in range(streamlines_length[lineId2]-1):
                            # compute displacement
                            xd = int(streamlines[lineId, pointId, 0] - streamlines[lineId2, pointId2, 0] + 0.5) # fast round
                            yd = int(streamlines[lineId, pointId, 1] - streamlines[lineId2, pointId2, 1] + 0.5) # fast round
                            zd = int(streamlines[lineId, pointId, 2] - streamlines[lineId2, pointId2, 2] + 0.5) # fast round
                            
                            # if position is outside the kernel bounds, skip
                            if xd > hn or -xd > hn or \
                               yd > hn or -yd > hn or \
                               zd > hn or -zd > hn:
                                continue
                            
                            # grab kernel value from LUT
                            score += lut[streamlines_nearestp[lineId, pointId], 
                                         streamlines_nearestp[lineId2, pointId2], 
                                         hn+xd, 
                                         hn+yd, 
                                         hn+zd]  # ang_v, ang_r, x, y, z
                                    
                    streamline_scores[lineId, pointId] = score
        
        # Save LFBC as class member
        self.streamlines_lfbc = streamline_scores
        
        # compute RFBC for each fiber
        self.streamlines_rfbc = compute_rfbc(streamlines_length, streamline_scores)
        
        #print np.array(streamline_scores)
        #print np.shape(streamline_scores)
        #print np.array(self.streamlines_rfbc)
        
def compute_rfbc(streamlines_length, streamline_scores):
    intLength = min(np.amin(streamlines_length), 7)
    intValue = np.apply_along_axis(lambda x: min_moving_average(x, intLength), 1, streamline_scores)
    #print np.shape(intValue)
    averageTotal = np.mean(np.apply_along_axis(lambda x:np.mean(np.extract(x>=0, x)), 1, streamline_scores))
    #print np.array(intValue)
    #print averageTotal
    return intValue/averageTotal
            
def min_moving_average(a, n):
    ret = np.cumsum(np.extract(a>=0, a))
    ret[n:] = ret[n:] - ret[:-n]
    return np.amin(ret[n - 1:] / n)
        

    
    
    
    
    