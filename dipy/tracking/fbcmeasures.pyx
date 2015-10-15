import numpy as np
from scipy.spatial import KDTree
from math import sqrt
cimport cython

from dipy.data import get_sphere
from dipy.denoise.kernel import EnhancementKernel
from dipy.core.ndindex import ndindex

cdef class FBCMeasures:

    cdef double [:, :] streamlines_lfbc

    ## Python functions
    
    def __init__(self, streamlines, kernel):
    
        self.compute(streamlines, kernel)
    
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
            int lineId
            int pointId
            int lineId2
            int pointId2
            double score
            int xd, yd, zd
            double [:, :, :, :, ::1] lut
            int N
            int hn
        
        numberOfFibers = len(py_streamlines)
        numberOfFibers = 3 # temp
        streamlines_length = np.array([len(x) for x in py_streamlines])
        maxLength = max(streamlines_length)
        dim = 3
        
        lut = kernel.get_lookup_table()
        N = lut.shape[2]
        hn = (N-1)/2
        
        streamlines = np.zeros((numberOfFibers, maxLength, dim), dtype=np.float64)
        streamlines_tangents = np.zeros((numberOfFibers, maxLength, dim), dtype=np.float64)
        streamlines_nearestp = np.zeros((numberOfFibers, maxLength), dtype=np.int)
        streamline_scores = np.zeros((numberOfFibers, maxLength), dtype=np.float64)
        
        # copy python streamlines into c++ buffer
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]):
                for dim in range(3):
                    streamlines[lineId, pointId, dim] = py_streamlines[lineId][pointId][dim]
        
        # compute tangents
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                tangent = np.subtract(streamlines[lineId, pointId+1], streamlines[lineId, pointId])
                streamlines_tangents[lineId, pointId] = np.divide(tangent, np.sqrt(np.dot(tangent, tangent)))
        
         # estimate which kernel LUT index corresponds to angles
        tree = KDTree(kernel.get_orientations())
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                streamlines_nearestp[lineId, pointId] = tree.query(streamlines[lineId])[1]
        
        # compute fiber LFBC measures
        for lineId in range(numberOfFibers):
            for pointId in range(streamlines_length[lineId]-1):
                score = 0.0
                # for lineId2 in range(numberOfFibers):
                    # if lineId == lineId2:
                        # continue
                    # for pointId2 in range(streamlines_length[lineId2]-1):
                        # displacement = np.subtract(streamlines[lineId, pointId], streamlines[lineId2, pointId2])
                        # xd = int(round(displacement[0]))
                        # yd = int(round(displacement[1]))
                        # zd = int(round(displacement[2]))
                        
                        # if xd > hn or -xd > hn or yd > hn or -yd > hn or \
                            # zd > hn or -zd > hn:
                            # continue
                            
                        # print xd
                        # print yd
                        # print zd
                        
                        # score += lut[streamlines_nearestp[lineId][pointId], 
                                    # streamlines_nearestp[lineId2][pointId2], 
                                    # hn+xd, 
                                    # hn+yd, 
                                    # hn+zd]  # ang_v, ang_r, x, y, z
                                
                # streamline_scores[lineId, pointId] = score


# def FBCMeasures(streamlines, kernel):
    
    # performFBC(streamlines, kernel)
    
    
    
# def performFBC(streamlines, kernel):

    # streamlines = streamlines[1:5]
    
    # numberOfFibers = len(streamlines)
    # minDistSquared = 15*15
    
    # # compute fiber tangents
    # streamline_tangents = []
    # for streamline in streamlines:
        # tangents = []
        # for n in range(len(streamline)-1): # skip last point
            # t = streamline[n+1] - streamline[n]
            # tangents.append( t/sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]) )
        # streamline_tangents.append(tangents)
    
    # # estimate which kernel LUT index corresponds to angles
    # tree = KDTree(kernel.get_orientations())
    # streamline_nearestp = []
    # for tangent in streamline_tangents:
        # nearestp = []
        # for n in tangent:
            # nearestp.append(tree.query(n)[1])
        # streamline_nearestp.append(nearestp)
    # print streamline_nearestp[0][1:10]
    
    # # calculate lengths
    # streamline_lengths = []
    # for tangent in streamline_tangents:
        # streamline_lengths.append(len(tangent))
        
    # lut = kernel.get_lookup_table()
    # N = lut.shape[2]
    # hn = (N-1)/2
    
    # # compute fiber scores
    # streamline_scores = []
    # for tangent in streamline_tangents:
        # streamline_scores.append([0]*len(tangent))
    # #pdb.set_trace()
    
    # # loop through all fibers
    # for lineId in range(numberOfFibers):
    
        # numberOfFiberPoints = streamline_lengths[lineId]
        
        # # loop through each point of the fiber
        # for pointId in range(numberOfFiberPoints):     

            # score = 0.0
        
            # # loop through all other fibers
            # for lineId2 in range(numberOfFibers):
                
                # if lineId == lineId2:
                    # continue
                    
                # numberOfFiberPoints2 = streamline_lengths[lineId2]
                
                # # compare to the points of all other fibers
                # for pointId2 in range(numberOfFiberPoints2):
                    
                    # displacement = streamlines[lineId][pointId] - streamlines[lineId2][pointId2]
                    # xd = int(round(displacement[0]))
                    # yd = int(round(displacement[1]))
                    # zd = int(round(displacement[2]))
                    
                    
                    
                    # # filter far-away points (also add angle?)
                    # #if displacement[0]*displacement[0] + \
                    # #    displacement[1]*displacement[1] + \
                    # #    displacement[2]*displacement[2] > minDistSquared:
                    # #    continue
                    
                    # if xd > hn or \
                        # -xd > hn or \
                        # yd > hn or \
                        # -yd > hn or \
                        # zd > hn or \
                        # -zd > hn:
                        # continue
                    
                    # score += lut[streamline_nearestp[lineId][pointId], streamline_nearestp[lineId2][pointId2], hn+xd, hn+yd, hn+zd]  # ang_v, ang_r, x, y, z
                    
                    # #pdb.set_trace()
                
            # streamline_scores[lineId][pointId] = score
                
    # print streamline_scores
    

    # # compute RFBC for each fiber
    
    
    
    
    
    
    
    
    
    
    