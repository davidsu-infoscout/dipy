# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from metricspeed cimport Metric, Streamline, Features
from metricspeed cimport Shape, tuple2shape, shape2tuple

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

DEF biggest_float = 3.4028235e+38  # np.finfo('f4').max
DEF biggest_int = 2147483647  # np.iinfo('i4').max

cdef struct Centroid:
    Features features
    int size

cdef class identity:
    def __getitem__(self, idx):
        return idx


cdef class Cluster:
    cdef int _id
    cdef ClusterMap _cluster_map

    def __init__(Cluster self, ClusterMap cluster_map, int id):
        self._id = id
        self._cluster_map = cluster_map
        self.id  # Implicitly test if id and cluster_map are valid!

    property cluster_map:
        def __get__(self):
            if self._cluster_map is None:
                raise ValueError("This cluster is not linked with a cluster map.")

            return self._cluster_map

    property id:
        def __get__(self):
            if self._id < 0 or self._id >= len(self.cluster_map):
                raise ValueError("Cluster id {0} can't not be found in linked cluster map.".format(self._id))

            return self._id

    property indices:
        def __get__(self):
            cdef ClusterMap cluster_map = <ClusterMap> self.cluster_map
            if cluster_map._clusters_size[self.id] == 0:
                return np.array([], dtype="int32")

            return np.asarray(<int[:cluster_map._clusters_size[self.id]]> cluster_map._clusters_indices[self.id])

    def __len__(self):
        return (<ClusterMap>self.cluster_map)._clusters_size[self.id]

    def __getitem__(self, idx):
        cdef ClusterMap cluster_map = <ClusterMap> self.cluster_map
        cdef int* indices = cluster_map._clusters_indices[self.id]

        if isinstance(idx, int) or isinstance(idx, np.integer):
            if idx < -len(self) or len(self) <= idx:
                raise IndexError("Index out of bound: idx={0}".format(idx))

            if idx < 0:
                idx += len(self)

            return cluster_map.refdata[indices[idx]]
        elif type(idx) is slice:
            return [cluster_map.refdata[indices[i]] for i in xrange(*idx.indices(len(self)))]

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

        return cluster_map.refdata[idx]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self.indices)) + "]"

    def __repr__(self):
        return "Cluster(" + str(self) + ")"

    def __richcmp__(self, other, op):
        # See http://docs.cython.org/src/userguide/special_methods.html#rich-comparisons
        if op == 2:
            return isinstance(other, Cluster) and self.id == other.id and self.cluster_map == other.cluster_map
        elif op == 3:
            return not self == other
        else:
            return NotImplemented("Cluster does not support this type of comparison!")

    def add(self, *indices):
        for id_data in indices:
            self.cluster_map.add(self.id, id_data)


cdef class ClusterCentroid(Cluster):
    property centroid:
        def __get__(self):
            cdef ClusterMapCentroid cluster_map = <ClusterMapCentroid> self.cluster_map
            shape = shape2tuple(cluster_map._features_shape)
            return np.asarray(cluster_map._centroids[self.id].features)

    def add(self, int id_features, features):
        self.cluster_map.add(self.id, id_features, features)


cdef class ClusterMap:
    cdef object _cluster_class
    cdef object refdata
    cdef int _nb_clusters
    cdef int** _clusters_indices
    cdef int* _clusters_size

    def __init__(ClusterMap self, refdata=identity()):
        self._nb_clusters = 0
        self._clusters_indices = NULL
        self._clusters_size = NULL
        self.refdata = refdata
        self._cluster_class = Cluster

    property refdata:
        def __get__(self):
            return self.refdata
        def __set__(self, refdata):
            self.refdata = refdata

    property clusters:
        def __get__(self):
            return list(self)

    def __len__(self):
        return self.c_size()

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.integer):
            if idx < -len(self) or len(self) <= idx:
                raise IndexError("Index out of bound: idx={0}".format(idx))

            if idx < 0:
                idx += len(self)

            return self._cluster_class(self, idx)
        elif type(idx) is slice:
            return [self._cluster_class(self, i) for i in xrange(*idx.indices(len(self)))]

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self)) + "]"

    def __repr__(self):
        return "ClusterMap(" + str(self) + ")"

    def __dealloc__(ClusterMap self):
        for i in range(self._nb_clusters):
            free(self._clusters_indices[i])
            self._clusters_indices[i] = NULL

        free(self._clusters_indices)
        self._clusters_indices = NULL
        free(self._clusters_size)
        self._clusters_size = NULL

    cdef void c_add(ClusterMap self, int id_cluster, int id_features) nogil except *:
        # Keep streamline's index in the given cluster
        cdef int C = self._clusters_size[id_cluster]
        self._clusters_indices[id_cluster] = <int*> realloc(self._clusters_indices[id_cluster], (C+1)*sizeof(int))
        self._clusters_indices[id_cluster][C] = id_features
        self._clusters_size[id_cluster] += 1

    cdef int c_create_cluster(ClusterMap self) nogil except -1:
        self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self._clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters+1)*sizeof(int))
        self._clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1

    cdef int c_size(ClusterMap self) nogil:
        return self._nb_clusters

    def create_cluster(ClusterMap self):
        id_cluster = self.c_create_cluster()
        return self[id_cluster]

    def add(self, int id_cluster, int id_features):
        if id_cluster >= len(self):
            raise IndexError("Index out of bound: id_cluster={0}".format(id_cluster))

        self.c_add(id_cluster, id_features)


cdef class ClusterMapCentroid(ClusterMap):
    cdef Centroid* _centroids
    cdef Shape _features_shape

    def __init__(ClusterMapCentroid self, feature_shape, *args, **kwargs):
        ClusterMap.__init__(self, *args, **kwargs)
        if isinstance(feature_shape, int):
            feature_shape = (1, feature_shape)

        if not isinstance(feature_shape, tuple):
            raise ValueError("'feature_shape' must be a tuple or a int.")

        self._features_shape = tuple2shape(feature_shape)

        self._centroids = NULL
        self._cluster_class = ClusterCentroid

    property centroids:
        def __get__(self):
            shape = shape2tuple(self._features_shape)
            return [np.asarray(self.c_get_centroid(i).features) for i in range(self._nb_clusters)]

    def __dealloc__(ClusterMapCentroid self):
        # __dealloc__ method of the superclass is automatically called.
        # see: http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        for i in range(self._nb_clusters):
            free(&(self._centroids[i].features[0, 0]))

        free(self._centroids)
        self._centroids = NULL

    cdef void c_add(ClusterMapCentroid self, int id_cluster, int id_features, Features features=None) nogil except *:
        cdef Features centroid = self._centroids[id_cluster].features
        cdef int C = self._clusters_size[id_cluster]

        cdef int N = centroid.shape[0], D = centroid.shape[1]
        for n in range(N):
            for d in range(D):
                centroid[n, d] = ((centroid[n, d] * C) + features[n, d]) / (C+1)

        ClusterMap.c_add(self, id_cluster, id_features)

    cdef int c_create_cluster(ClusterMapCentroid self) nogil except -1:
        self._centroids = <Centroid*> realloc(self._centroids, (self._nb_clusters+1)*sizeof(Centroid))
        memset(&self._centroids[self._nb_clusters], 0, sizeof(Centroid))  # Zero-initialize the new centroid

        with gil:
            self._centroids[self._nb_clusters].features = <float[:self._features_shape.dims[0], :self._features_shape.dims[1]]> calloc(self._features_shape.size, sizeof(float))

        return ClusterMap.c_create_cluster(self)

    cdef Centroid* c_get_centroid(ClusterMapCentroid self, int idx) nogil:
        return &self._centroids[idx]

    def create_cluster(ClusterMapCentroid self):
        id_cluster = self.c_create_cluster()
        return self[id_cluster]

    def add(self, int id_cluster, int id_features, features):
        if id_cluster >= len(self):
            raise IndexError("Index out of bound: id_cluster={0}".format(id_cluster))

        if shape2tuple(self._features_shape) != features.shape:
            raise ValueError("The shape of the centroid and the features to add must be the same!")

        self.c_add(id_cluster, id_features, features)


cpdef quickbundles(streamlines, Metric metric, float threshold=10., int max_nb_clusters=biggest_int, ordering=None):
    if ordering is None:
        ordering = np.arange(len(streamlines), dtype="int32")

    # Threshold of np.inf is not supported, set it to 'biggest_float'
    threshold = min(threshold, biggest_float)
    # Threshold of -np.inf is not supported, set it to 0
    threshold = max(threshold, 0)

    dtype = streamlines[0].dtype
    features_shape = metric.infer_features_shape(streamlines[0])
    cdef:
        int idx
        ClusterMapCentroid clusters = ClusterMapCentroid(features_shape)
        Features features_s_i = np.empty(features_shape, dtype=dtype)
        Features features_s_i_flip = np.empty(features_shape, dtype=dtype)

    for idx in ordering:
        _quickbundles(streamlines[idx], idx, metric, clusters, features_s_i, features_s_i_flip, threshold, max_nb_clusters)

    return clusters


cdef void _quickbundles(Streamline s_i, int streamline_idx, Metric metric, ClusterMapCentroid clusters, Features features_s_i, Features features_s_i_flip, float threshold=10, int max_nb_clusters=biggest_int) nogil except *:
    cdef:
        Centroid* centroid
        int closest_cluster
        float dist, dist_min, dist_min_flip
        Features features_to_add = features_s_i

    # Find closest cluster to s_i
    metric.c_extract_features(s_i, features_s_i)
    dist_min = biggest_float
    for k in range(clusters.c_size()):
        #centroid = clusters.c_get_centroid(k)
        #dist = metric.c_dist(centroid.features, features_s_i)
        dist = metric.c_dist(clusters._centroids[k].features, features_s_i)

        # Keep track of the closest cluster
        if dist < dist_min:
            dist_min = dist
            closest_cluster = k

    # Find closest cluster to s_i_flip if metric is not order invariant
    if not metric.is_order_invariant:
        dist_min_flip = dist_min  # Initialize to the min distance not flipped.
        metric.c_extract_features(s_i[::-1], features_s_i_flip)
        for k in range(clusters.c_size()):
            #centroid = clusters.c_get_centroid(k)
            #dist = metric.c_dist(centroid.features, features_s_i_flip)
            dist = metric.c_dist(clusters._centroids[k].features, features_s_i_flip)

            # Keep track of the closest cluster
            if dist < dist_min_flip:
                dist_min_flip = dist
                closest_cluster = k

        # If we found a lower distance using a flipped streamline,
        #  add the flipped version instead
        if dist_min_flip < dist_min:
            dist_min = dist_min_flip
            features_to_add = features_s_i_flip

    # Check if distance with the closest cluster is below some threshold
    # or if we already have the maximum number of clusters.
    # If the former or the latter is true, assign streamline to its closest cluster
    # otherwise create a new cluster and assign the streamline to it.
    if not (dist_min < threshold or clusters.c_size() >= max_nb_clusters):
        closest_cluster = clusters.c_create_cluster()

    clusters.c_add(closest_cluster, streamline_idx, features_to_add)