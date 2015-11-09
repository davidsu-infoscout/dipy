# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

import numpy as np
cimport numpy as cnp

from dipy.segment.clustering import ClusterCentroid, ClusterMapCentroid

from libc.math cimport fabs
from cythonutils cimport Data2D, Shape, shape2tuple, tuple2shape, same_shape

cdef extern from "math.h" nogil:
    double fabs(double x)

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t elsize)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

DTYPE = np.float32
DEF BIGGEST_DOUBLE = 1.7976931348623157e+308  # np.finfo('f8').max
DEF BIGGEST_INT = 2147483647  # np.iinfo('i4').max
DEF BIGGEST_FLOAT = 3.4028235e+38  # np.finfo('f4').max
DEF SMALLEST_FLOAT = -3.4028235e+38  # np.finfo('f4').max

THRESHOLD_MULTIPLIER = 2.

cdef print_node(CentroidNode* node, prepend=""):
    if node == NULL:
        return ""

    txt = "{}".format(np.asarray(node.centroid).tolist())
    txt += " {" + ",".join(map(str, np.asarray(<int[:node.size]> node.indices))) + "}"
    txt += " children({})".format(node.nb_children)
    txt += " count({})".format(node.size)
    txt += " thres({})".format(node.threshold)
    txt += "\n"

    cdef int i
    for i in range(node.nb_children):
        txt += prepend
        if i == node.nb_children-1:
            # Last child
            txt += "`-- " + print_node(node.children[i], prepend + "    ")
        else:
            txt += "|-- " + print_node(node.children[i], prepend + "|   ")

    return txt


cdef void aabb_creation(Data2D streamline, float* aabb) nogil:
    """ Creates AABB enveloping the given streamline.

        Notes
        -----
        This currently assumes streamline is made of 3D points.
    """
    cdef:
        int N = streamline.shape[0], D = streamline.shape[1]
        int n, d
        float min_[3]
        float max_[3]

    for d in range(D):
        min_[d] = BIGGEST_FLOAT
        max_[d] = SMALLEST_FLOAT
        for n in range(N):

            if max_[d] < streamline[n, d]:
                max_[d] = streamline[n, d]

            if min_[d] > streamline[n, d]:
                min_[d] = streamline[n, d]

        aabb[d + 3] = (max_[d] - min_[d]) / 2.0 # radius
        aabb[d] = min_[d] + aabb[d + 3]  # center


cdef inline int aabb_overlap(float* aabb1, float* aabb2, float padding=0.) nogil:
    """ SIMD optimized AABB-AABB test

    Optimized by removing conditional branches
    """
    cdef:
        int x = fabs(aabb1[0] - aabb2[0]) <= (aabb1[3] + aabb2[3] + padding)
        int y = fabs(aabb1[1] - aabb2[1]) <= (aabb1[4] + aabb2[4] + padding)
        int z = fabs(aabb1[2] - aabb2[2]) <= (aabb1[5] + aabb2[5] + padding)

    return x & y & z;


cdef CentroidNode* create_empty_node(Shape centroid_shape, float threshold) nogil:
    # Important: because the CentroidNode structure contains an uninitialized memview,
    # we need to zero-initialize the allocated memory (calloc or via memset),
    # otherwise during assignment CPython will try to call _PYX_XDEC_MEMVIEW on it and segfault.
    cdef CentroidNode* node = <CentroidNode*> calloc(1, sizeof(CentroidNode))
    with gil:
        node.centroid = <float[:centroid_shape.dims[0], :centroid_shape.dims[1]]> calloc(centroid_shape.size, sizeof(float))
        #node.updated_centroid = <float[:centroid_shape.dims[0], :centroid_shape.dims[1]]> calloc(centroid_shape.size, sizeof(float))

    node.father = NULL
    node.children = NULL
    node.nb_children = 0
    node.aabb[0] = 0
    node.aabb[1] = 0
    node.aabb[2] = 0
    node.aabb[3] = BIGGEST_FLOAT
    node.aabb[4] = BIGGEST_FLOAT
    node.aabb[5] = BIGGEST_FLOAT
    node.threshold = threshold
    node.indices = NULL
    node.size = 0
    node.centroid_shape = centroid_shape
    return node


cdef class HierarchicalQuickBundles(object):
    cdef CentroidNode* root
    cdef Metric metric
    cdef Shape features_shape
    cdef Data2D features
    cdef Data2D features_flip
    cdef float min_threshold

    def __init__(self, features_shape, Metric metric, float min_threshold):
        self.features_shape = tuple2shape(features_shape)
        self.metric = metric
        self.min_threshold = min_threshold
        self.root = NULL

        self.features = np.empty(features_shape, dtype=DTYPE)
        self.features_flip = np.empty(features_shape, dtype=DTYPE)

    def __dealloc__(self):
        # Free indices, father and children
        # TODO: free children
        print "Deallocating..."
        if self.root != NULL:
            print "Freeing root..."
            free(self.root)
            self.root = NULL

    cdef hqb_add_to_node(self, CentroidNode* node, Streamline* streamline, int flip):

        cdef Data2D element = streamline.features
        cdef int C = node.size
        cdef cnp.npy_intp n, d

        if flip:
            element = streamline.features_flip

        cdef cnp.npy_intp N = node.centroid.shape[0], D = node.centroid.shape[1]
        for n in range(N):
            for d in range(D):
                node.centroid[n, d] = ((node.centroid[n, d] * C) + element[n, d]) / (C+1)

        node.indices = <int*> realloc(node.indices, (C+1)*sizeof(int))
        node.indices[C] = streamline.idx
        node.size += 1

    cdef int hqb_add_child(self, CentroidNode* node, Streamline* streamline):
        #print "Adding child..."
        # Create new child.
        cdef CentroidNode* child = create_empty_node(node.centroid_shape, node.threshold/THRESHOLD_MULTIPLIER)
        #hqb_add_to_node(child, streamline, False)

        # Add new child.
        child.father = node
        node.children = <CentroidNode**> realloc(node.children, (node.nb_children+1)*sizeof(CentroidNode*))
        node.children[node.nb_children] = child
        node.nb_children += 1
        #print "Added child..."
        return node.nb_children-1

    cdef void hqb_update(self, CentroidNode* node):
        cdef int i, j

        # Update indices
        node.size = 0
        for i in range(node.nb_children):
            node.size += node.children[i].size

        node.indices = <int*> realloc(node.indices, node.size*sizeof(int))

        cdef int cpt = 0
        for i in range(node.nb_children):
            for j in range(node.children[i].size):
                node.indices[cpt] = node.children[i].indices[j]
                cpt += 1

        # Update centroid
        cdef int n, d
        cdef cnp.npy_intp N = node.centroid.shape[0], D = node.centroid.shape[1]
        for n in range(N):
            for d in range(D):
                node.centroid[n, d] = 0
                for i in range(node.nb_children):
                    node.centroid[n, d] += node.children[i].centroid[n, d]

                node.centroid[n, d] /= node.nb_children

        # Update AABB
        aabb_creation(node.centroid, node.aabb)

    cdef void hqb_insert(self, CentroidNode* node, Streamline* streamline):
        cdef:
            float dist, dist_flip
            cnp.npy_intp k
            NearestCluster nearest_cluster

        #print "Inserting..."
        if node.threshold <= self.min_threshold:
            #print "Hit the bottom!"
            self.hqb_add_to_node(node, streamline, False)
            return

        nearest_cluster.id = -1
        nearest_cluster.dist = BIGGEST_DOUBLE
        nearest_cluster.flip = 0

        #print "Comparing with centroids..."
        for k in range(node.nb_children):
            # Check streamline's aabb colides with the current child.
            if aabb_overlap(node.children[k].aabb, streamline.aabb, node.children[k].threshold):
                dist = self.metric.c_dist(node.children[k].centroid, streamline.features)

                # Keep track of the nearest cluster
                if dist < nearest_cluster.dist:
                    nearest_cluster.dist = dist
                    nearest_cluster.id = k
                    nearest_cluster.flip = 0

                dist_flip = self.metric.c_dist(node.children[k].centroid, streamline.features_flip)
                if dist_flip < nearest_cluster.dist:
                    nearest_cluster.dist = dist_flip
                    nearest_cluster.id = k
                    nearest_cluster.flip = 1

        if node.nb_children == 0 or nearest_cluster.dist > node.children[k].threshold:
            # No near cluster, create a new one.
            nearest_cluster.id = self.hqb_add_child(node, streamline)

        self.hqb_insert(node.children[nearest_cluster.id], streamline)
        self.hqb_update(node)

    cdef void _insert(self, Streamline* streamline):

        # Check if we have a root.
        if self.root == NULL:
            print "Building root...",
            self.root = create_empty_node(self.features_shape, self.min_threshold)

            # Add the streamline to the root.
            # Build indices
            self.root.size = 1
            self.root.indices = <int*> malloc(self.root.size*sizeof(int))
            self.root.indices[0] = streamline.idx

            # Build centroid
            for n in range(self.root.centroid.shape[0]):
                for d in range(self.root.centroid.shape[1]):
                    self.root.centroid[n, d] = streamline.features[n, d]

            # Build AABB
            aabb_creation(self.root.centroid, self.root.aabb)

            print "Done"
            return


        # Check if the streamline belongs to the tree currently spawn by self.root.
        cdef CentroidNode* new_root
        cdef double dist = BIGGEST_DOUBLE
        cdef double dist_flip

        if aabb_overlap(self.root.aabb, streamline.aabb, self.root.threshold):
            dist = self.metric.c_dist(self.root.centroid, streamline.features)
            dist_flip = self.metric.c_dist(self.root.centroid, streamline.features_flip)

            if dist_flip < dist:
                dist = dist_flip

        #print "***AABB overlap", aabb_overlap(self.root.aabb, streamline.aabb)
        #print "***Dist", dist
        if dist <= self.root.threshold:
            # The streamline belong in this tree, let's add it.
            #print "Inserting into current root..."
            self.hqb_insert(self.root, streamline)
        else:
            # The streamline does not belong in this tree, we need to expand it.
            # Build a new root.
            print "Building new root with thres: {}...".format(self.root.threshold*THRESHOLD_MULTIPLIER)
            new_root = create_empty_node(self.root.centroid_shape,
                                         self.root.threshold*THRESHOLD_MULTIPLIER)

            # Copy indices
            #print "Copying indices..."
            new_root.size = self.root.size
            new_root.indices = <int*> malloc(new_root.size*sizeof(int))
            for i in range(self.root.size):
                new_root.indices[i] = self.root.indices[i]

            # Copy centroid
            #print "Copying centroid..."
            for n in range(self.root.centroid.shape[0]):
                for d in range(self.root.centroid.shape[1]):
                    new_root.centroid[n, d] = self.root.centroid[n, d]

            # Create AABB with padding
            aabb_creation(new_root.centroid, new_root.aabb)

            # Add self.root as a child of new_root
            #print "Adding child..."
            new_root.children = <CentroidNode**> malloc((new_root.nb_children+1)*sizeof(CentroidNode*))
            new_root.children[0] = self.root
            new_root.nb_children += 1

            # Set old root's father.
            self.root.father = new_root

            # Change actual root.
            self.root = new_root

            # Insert the streamline in this new tree
            #print "Try reinserting..."
            self._insert(streamline)

    cpdef void insert(self, Data2D datum, int datum_idx):
        self.metric.feature.c_extract(datum, self.features)
        self.metric.feature.c_extract(datum[::-1], self.features_flip)

        # Important: because the CentroidNode structure contains an uninitialized memview,
        # we need to zero-initialize the allocated memory (calloc or via memset),
        # otherwise during assignment CPython will try to call _PYX_XDEC_MEMVIEW on it and segfault.
        cdef Streamline* streamline = <Streamline*> calloc(1, sizeof(Streamline))
        streamline.features = self.features
        streamline.features_flip = self.features_flip
        streamline.idx = datum_idx

        aabb_creation(streamline.features, streamline.aabb)

        self._insert(streamline)
        free(streamline)

    cdef void _expand(self, CentroidNode* node):
        pass

    def __str__(self):
        #print "Printing tree..."
        return print_node(self.root)


cdef class QuickBundlesX(object):

    def __init__(self, features_shape, levels_thresholds, Metric metric):
        self.features_shape = tuple2shape(features_shape)

        self.nb_levels = len(levels_thresholds)
        self.thresholds = <double*> malloc(self.nb_levels*sizeof(double))
        cdef int i
        for i in range(self.nb_levels):
            self.thresholds[i] = levels_thresholds[i]

        self.metric = metric
        self.root = NULL

        self.features = np.empty(features_shape, dtype=DTYPE)
        self.features_flip = np.empty(features_shape, dtype=DTYPE)

        self.level = None
        self.clusters = None

        self.stats.stats_per_layer = <QuickBundlesXStatsLayer*> calloc(self.nb_levels, sizeof(QuickBundlesXStatsLayer))
        self.stats.nb_mdf_calls_when_updating = 0

    def __dealloc__(self):
        print "Deallocating QuickBundlesX object..."
        self.traverse_postorder(self.root, self._dealloc_node)
        self.root = NULL

        if self.thresholds != NULL:
            free(self.thresholds)
            self.thresholds = NULL

        if self.stats.stats_per_layer != NULL:
            free(self.stats.stats_per_layer)
            self.stats.stats_per_layer = NULL

    cdef int _add_child_to(self, CentroidNode* node) nogil:
        # Create new child.
        cdef double threshold = 0.0
        if node.level+1 < self.nb_levels:
            threshold = self.thresholds[node.level+1]

        cdef CentroidNode* child = create_empty_node(self.features_shape, threshold)
        child.level = node.level+1

        # Add new child.
        child.father = node
        node.children = <CentroidNode**> realloc(node.children, (node.nb_children+1)*sizeof(CentroidNode*))
        node.children[node.nb_children] = child
        node.nb_children += 1

        return node.nb_children-1

    cdef void _add_streamline_to(self, CentroidNode* node, Streamline* streamline, int flip) nogil:
        cdef Data2D element = streamline.features
        cdef int C = node.size
        cdef cnp.npy_intp n, d

        if flip:
            element = streamline.features_flip

        # Update centroid
        cdef cnp.npy_intp N = node.centroid.shape[0], D = node.centroid.shape[1]
        for n in range(N):
            for d in range(D):
                node.centroid[n, d] = ((node.centroid[n, d] * C) + element[n, d]) / (C+1)

        # Update list of indices
        node.indices = <int*> realloc(node.indices, (C+1)*sizeof(int))
        node.indices[C] = streamline.idx
        node.size += 1

        # Update AABB
        aabb_creation(node.centroid, node.aabb)

    cdef void _insert_in(self, CentroidNode* node, Streamline* streamline, int flip) nogil:
        cdef:
            float dist, dist_flip
            cnp.npy_intp k
            NearestCluster nearest_cluster

        self._add_streamline_to(node, streamline, flip)

        if node.level == self.nb_levels:
            return

        nearest_cluster.id = -1
        nearest_cluster.dist = BIGGEST_DOUBLE
        nearest_cluster.flip = 0

        for k in range(node.nb_children):
            # Check streamline's aabb colides with the current child.
            self.stats.stats_per_layer[node.level].nb_aabb_calls += 1
            if aabb_overlap(node.children[k].aabb, streamline.aabb, node.threshold):
                self.stats.stats_per_layer[node.level].nb_mdf_calls += 1
                dist = self.metric.c_dist(node.children[k].centroid, streamline.features)

                # Keep track of the nearest cluster
                if dist < nearest_cluster.dist:
                    nearest_cluster.dist = dist
                    nearest_cluster.id = k
                    nearest_cluster.flip = 0

                self.stats.stats_per_layer[node.level].nb_mdf_calls += 1
                dist_flip = self.metric.c_dist(node.children[k].centroid, streamline.features_flip)
                if dist_flip < nearest_cluster.dist:
                    nearest_cluster.dist = dist_flip
                    nearest_cluster.id = k
                    nearest_cluster.flip = 1

        if nearest_cluster.dist > node.threshold:
            # No near cluster, create a new one.
            nearest_cluster.id = self._add_child_to(node)

        self._insert_in(node.children[nearest_cluster.id], streamline, nearest_cluster.flip)

    cdef void _insert(self, Streamline* streamline) nogil:
        # Create root if needed.
        if self.root == NULL:
            self.root = create_empty_node(self.features_shape, self.thresholds[0])

        self._insert_in(self.root, streamline, 0)

    cpdef void insert(self, Data2D datum, int datum_idx):
        self.metric.feature.c_extract(datum, self.features)
        self.metric.feature.c_extract(datum[::-1], self.features_flip)

        # Important: because the CentroidNode structure contains an uninitialized memview,
        # we need to zero-initialize the allocated memory (calloc or via memset),
        # otherwise during assignment CPython will try to call _PYX_XDEC_MEMVIEW on it and segfault.
        cdef Streamline* streamline = <Streamline*> calloc(1, sizeof(Streamline))
        streamline.features = self.features
        streamline.features_flip = self.features_flip
        streamline.idx = datum_idx

        aabb_creation(streamline.features, streamline.aabb)
        self._insert(streamline)
        free(streamline)

    def __str__(self):
        #print "Printing tree..."
        return print_node(self.root)

    cdef void traverse_postorder(self, CentroidNode* node, void (*visit)(QuickBundlesX, CentroidNode*)):
        cdef int i
        for i in range(node.nb_children):
            self.traverse_postorder(node.children[i], visit)

        visit(self, node)

    cdef void _dealloc_node(self, CentroidNode* node):
        free(&(node.centroid[0, 0]))
        node.centroid = None  # Necessary to decrease refcount

        if node.children != NULL:
            free(node.children)
            node.children = NULL

        free(node.indices)
        node.indices = NULL

        # No need to free node.father, only the current node.
        free(node)

    cdef void _fetch_level(self, CentroidNode* node):
        if node.level == self.level:
            cluster = ClusterCentroid(np.asarray(node.centroid))
            cluster.indices = np.asarray(<int[:node.size]> node.indices)
            self.clusters.add_cluster(cluster)

    def get_clusters(self, int level):
        self.clusters = ClusterMapCentroid()
        self.level = level

        self.traverse_postorder(self.root, self._fetch_level)
        return self.clusters

    def get_stats(self):
        stats_per_level = []
        for i in range(self.nb_levels):
            stats_per_level.append({'nb_mdf_calls': self.stats.stats_per_layer[i].nb_mdf_calls,
                                    'nb_aabb_calls': self.stats.stats_per_layer[i].nb_aabb_calls})

        stats = {'stats_per_level': stats_per_level,
                 'nb_mdf_calls_when_updating': self.stats.nb_mdf_calls_when_updating}

        return stats


cdef class Clusters:
    """ Provides Cython functionalities to interact with clustering outputs.

    This class allows to create clusters and assign elements to them.
    Assignements of a cluster are represented as a list of element indices.
    """
    def __init__(Clusters self):
        self._nb_clusters = 0
        self.clusters_indices = NULL
        self.clusters_size = NULL

    def __dealloc__(Clusters self):
        """ Deallocates memory created with `c_create_cluster` and `c_assign`. """
        for i in range(self._nb_clusters):
            free(self.clusters_indices[i])
            self.clusters_indices[i] = NULL

        free(self.clusters_indices)
        self.clusters_indices = NULL
        free(self.clusters_size)
        self.clusters_size = NULL

    cdef int c_size(Clusters self) nogil:
        """ Returns the number of clusters. """
        return self._nb_clusters

    cdef void c_assign(Clusters self, int id_cluster, int id_element, Data2D element) nogil except *:
        """ Assigns an element to a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster to which the element will be assigned.
        id_element : int
            Index of the element to assign.
        element : 2d array (float)
            Data of the element to assign.
        """
        cdef cnp.npy_intp C = self.clusters_size[id_cluster]
        self.clusters_indices[id_cluster] = <int*> realloc(self.clusters_indices[id_cluster], (C+1)*sizeof(int))
        self.clusters_indices[id_cluster][C] = id_element
        self.clusters_size[id_cluster] += 1

    cdef int c_create_cluster(Clusters self) nogil except -1:
        """ Creates a cluster and adds it at the end of the list.

        Returns
        -------
        id_cluster : int
            Index of the new cluster.
        """
        self.clusters_indices = <int**> realloc(self.clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self.clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self.clusters_size = <int*> realloc(self.clusters_size, (self._nb_clusters+1)*sizeof(int))
        self.clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1


cdef class ClustersCentroid(Clusters):
    """ Provides Cython functionalities to interact with clustering outputs
    having the notion of cluster's centroid.

    This class allows to create clusters, assign elements to them and
    update their centroid.

    Parameters
    ----------
    centroid_shape : int, tuple of int
        Information about the shape of the centroid.
    eps : float, optional
        Consider the centroid has not changed if the changes per dimension
        are less than this epsilon. (Default: 1e-6)
    """
    def __init__(ClustersCentroid self, centroid_shape, float eps=1e-6, *args, **kwargs):
        Clusters.__init__(self, *args, **kwargs)
        if isinstance(centroid_shape, int):
            centroid_shape = (1, centroid_shape)

        if not isinstance(centroid_shape, tuple):
            raise ValueError("'centroid_shape' must be a tuple or a int.")

        self._centroid_shape = tuple2shape(centroid_shape)
        # self.aabb
        self.centroids = NULL
        self._updated_centroids = NULL
        self.eps = eps

    def __dealloc__(ClustersCentroid self):
        """ Deallocates memory created with `c_create_cluster` and `c_assign`.

        Notes
        -----
        The `__dealloc__` method of the superclass is automatically called:
        http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        """
        cdef cnp.npy_intp i
        for i in range(self._nb_clusters):
            free(&(self.centroids[i].features[0, 0]))
            free(&(self._updated_centroids[i].features[0, 0]))
            self.centroids[i].features = None  # Necessary to decrease refcount
            self._updated_centroids[i].features = None  # Necessary to decrease refcount

        free(self.centroids)
        self.centroids = NULL
        free(self._updated_centroids)
        self._updated_centroids = NULL

    cdef void c_assign(ClustersCentroid self, int id_cluster, int id_element, Data2D element) nogil except *:
        """ Assigns an element to a cluster.

        In addition of keeping element's index, an updated version of the
        cluster's centroid is computed. The centroid is the average of all
        elements in a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster to which the element will be assigned.
        id_element : int
            Index of the element to assign.
        element : 2d array (float)
            Data of the element to assign.
        """
        cdef Data2D updated_centroid = self._updated_centroids[id_cluster].features
        cdef cnp.npy_intp C = self.clusters_size[id_cluster]
        cdef cnp.npy_intp n, d

        cdef cnp.npy_intp N = updated_centroid.shape[0], D = updated_centroid.shape[1]
        for n in range(N):
            for d in range(D):
                updated_centroid[n, d] = ((updated_centroid[n, d] * C) + element[n, d]) / (C+1)

        Clusters.c_assign(self, id_cluster, id_element, element)

    cdef int c_update(ClustersCentroid self, cnp.npy_intp id_cluster) nogil except -1:
        """ Update the centroid of a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster of which its centroid will be updated.

        Returns
        -------
        int
            Tells whether the centroid has changed or not, i.e. converged.
        """
        cdef Data2D centroid = self.centroids[id_cluster].features
        cdef Data2D updated_centroid = self._updated_centroids[id_cluster].features
        cdef cnp.npy_intp N = updated_centroid.shape[0], D = centroid.shape[1]
        cdef cnp.npy_intp n, d
        cdef int converged = 1

        for n in range(N):
            for d in range(D):
                converged &= fabs(centroid[n, d] - updated_centroid[n, d]) < self.eps
                centroid[n, d] = updated_centroid[n, d]

        #cdef float * aabb = &self.centroids[id_cluster].aabb[0]

        aabb_creation(centroid, self.centroids[id_cluster].aabb)

        return converged

    cdef int c_create_cluster(ClustersCentroid self) nogil except -1:
        """ Creates a cluster and adds it at the end of the list.

        Returns
        -------
        id_cluster : int
            Index of the new cluster.
        """
        self.centroids = <Centroid*> realloc(self.centroids, (self._nb_clusters+1)*sizeof(Centroid))
        # Zero-initialize the Centroid structure
        memset(&self.centroids[self._nb_clusters], 0, sizeof(Centroid))

        self._updated_centroids = <Centroid*> realloc(self._updated_centroids, (self._nb_clusters+1)*sizeof(Centroid))
        # Zero-initialize the new Centroid structure
        memset(&self._updated_centroids[self._nb_clusters], 0, sizeof(Centroid))

        with gil:
            self.centroids[self._nb_clusters].features = <float[:self._centroid_shape.dims[0], :self._centroid_shape.dims[1]]> calloc(self._centroid_shape.size, sizeof(float))
            self._updated_centroids[self._nb_clusters].features = <float[:self._centroid_shape.dims[0], :self._centroid_shape.dims[1]]> calloc(self._centroid_shape.size, sizeof(float))

        aabb_creation(self.centroids[self._nb_clusters].features, self.centroids[self._nb_clusters].aabb)

        return Clusters.c_create_cluster(self)


cdef class QuickBundles(object):
    def __init__(QuickBundles self, features_shape, Metric metric, double threshold,
                 int max_nb_clusters=BIGGEST_INT, int bvh=0):
        self.metric = metric
        self.features_shape = tuple2shape(features_shape)
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters
        self.clusters = ClustersCentroid(features_shape)
        self.features = np.empty(features_shape, dtype=DTYPE)
        self.features_flip = np.empty(features_shape, dtype=DTYPE)
        self.bvh = bvh

        self.stats.nb_mdf_calls = 0
        self.stats.nb_aabb_calls = 0

    cdef NearestCluster find_nearest_cluster(QuickBundles self, Data2D features) nogil except *:
        """ Finds the nearest cluster of a datum given its `features` vector.

        Parameters
        ----------
        features : 2D array
            Features of a datum.

        Returns
        -------
        `NearestCluster` object
            Nearest cluster to `features` according to the given metric.
        """
        cdef:
            cnp.npy_intp k
            double dist
            NearestCluster nearest_cluster
            float aabb[6]

        nearest_cluster.id = -1
        nearest_cluster.dist = BIGGEST_DOUBLE

        if self.bvh == 1:

            aabb_creation(features, &aabb[0])

            for k in range(self.clusters.c_size()):

                self.stats.nb_aabb_calls += 1
                if aabb_overlap(self.clusters.centroids[k].aabb, &aabb[0], self.threshold) == 1:

                    self.stats.nb_mdf_calls += 1
                    dist = self.metric.c_dist(self.clusters.centroids[k].features, features)

                    # Keep track of the nearest cluster
                    if dist < nearest_cluster.dist:
                        nearest_cluster.dist = dist
                        nearest_cluster.id = k

        if self.bvh == 0:

            for k in range(self.clusters.c_size()):

                self.stats.nb_mdf_calls += 1
                dist = self.metric.c_dist(self.clusters.centroids[k].features, features)

                # Keep track of the nearest cluster
                if dist < nearest_cluster.dist:
                    nearest_cluster.dist = dist
                    nearest_cluster.id = k


        return nearest_cluster

    cdef int assignment_step(QuickBundles self, Data2D datum, int datum_id) nogil except -1:
        """ Compute the assignment step of the QuickBundles algorithm.

        It will assign a datum to its closest cluster according to a given
        metric. If the distance between the datum and its closest cluster is
        greater than the specified threshold, a new cluster is created and the
        datum is assigned to it.

        Parameters
        ----------
        datum : 2D array
            The datum to assign.
        datum_id : int
            ID of the datum, usually its index.

        Returns
        -------
        int
            Index of the cluster the datum has been assigned to.
        """
        cdef:
            Data2D features_to_add = self.features
            NearestCluster nearest_cluster, nearest_cluster_flip
            Shape features_shape = self.metric.feature.c_infer_shape(datum)

        # Check if datum is compatible with the metric
        if not same_shape(features_shape, self.features_shape):
            with gil:
                raise ValueError("All features do not have the same shape! QuickBundles requires this to compute centroids!")

        # Check if datum is compatible with the metric
        if not self.metric.c_are_compatible(features_shape, self.features_shape):
            with gil:
                raise ValueError("Data features' shapes must be compatible according to the metric used!")

        # Find nearest cluster to datum
        self.metric.feature.c_extract(datum, self.features)
        nearest_cluster = self.find_nearest_cluster(self.features)

        # Find nearest cluster to s_i_flip if metric is not order invariant
        if not self.metric.feature.is_order_invariant:
            self.metric.feature.c_extract(datum[::-1], self.features_flip)
            nearest_cluster_flip = self.find_nearest_cluster(self.features_flip)

            # If we found a lower distance using a flipped datum,
            #  add the flipped version instead
            if nearest_cluster_flip.dist < nearest_cluster.dist:
                nearest_cluster.id = nearest_cluster_flip.id
                nearest_cluster.dist = nearest_cluster_flip.dist
                features_to_add = self.features_flip

        # Check if distance with the nearest cluster is below some threshold
        # or if we already have the maximum number of clusters.
        # If the former or the latter is true, assign datum to its nearest cluster
        # otherwise create a new cluster and assign the datum to it.
        if not (nearest_cluster.dist < self.threshold or self.clusters.c_size() >= self.max_nb_clusters):
            nearest_cluster.id = self.clusters.c_create_cluster()

        self.clusters.c_assign(nearest_cluster.id, datum_id, features_to_add)
        return nearest_cluster.id

    cdef void update_step(QuickBundles self, int cluster_id) nogil except *:
        """ Compute the update step of the QuickBundles algorithm.

        It will update the centroid of a cluster given its index.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster to update.

        """
        self.clusters.c_update(cluster_id)

    def get_stats(self):
        stats = {'nb_mdf_calls': self.stats.nb_mdf_calls,
                 'nb_aabb_calls': self.stats.nb_aabb_calls}

        return stats

def evaluate_aabbb_checks():
    cdef:
        Data2D feature1 = np.array([[1, 0, 0], [1, 1, 0], [1 + np.sqrt(2)/2., 1 + np.sqrt(2)/2., 0]], dtype='f4')
        Data2D feature2 = np.array([[1, 0, 0], [1, 1, 0], [1 + np.sqrt(2)/2., 1 + np.sqrt(2)/2., 0]], dtype='f4') + np.array([0.5, 0, 0], dtype='f4')
        float[6] aabb1
        float[6] aabb2
        int res

    aabb_creation(feature1, &aabb1[0])
    aabb_creation(feature2, &aabb2[0])

    res = aabb_overlap(&aabb1[0], &aabb2[0])

    return np.asarray(aabb1), np.asarray(aabb2), res
