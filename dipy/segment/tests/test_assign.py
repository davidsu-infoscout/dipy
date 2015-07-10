# Use ICM to segment T1 image with MRF

import numpy as np
import nibabel as nib

from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.segment.rois_stats import seg_stats
from dipy.segment.energy_mrf import total_energy

# dname = '/Users/jvillalo/Documents/GSoC_2015/Code/Data/'
dname = '/home/eleftherios/Dropbox/DIPY_GSoC_2015/'

img = nib.load(dname + '3587_BL_T1_to_MNI_Linear_6p.nii.gz')
dataimg = img.get_data()

mask_image = nib.load(dname + '3587_mask.nii.gz')
datamask = mask_image.get_data()

masked_img = applymask(dataimg, datamask)
print('masked_img.shape (%d, %d, %d)' % masked_img.shape)
shape = masked_img.shape[:3]


def cluster_t1(data, mask, dist_thr=20.):

    labels = np.zeros_like(data)


    cluster_centroids = [0]
    cluster_sizes = [1]
    cluster_labels = [0]

    for idx in ndindex(data.shape):

        if mask[idx] > 0:
            print(idx)
            close_centroid = 0

            for (centroid, size, label) in zip(cluster_centroids, cluster_sizes, cluster_labels):

                if np.abs(data[idx] - centroid / np.float(size)) <= dist_thr:

                    labels[idx] = label
                    cluster_centroids[label] += data[idx]
                    cluster_sizes[label] += 1
                    close_centroid = 1

            if close_centroid == 0 :

                label = len(cluster_labels)
                cluster_labels.append(label)

                labels[idx] = label
                cluster_centroids.append(data[idx])
                cluster_sizes.append(1)


        # print(cluster_sizes)
        # print(cluster_labels)
        # print(cluster_centroids)

    return labels, cluster_centroids, cluster_sizes, cluster_labels

data = masked_img[:, :, 90:92]
mask = datamask[:, :, 90:92]
labels, centroids, sizes, label_indices = cluster_t1(data, mask, 30.)

figure()
imshow(data[:, :, 1], cmap='rainbow')
figure()
imshow(labels[:, :, 1], cmap='rainbow')
figure()
centroids = np.array(centroids)/np.array(sizes)
plot(centroids)
figure()
plot(np.sort(centroids))

#for i in range(labels.max()):
#    figure()
#    imshow(labels[:, :, 1]==i)
