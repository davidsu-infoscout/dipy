from os.path import join as pjoin
import numpy as np
import nibabel as nib
from time import time
from dipy.align.imaffine import (align_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)
from dipy.viz import window, actor, widget


def show_slices(data, affine=None, factor=1.5, size=(900, 900), show_slider=True, border = 10):

    renderer = window.renderer()

    def im_actor(data, affine, factor):
        mean, std = data[data > 0].mean(), data[data > 0].std()
        value_range = (0, mean + factor * std)
        image_actor = actor.slice(data, affine)
        return image_actor

    if isinstance(data, list):
        pass
    else:
        data = [data]
        affine = [affine]

    im_actors = []
    for (i, d) in enumerate(data):
        ima = im_actor(d, affine[i], factor)
        ima.set_position(i * ima.shape[0] + border, 0, 0)
        im_actors.append(ima)
        renderer.add(ima)

    renderer.projection('parallel')

    show_m = window.ShowManager(renderer, size=(1200, 900))
    show_m.initialize()

    def change_slice(obj, event):
        z = int(np.round(obj.get_value()))
        for im_actor in im_actors:
            im_actor.display(None, None, z)

    slider = widget.slider(show_m.iren, show_m.ren,
                           callback=change_slice,
                           min_value=0,
                           max_value=im_actors[0].shape[2] - 1,
                           value=im_actors[0].shape[2] / 2,
                           label="Move slice",
                           right_normalized_pos=(.98, 0.6),
                           size=(120, 0), label_format="%0.lf",
                           color=(1., 1., 1.),
                           selected_color=(0.86, 0.33, 1.))

    show_m.render()
    show_m.start()


def flowless_bet(fname, atlas_fname):

    pass



def affine_registration(static_fname, moving_fname,
                        moved_fname, moved_mat_fname):

    static_img = nib.load(static_fname)

    static = static_img.get_data()
    static_grid2world = static_img.get_affine()

    moving_img = nib.load(moving_fname)

    moving = moving_img.get_data()
    moving_grid2world = moving_img.get_affine()

    print('Transform centers of mass')
    c_of_mass = align_centers_of_mass(static, static_grid2world,
                                      moving, moving_grid2world)

    transformed = c_of_mass.transform(moving)

    print('Use Mutual Information as cost function')
    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 100]

    sigmas = [3.0, 1.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    print('Translation only')
    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=starting_affine)

    transformed = translation.transform(moving)

    print('Rigid only')
    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(static, moving, transform, params0,
                            static_grid2world, moving_grid2world,
                            starting_affine=starting_affine)

    print('Affine registration')
    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine
    affine = affreg.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=starting_affine)

    transformed = affine.transform(moving)

    nib.save(nib.Nifti1Image(transformed, static_grid2world), moved_fname)

    np.savetxt(moved_mat_fname, affine.affine)


disp = False
linear_reg = False


print('Calculate MTR')

dname = '/home/eleftherios/Data/badgiu7012/'
subj_id = 'badgiu7012'

img_mt1 = nib.load(pjoin(dname, subj_id + '_d1_mri.nii'))
img_mt2 = nib.load(pjoin(dname, subj_id + '_d2_mri.nii'))

print(img_mt1.affine)
print(img_mt2.affine)

mtr = np.abs(img_mt1.get_data() - img_mt2.get_data())
nib.save(nib.Nifti1Image(mtr, img_mt1.affine),
         pjoin(dname, subj_id + '_mtr.nii'))

print('MTR shape is ', mtr.shape)

if disp:
    show_slices(img_mt1.get_data(), None, 3)
    show_slices(img_mt2.get_data(), None, 3)
    show_slices(mtr, None, 3)


if linear_reg:
    print('Registration of MT1 to T1')
    t1 = time()
    affine_registration(pjoin(dname, subj_id + '_VOLISOTR_mri.nii'),
                        pjoin(dname, subj_id + '_d1_mri.nii'),
                        pjoin(dname, subj_id + '_d1_mri_to_T1.nii'),
                        pjoin(dname, subj_id + '_d1_mri_to_T1.txt'))

    print('Finished in %.2f minutes' % (time() - t1) / 60.)

img_mt1_to_t1 = nib.load(pjoin(dname, subj_id + '_d1_mri_to_T1.nii'))
img_t1 = nib.load(pjoin(dname, subj_id + '_VOLISOTR_mri.nii'))

show_slices([img_mt1_to_t1.get_data(), img_t1.get_data(), np.abs(img_mt1_to_t1.get_data() - img_t1.get_data())],
            [img_mt1_to_t1.get_affine(), img_t1.get_affine(), img_t1.get_affine()], factor=1.)
