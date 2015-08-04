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
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.segment.mask import median_otsu


def show_slices(data, affine=None, factors=1.5, mins=50, size=(900, 900),
                show_slider=True, border=10):

    renderer = window.renderer()

    def im_actor(data, affine, factor, min_):
        mean, std = data[data > min_].mean(), data[data > min_].std()
        value_range = (min_, mean + factor * std)

        image_actor = actor.slice(data, affine, value_range)
        return image_actor

    if isinstance(data, list):
        pass
    else:
        data = [data]
        affine = [affine]
        factors = [factors]
        mins = [mins]

    im_actors = []
    for (i, d) in enumerate(data):
        ima = im_actor(d, affine[i], factors[i], mins[i])
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

    global wsize
    wsize = renderer.GetSize()

    def win_callback(obj, event):
        global wsize
        if wsize != obj.GetSize():

            slider.place(renderer)
            wsize = obj.GetSize()

    show_m.initialize()
    show_m.add_window_callback(win_callback)

    show_m.render()
    show_m.start()


def flowless_bet(fname, fname_bet, fname_den, n_coil=1,
                 sigma_factor=1.,
                 median_radius=4, numpass=4,
                 autocrop=False, vol_idx=None, dilate=None):

    img = nib.load(fname)
    data = img.get_data()

    print('Sigma estimation')
    sigma = estimate_sigma(data, N=n_coil)

    print(sigma)
    print('Non-local means')
    den = nlmeans(data, sigma=sigma_factor * sigma)

    nib.save(nib.Nifti1Image(den, img.get_affine()), fname_den)

    print('Median Otsu')
    masked, mask = median_otsu(den, median_radius=median_radius,
                               numpass=numpass, autocrop=autocrop,
                               vol_idx=vol_idx, dilate=dilate)

    nib.save(nib.Nifti1Image(masked, img.get_affine()), fname_bet)


def flowless_bet_atlas(fname, fatlas_moved, fatlas_moved_mat):

    # /home/eleftherios/Data/mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii
    # /home/eleftherios/Data/mni_icbm152_nlin_asym_09c/mni_icbm152_csf_tal_nlin_asym_09c.nii
    # /home/eleftherios/Data/mni_icbm152_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c.nii
    # /home/eleftherios/Data/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii
    # /home/eleftherios/Data/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii

    dname_atlas = '/home/eleftherios/Data/mni_icbm152_nlin_asym_09c/'
    fatlas = pjoin(dname_atlas, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')

    print('Registering atlas to ...')
    print(fname)
    affine_registration(fname, fatlas, fatlas_moved, fatlas_moved_mat,
                        level_iters=[1000, 100, 10],
                        sigmas=[3.0, 1.0, 0.0],
                        factors=[4, 2, 1])


def affine_registration(static_fname, moving_fname,
                        moved_fname, moved_mat_fname,
                        level_iters=[10000, 1000, 100],
                        sigmas=[3.0, 1.0, 0.0],
                        factors=[4, 2, 1], other_fnames=None):

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
enable_linear_reg = False
enable_flowless_bet = False
enable_flowless_bet_atlas = True

print('Calculate MTR')

dname = '/home/eleftherios/Data/badgiu7012/'
subj_id = 'badgiu7012'

f_mt1 = pjoin(dname, subj_id + '_d1_mri.nii')
f_mt2 = pjoin(dname, subj_id + '_d2_mri.nii')
f_mtr = pjoin(dname, subj_id + '_mtr.nii')
f_t1 = pjoin(dname, subj_id + '_VOLISOTR_mri.nii')
f_t1_den = pjoin(dname, subj_id + '_VOLISOTR_mri_den.nii')
f_t1_bet = pjoin(dname, subj_id + '_VOLISOTR_mri_bet.nii')
f_mt1_to_t1 = pjoin(dname, subj_id + '_d1_mri_to_T1.nii')
f_mt1_to_t1_mat = pjoin(dname, subj_id + '_d1_mri_to_T1.txt')
f_atlas_to_t1 = pjoin(dname, subj_id + '_atlas_to_T1.nii')
f_atlas_to_t1_mat = pjoin(dname, subj_id + '_atlas_to_T1.txt')


img_mt1 = nib.load(f_mt1)
img_mt2 = nib.load(f_mt2)

mtr = np.abs(img_mt1.get_data() - img_mt2.get_data())
nib.save(nib.Nifti1Image(mtr, img_mt1.get_affine()), f_mtr)

if enable_flowless_bet:
    flowless_bet(f_t1, f_t1_bet, f_t1_den, sigma_factor=5,
                 median_radius=4, numpass=10)

    img_t1_bet = nib.load(f_t1_bet)
    img_t1 = nib.load(f_t1)
    img_t1_den = nib.load(f_t1_den)

    show_slices([img_t1.get_data(), img_t1_bet.get_data(), img_t1_den.get_data()],
     [img_t1.get_affine(), img_t1_bet.get_affine(), img_t1_den.get_affine()],
     factors=[2., 2., 2.], mins=[50, 50., 50.])

if enable_flowless_bet_atlas:

    flowless_bet_atlas(f_t1, f_atlas_to_t1, f_atlas_to_t1_mat)

print('MTR shape is ', mtr.shape)

if disp:
    show_slices(img_mt1.get_data(), None, 3)
    show_slices(img_mt2.get_data(), None, 3)
    show_slices(mtr, None, 3)


if enable_linear_reg:
    print('Registration of MT1 to T1')
    t1 = time()
    affine_registration(f_t1,
                        f_mt1,
                        f_mt1_to_t1,
                        f_mt1_to_t1_mat)

    print('Finished in %.2f minutes' % (time() - t1) / 60.)

    img_mt1_to_t1 = nib.load(f_mt1_to_t1)

    show_slices([img_mt1_to_t1.get_data(), img_t1.get_data()],
                [img_mt1_to_t1.get_affine(), img_t1.get_affine()],
                factors=[2., 2.], mins=[50, 50.])
