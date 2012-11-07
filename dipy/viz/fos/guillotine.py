import numpy as np
from fos import Window, Scene
from fos.actor.slicer import Slicer
from pyglet.gl import *
from fos.coords import rotation_matrix, from_matvec
from fos import Init, Run


class Guillotine(Slicer):
    """ Head slicer actor

    Notes
    ------
    Coordinate Systems
    http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    http://www.slicer.org/slicerWiki/index.php/Coordinate_systems
    http://eeg.sourceforge.net/mri_orientation_notes.html

    """
    def __init__(self, name, data, affine, 
                    convention='LAS', look='anteriorz+'):

        data[np.isnan(data)] = 0
        data = np.interp(data, [data.min(), data.max()], [0, 255])
        data = data.astype(np.ubyte)
        
        if convention == 'LAS' and look == 'anteriorz+':
            axis = np.array([1, 0, 0.])
            theta = -90. 
            post_mat = from_matvec(rotation_matrix(axis, theta))
            axis = np.array([0, 0, 1.])
            theta = -90. 
            post_mat = np.dot(
                        from_matvec(rotation_matrix(axis, theta)), 
                        post_mat)
        
        super(Guillotine, self).__init__(name, data, affine, post_mat)

    def right2left(self, step):
        if self.i + step < self.I:
            self.slice_i(self.i + step)
        else:
            self.slice_i(self.I - 1)

    def left2right(self, step):
        if self.i - step >= 0:
            self.slice_i(self.i - step)
        else:
            self.slice_i(0)

    def inferior2superior(self, step):
        if self.k + step < self.K:
            self.slice_k(self.k + step)
        else:
            self.slice_k(self.K - 1)

    def superior2inferior(self, step):
        if self.k - step >= 0:
            self.slice_k(self.k - step)
        else:
            self.slice_k(0)

    def anterior2posterior(self, step):
        if self.j + step < self.J:
            self.slice_j(self.j + step)
        else:
            self.slice_j(self.J - 1)

    def posterior2anterior(self, step):
        if self.j - step >= 0:
            self.slice_j(self.j - step)
        else:
            self.slice_j(0)

    def reset_slices(self):
        self.slice_i(self.I / 2)
        self.slice_j(self.J / 2)
        self.slice_k(self.K / 2)

    def slices_ijk(self, i, j, k):
        self.slice_i(i)
        self.slice_j(j)
        self.slice_k(k)
        
    def show_coronal(self, bool=True):
        self.show_k = bool

    def show_axial(self, bool=True):
        self.show_i = bool

    def show_saggital(self, bool=True):
        self.show_j = bool

    def show_all(self, bool=True):
        self.show_i = bool
        self.show_j = bool
        self.show_k = bool


if __name__ == '__main__':

    import nibabel as nib    

    #dname='/home/eg309/Data/trento_processed/subj_03/MPRAGE_32/'
    #fname = dname + 'T1_flirt_out.nii.gz'
    #dname = '/home/eg309/Data/111104/subj_05/'
    #fname = dname + '101_32/DTI/fa.nii.gz'
    dname = '/usr/share/fsl/data/standard/'
    fname = dname + 'FMRIB58_FA_1mm.nii.gz'
    #fname = '/home/eg309/Data/trento_processed/subj_01/MPRAGE_32/rawbet.nii.gz'
    img = nib.load(fname)
    data = img.get_data()
    affine = img.get_affine()

    Init()

    window = Window(caption="[F]OS", bgcolor=(0.4, 0.4, 0.9))
    scene = Scene(activate_aabb=False)

    guil = Guillotine('VolumeSlicer', data, affine)

    scene.add_actor(guil)
    window.add_scene(scene)
    window.refocus_camera()
    window.show()

    Run()
