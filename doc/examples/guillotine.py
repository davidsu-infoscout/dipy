"""
======================================================
Using the Guillotine to visualize different data types
======================================================
"""

"""
Create a useful methods to load data volumes.
"""
import nibabel as nib
from dipy.viz.guillotine import Guillotine

def load_data_volume(filename):
    data = nib.load(filename)
    volume_data = data.get_data()
    affine = data.get_affine()

    return volume_data, affine

"""
Instantiate a guillotine for a simple example.
"""
g1 = Guillotine()
 
"""
Load an FA and it's affine transformation for a simple example.
"""
simple_fa = '/home/eleftherios/Data/MPI_Elef/fa_1x1x1.nii.gz'
simple_fa_data, simple_fa_affine = load_data_volume(simple_fa)
 
"""
Add data to the first guillotine.
"""
g1.add_data_volume(simple_fa_data, affine=simple_fa_affine)
 
"""
Build the guillotine.
"""
g1.build()
 
"""
Show the visualization in an interactive window.
"""
g1.show()
 
"""
Guillotine supports multiple volumes blending.
"""
g2 = Guillotine()
 
"""
Load RGB FA and B0 data. Note that the first one to be added is the one with
the most components, in color, and that opacities are respectively always 100%
for the first, then 50% and 25%. 
"""
#rgb = "/home/algo/Documents/data/set2/rgb.nii"
rgb = '/home/eleftherios/Data/MPI_Elef/rgb_1x1x1.nii.gz'
rgb_data, rgb_affine = load_data_volume(rgb)
g2.add_data_volume(rgb_data, affine=rgb_affine)
  

fa = '/home/eleftherios/Data/MPI_Elef/fa_1x1x1.nii.gz'
fa_data, fa_affine = load_data_volume(fa)
g2.add_data_volume(fa_data, 0.5, fa_affine)
  
#b0 = "/home/algo/Documents/data/set2/b0.nii"
b0 = '/home/eleftherios/Data/MPI_Elef/fa_1x1x1.nii.gz'
b0_data, b0_affine = load_data_volume(b0)
g2.add_data_volume(b0_data, 0.25, b0_affine)
 
g2.build()
g2.show()
 
"""
Guillotine supports additional actors viewing. As an example, add some axes at
the origin.
"""
g3 = Guillotine()
g3.add_data_volume(simple_fa_data, affine=simple_fa_affine)
 
from dipy.viz import fvtk
axes = fvtk.axes((20, 20, 20))
g3.add_actor(axes)
g3.build()
g3.show()

"""
Besides using the keyboard shortcuts, you can set the scene before showing it
and, like in this example, create a specific screen shot.
"""
g4 = Guillotine()
g4.add_data_volume(simple_fa_data, affine=simple_fa_affine)
g4.build()
g4.set_view_angle("sagital")
g4.set_plane((25, 26, 34))
g4.move_camera(30, 30, 30, 1.25)
g4.toggle_axes()
g4.show()
g4.snapshot("snapshot.png", (2 * 1920, 2 * 1080))
