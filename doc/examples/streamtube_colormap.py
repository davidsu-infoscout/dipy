"""

==================================
Display streamlines with colormap
==================================

Overview
========

dipy can display streamlines with a colormap
In this example we give a short introduction on how to use it 
to display streamlines with colormap and scalar_bar from fvtk

"""

"""
1. Import libraries
"""
import numpy as np
import nibabel as nib
from dipy.viz import fvtk
import dipy.viz.fvtk_actors as vtk_a

"""
2. Read/write trackvis streamline files with nibabel.
"""

dname = '/home/eleftherios/Data/fancy_data/2013_02_14_Samuel_St-Jean/TRK_files/'

streamlines_file = dname + "bundles_cc_1.trk"
streams, hdr = nib.trackvis.read(streamlines_file, points_space='rasmm')
lines = [s[0] for s in streams]

"""
3. Load colormap (FA map for this example)
"""
fa_file = nib.load(dname + "../fa_1x1x1.nii.gz")
fa_colormap = fa_file.get_data()
colormap_affine = fa_file.get_affine()

"""
4. Transform lines in the same coordinates
"""
transfo = np.linalg.inv(colormap_affine)
lines = [nib.affines.apply_affine(transfo, s) for s in lines]

"""
5. Generate and render fvtk streamline with scalar_bar
"""
width = 0.1
fvtk_tubes = vtk_a.streamtube(lines, fa_colormap, linewidth=width)
scalar_bar = vtk_a.scalar_bar(fvtk_tubes.GetMapper().GetLookupTable())

renderer = fvtk.ren()
fvtk.add(renderer, fvtk_tubes)
fvtk.add(renderer, scalar_bar)
fvtk.show(renderer)

"""
6. Generate and render fvtk streamline with scalar_bar
"""

saturation = [0.0,1.0] # white to red
hue = [0.0,0.0] # Red only

lut_cmap = vtk_a.colormap_lookup_table(hue_range=hue, saturation_range=saturation)

fvtk_tubes = vtk_a.streamtube(lines, fa_colormap, linewidth=width,
                              lookup_colormap=lut_cmap)

scalar_bar = vtk_a.scalar_bar(fvtk_tubes.GetMapper().GetLookupTable())

renderer = fvtk.ren()
fvtk.add(renderer, fvtk_tubes)
fvtk.add(renderer, scalar_bar)
fvtk.show(renderer)
