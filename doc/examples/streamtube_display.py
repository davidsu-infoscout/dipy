"""

========================================
Display streamlines with lines or tubes 
========================================

Overview
========

dipy can display streamlines with line or tube
In this example we give a short introduction on how to use it 
to display streamlines with different colors and width

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
3. Generate and render fvtk streamline with line (without width)
"""
fvtk_lines = vtk_a.line(lines)

renderer = fvtk.ren() 
fvtk.add(renderer, fvtk_lines)
fvtk.show(renderer)

"""
4. Generate and render fvtk streamline with tubes (with width)
"""
tube_width = 0.1
fvtk_tubes = vtk_a.streamtube(lines, linewidth=tube_width)

renderer = fvtk.ren() 
fvtk.add(renderer, fvtk_tubes)
fvtk.show(renderer)

"""
5. Generate random color 
"""
colors = np.random.rand(len(lines), 3) #color per line
#colors = [np.random.rand(len(lines[i]), 3) 
#			for i in range(len(lines))] #color per point
#colors = np.array([0.8,0.0,0.0]) #color per bundle/display
#colors = None #default color (direction rgb)

"""
6. Generate and render fvtk streamline with color
"""
fvtk_lines = vtk_a.streamtube(lines, colors, linewidth=tube_width)

renderer = fvtk.ren() 
fvtk.add(renderer, fvtk_lines)
fvtk.show(renderer)
