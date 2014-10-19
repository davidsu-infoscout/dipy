import vtk
import numpy as np
from nibabel import trackvis as tv
from fvtk_widgets import slider_widget, button_widget
from fvtk_actors import line


def center_streamlines(streamlines):
    center = np.mean(np.concatenate(streamlines, axis=0), axis=0)
    return [s - center for s in streamlines], center


def load_bundle(name):
    fname = dname + name + '.trk'
    streams, hdr = tv.read(fname)
    streamlines = [s[0] for s in streams]
    return streamlines


def load_big_data():

    fname = '/home/eleftherios/Data/fancy_data/2013_02_08_Gabriel_Girard/streamlines_100K.trk'
    streams, hdr = tv.read(fname)
    streamlines = [s[0] for s in streams]
    return streamlines


if __name__ == '__main__':


    dname = '/home/eleftherios/bundle_paper/data/faisceaux/'

    bundle_names = ['CC_front', 'CC_middle', 'CC_back', \
                    'cingulum_left', 'cingulum_right', \
                    'CST_left', 'CST_right', \
                    'IFO_left', 'IFO_right', \
                    'ILF_left', 'ILF_right',
                    'SCP_left', 'SCP_right', \
                    'SLF_left', 'SLF_right', \
                    'uncinate_left', 'uncinate_right']

    #streamlines = load_bundle('CC_middle')
    streamlines = load_big_data()

    streamlines, _ = center_streamlines(streamlines)

    stream_actor = line(streamlines)

    stream_poly_data = stream_actor.GetMapper().GetInput()

    # implicit = vtk.vtkBox()
    implicit = vtk.vtkSphere()
    implicit.SetCenter(0, 0, 0)
    implicit.SetRadius(10.)

    diameter = 100*np.sqrt(2)/2.

    cube = vtk.vtkCubeSource()
    cube.SetBounds(-diameter, diameter, -diameter, diameter,
                   -diameter, diameter)
    cube.Update()

    extract_geometry = vtk.vtkExtractPolyDataGeometry()
    extract_geometry.SetInput(stream_poly_data)
    extract_geometry.SetImplicitFunction(implicit)
    extract_geometry.ExtractBoundaryCellsOn()
    extract_geometry.ExtractInsideOn()
    extract_geometry.Update()

    extracted_mapper = vtk.vtkPolyDataMapper()
    extracted_mapper.SetInputConnection(extract_geometry.GetOutputPort())
    extracted_mapper.GlobalImmediateModeRenderingOn()

    extracted_actor = vtk.vtkActor()
    extracted_actor.SetMapper(extracted_mapper)

    # extracted_actor = vtk.vtkLODActor()
    # extracted_actor.SetMapper(extracted_mapper)
    # extracted_actor.SetNumberOfCloudPoints(100000)

    ren = vtk.vtkRenderer()
    #ren.AddActor(stream_actor)
    ren.AddActor(extracted_actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    transform = vtk.vtkTransform()

    def filter_streamlines(obj, event):
        global implicit, transform

        obj.GetTransform(transform)
        implicit.SetCenter(transform.GetPosition())
        radius = 5 * max(transform.GetScale())
        implicit.SetRadius(radius)

    #ren.AddActor(fvtk.axes(scale=(100, 100, 100)))

    stream_actor.GetProperty().SetOpacity(.2)
    #ren.AddActor(stream_actor)

    def x_filter(obj, event):

        global implicit
        value = np.round(obj.GetSliderRepresentation().GetValue())

        center = implicit.GetCenter()
        print(center)
        implicit.SetCenter(center[0] + value, center[1], center[2])
        #obj.SetValue(value)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    box_widget = vtk.vtkBoxWidget()
    box_widget.SetInteractor(iren)
    box_widget.SetInput(cube.GetOutput())

    box_widget.HandlesOn()


    box_widget.PlaceWidget()
    box_widget.AddObserver('InteractionEvent', filter_streamlines)
    box_widget.On()

    def press_button(obj, event):
        print('Button pressed')

    button = button_widget(iren, press_button)


    slider = slider_widget(iren, x_filter,
                           min_value=-20, max_value=20,
                           value=0, label='X')

    slider2 = slider_widget(iren, x_filter,
                            min_value=-20, max_value=20,
                            value=0, label='Y',
                            coord1=(0.8, 0.3), coord2=(0.9, 0.3))





    iren.Initialize()

    renWin.Render()
    iren.Start()





