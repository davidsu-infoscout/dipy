#    TODO:
#         Tutorial
#         Have a memorized current camera transformation
#         Add display of plane and camera information
#         Support VTK6
#         Find a place to put utils.py
#         Add support for LUT input

import vtk
import utils
import os



class Guillotine:
    KeyboardShortcuts = """
VTK default interactor keys:
    j / t : toggle between joystick (position sensitive) and trackball
            (motion sensitive) styles. In joystick style, motion occurs
            continuously as long as a mouse button is pressed. In
            trackball style, motion occurs when the mouse button is
            pressed and the mouse pointer moves.
    c / a : toggle between camera and actor modes. In camera mode,
            mouse events affect the camera position and focal point. In
            actor mode, mouse events affect the actor that is under the
            mouse pointer.
    1 :     rotate the camera around its focal point (if camera mode)
            or rotate the actor around its origin (if actor mode). The
            rotation is in the direction defined from the center of the
            renderer's viewport towards the mouse position. In joystick
            mode, the magnitude of the rotation is determined by the
            distance the mouse is from the center of the render window.
    2 :     pan the camera (if camera mode) or translate the actor (if
            actor mode). In joystick mode, the direction of pan or
            translation is from the center of the viewport towards the
            mouse position. In trackball mode, the direction of motion
            is the direction the mouse moves. (Note: with 2-button
            mice, pan is defined as <Shift>-Button 1.)
    3 :     zoom the camera (if camera mode) or scale the actor (if
            actor mode). Zoom in/increase scale if the mouse position
            is in the top half of the viewport; zoom out/decrease scale
            if the mouse position is in the bottom half. In joystick
            mode, the amount of zoom is controlled by the distance of
            the mouse pointer from the horizontal centerline of the
            window.
    3 :     toggle the render window into and out of stereo mode. By
            default, red-blue stereo pairs are created. Some systems
            support Crystal Eyes LCD stereo glasses; you have to invoke
            SetStereoTypeToCrystalEyes() on the rendering window.
    e :     exit the application.
    f :     fly to the picked point
    i :     toggle display of the interactor
    p :     perform a pick operation. The render window interactor has
            an internal instance of vtkCellPicker that it uses to pick.
    r :     reset the camera view along the current view direction.
            Centers the actors and moves the camera so that all actors
            are visible.
    s :     modify the representation of all actors so that they are
            surfaces.
    u :     invoke the user-defined function. Typically, this keypress
            will bring up an interactor that you can type commands in.
            Typing u calls UserCallBack() on the
            vtkRenderWindowInteractor, which invokes a
            vtkCommand::UserEvent. In other words, to define a
            user-defined callback, just add an observer to the
            vtkCommand::UserEvent on the vtkRenderWindowInteractor
            object.
    w :     modify the representation of all actors so that they are
            wireframe.

Guillotine Keys:
    x :                 clip plane normal to x axis
    y :                 clip plane normal to y axis
    z :                 clip plane normal to z axis
    r :                 reset the camera to initial standard position
    Shift-s :           set a sagital view
    Shift-c :           set a coronal view
    Shift-a :           set an axial view
    o :                 take a snapshot of the window and stores the
                        PNG file into the working directory
    Delete :            toggle display of the axes
    Right :             tilt the guillotine to the right
    Left :              tilt the guillotine to the left
    Up :                tilt the guillotine up
    Down :              tilt the guillotine down
    Prior(Page down) :  decrease the slice index along the normal
    Next (Page Up) :    increase the slice index along the normal
    Shift-Right :       roll the camera right
    Shift-Left :        roll the camera left
    Shift-Up :          zoom in the camera's view angle
    Shift-Down :        zoom out the camera's view angle
    Shift-Prior :       (Page down) fast prior key
    Shift-Next :        (Page Up) fast n
"""
    def __init__(self):
        """Data volume slicing utility
            Once instantiated, several volumes can be added to this class to be
            mixed and cut together in an interactive window. Also, any actors
            like streamlines or glyphs could be added so different data could
            be visualized at once. The main feature of this class is the
            ability to explore volume data through interactive slicing.

        Methods
        -------
            add_data_volume :    Adds volume data in the blending
            add_actor :          Adds actor to the renderer (optional)
            build :              Sets the guillotine ready for showing or
                                 snapshot.
            set_plane :          Sets the plane origin and normal
            set_plane_angle :    Sets the plane to a standard angle
            move_camera :        Move the camera from its current position
                                 around the object
            set_camera :         Sets the camera's position around the object
            set_camera_angle :   Sets the camera to a standard angle
            set_view_angle :     Sets the plane and camera to a standard angle
            toggle_axes :        Toggles the display of the axes
            show :               Displays an interactive window of the current
                                 view
            snapshot :           Takes a snapshot of the current view

        Example
        -------
        # First, instantiate the class
        g = Guillotine()

        # Use input methods to add data volume(s) and actor(s) if desired
        d = file.data
        d_affine = file.affine
        g.add_data_volume(d, affine = d_affine)

        a = vtk.vtkActor()
        g.add_actor(a)

        # Use the build() method to create the guillotine
        g.build()

        # You can now place the camera, set the cutting plane or display the
        # axes using the display methods
        g.set_view_angle("sagital")

        # Display or create a snapshot from the result
        g.show()
        """

        self.is_built = False
        self.nb_data_volumes = 0
        self.plane = vtk.vtkPlane()
        self.blender = vtk.vtkImageBlend()
        self.plane_widget = vtk.vtkImplicitPlaneWidget()
        self.axes = vtk.vtkCubeAxesActor()

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0, 0, 0)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1024, 768)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.AddObserver("KeyPressEvent", self._callback)

    def _callback(self, object, event):
        """Callback function for interaction events
            This function is called every time a key is hit or interaction is
            done.

        Parameters
        ----------
            object : vtkObject
                Object that calls this function
            event : vtkEvent
                Event that triggered the calling of this function
        """

        if event == "InteractionEvent":
            self._update_plane()
        elif event == "KeyPressEvent":
            key = object.GetKeySym()
            shift_pressed = object.GetShiftKey()

            if key == "x":
                self.set_plane_angle("sagital")
            elif key == "y":
                self.set_plane_angle("coronal")
            elif key == "z":
                self.set_plane_angle("axial")
            elif key == "r":
                self.set_plane((0, 0, 0))
                self.set_view_angle("axial")
            elif key == "S":
                self.set_view_angle("sagital")
            elif key == "C":
                self.set_view_angle("coronal")
            elif key == "A":
                self.set_view_angle("axial")
            elif key == "o":
                self.snapshot()
            elif key == "Delete":
                self.toggle_axes()
            elif key == "Right":
                if shift_pressed:
                    self.move_camera(roll=-15)
                else:
                    self.move_camera(azimuth=-15)
            elif key == "Left":
                if shift_pressed:
                    self.move_camera(roll=15)
                else:
                    self.move_camera(azimuth=15)
            elif key == "Down":
                if shift_pressed:
                    self.move_camera(zoom=0.9)
                else:
                    self.move_camera(elevation=15)
            elif key == "Up":
                if shift_pressed:
                    self.move_camera(zoom=1.1)
                else:
                    self.move_camera(elevation=-15)
            elif key == "Prior" or key == "Next":
                if key == "Prior":
                    factor = 1.0
                else:
                    factor = -1.0
                if shift_pressed:
                    factor *= 5.0
                origin = self.plane_widget.GetOrigin()
                normal = self.plane_widget.GetNormal()
                self.set_plane((origin[0] + factor*normal[0],
                                origin[1] + factor*normal[1],
                                origin[2] + factor*normal[2]))

    def _update_plane(self):
        """Updates the cutting plane
            When this function is called, the plane is updated according to
            its normal and origin settings.

        Parameters
        ----------
            object : vtkObject
                Object that calls this function
            event : vtkEvent
                Event that triggered the calling of this function
        """

        self.plane_widget.GetPlane(self.plane)
        self.interactor.Render()

    """
    INPUT METHODS
    -------------
        Primary functions to build a guillotine
    """

    def add_data_volume(self, data, opacity=0.5, affine=None):
        """ Adds volume data in the blending
            This function takes a given volume, transforms it according to its
            optional affine matrix and blends it with previous volume, if
            present, and with optional opacity.

        Parameters
        ----------
            data : 3D ndarray
            opacity : float
            affine : 4x4 ndarray

        Note
        ----
        Every data volume added should have the same size and the one with
        the most components added first (rgb before grayscale).
        """

        image_data = utils.ndarray2vtkImageData(data)

        # Set the transform (identity if none given)
        transform = vtk.vtkTransform()
        if affine is not None:
            transform_matrix = vtk.vtkMatrix4x4()
            transform_matrix.DeepCopy((
                affine[0][0], affine[0][1], affine[0][2], affine[0][3],
                affine[1][0], affine[1][1], affine[1][2], affine[1][3],
                affine[2][0], affine[2][1], affine[2][2], affine[2][3],
                affine[3][0], affine[3][1], affine[3][2], affine[3][3]))
            transform.SetMatrix(transform_matrix)
            transform.Inverse()

        # Set the reslicing
        image_reslice = vtk.vtkImageReslice()
        image_reslice.SetInput(image_data)
        image_reslice.SetResliceTransform(transform)
        image_reslice.AutoCropOutputOn()
        image_reslice.SetInterpolationModeToLinear()
        image_reslice.Update()

        # Check if the extent is compatible with previous volumes
        if self.nb_data_volumes > 0:
            assert(
                image_reslice.GetOutput().GetExtent() ==
                self.blender.GetOutput().GetWholeExtent()), \
                "image data extent doesn't fit the actual extent. (" + \
                str(image_reslice.GetOutput().GetExtent()) + \
                " vs actual " + \
                str(self.blender.GetOutput().GetWholeExtent()) + \
                ")"

        # Blend volumes
        self.blender.SetInput(self.nb_data_volumes, image_reslice.GetOutput())
        self.blender.SetOpacity(self.nb_data_volumes, opacity)
        self.blender.UpdateWholeExtent()
        extent = self.blender.GetOutput().GetExtent()
        ori = self.blender.GetOutput().GetOrigin()
        self.bbox = ((extent[0] + ori[0]),
                (extent[1] + ori[0]),
                (extent[2] + ori[1]),
                (extent[3] + ori[1]),
                (extent[4] + ori[2]),
                (extent[5] + ori[2]))
        self.nb_data_volumes += 1

    def add_actor(self, actor):
        """Adds actor to the renderer (optional)
            This function is useful to add any extra actor needed in the
            display.

        Parameters
        ----------
            actor : vtkActor
                Actor to add in the display
        """

        self.renderer.AddActor(actor)

    def build(self):
        """Sets the guillotine ready for showing or snapshot.
            This function must be called after add_data_volume as been used at
            least once. add_actor must be called before this one too. It sets
            the renderer, window, interactor and widget ready for rendering.
        """

        # Get blending result
        assert(
            self.nb_data_volumes > 0), \
            "No data volume, use function add_data_volume."

        # Set cutter filter
        cutter = vtk.vtkCutter()
        cutter.SetInput(self.blender.GetOutput())
        cutter.SetCutFunction(self.plane)

        # Set the cutter mappers
        cutter_mapper = vtk.vtkPolyDataMapper()
        cutter_mapper.SetInputConnection(cutter.GetOutputPort())

        # Set cutter actor
        cutter_actor = vtk.vtkActor()
        cutter_actor.SetMapper(cutter_mapper)
        self.add_actor(cutter_actor)

        # Set plane widget
        self.plane_widget.SetInput(self.blender.GetOutput())
        self.plane_widget.SetInteractor(self.interactor)
        self.plane_widget.AddObserver("InteractionEvent", self._callback)
        self.plane_widget.SetDrawPlane(0)
        self.plane_widget.SetScaleEnabled(0)
        self.plane_widget.SetTubing(0)
        self.plane_widget.OutlineTranslationOff()
        self.plane_widget.OutsideBoundsOff()
        self.plane_widget.SetPlaceFactor(1)
        self.plane_widget.PlaceWidget()
        self.plane_widget.SetNormal(0, 0, 1)

        self.is_built = True
        
        # Set axes
        self.axes.SetBounds(self.bbox[0],
                            self.bbox[1],
                            self.bbox[2],
                            self.bbox[3],
                            self.bbox[4],
                            self.bbox[5])
        self.axes.SetTickLocationToBoth()
        self.axes.SetFlyModeToOuterEdges()
        self.axes.SetCamera(self.renderer.GetActiveCamera())
        self.toggle_axes()
        self.add_actor(self.axes)

        # Initialize the view angle
        self.set_camera_angle("axial")

    """
    DISPLAY METHODS
    ---------------
        Optional functions to change the guillotine's view
    """

    def set_plane(self, origin=None, normal=None):
        """Sets the plane origin and normal

        Parameters
        ----------
            origin : tuple of size 3
                (x, y, z) position of the plane's origin
            normal : tuple of size 3
                (x, y, z) direction of the normal
        """
        
        assert (self.is_built), "Error ! Call build() method before."
        
        if origin is not None:
            # Constraint the origin inside the data limits
            eps = 0.0001
            origin = (min(max(origin[0], self.bbox[0] + eps), self.bbox[1] - eps),
                      min(max(origin[1], self.bbox[2] + eps), self.bbox[3] - eps),
                      min(max(origin[2], self.bbox[4] + eps), self.bbox[5] - eps))
            self.plane_widget.SetOrigin(origin)
        if normal is not None:
            self.plane_widget.SetNormal(normal)

        self.plane_widget.InvokeEvent("InteractionEvent")

    def set_plane_angle(self, angle):
        """Sets the plane to a standard angle

        Parameters
        ----------
            angle : string ("sagital", "coronal" or "axial")
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        if angle == "sagital":
            self.plane_widget.SetNormal(1, 0, 0)
        elif angle == "coronal":
            self.plane_widget.SetNormal(0, 1, 0)
        elif angle == "axial":
            self.plane_widget.SetNormal(0, 0, 1)
        else:
            print "Not a valid angle"
            return

        self.plane_widget.InvokeEvent("InteractionEvent")

    def move_camera(self, azimuth=0, elevation=0, roll=0, zoom=0):
        """Move the camera from its current position around the object

        Parameters
        ----------
            azimuth : float
                angle in degrees around the Y axis of the object
            elevation : float
                angle in degrees relative to the Y axis of the object
            roll : float
                angle in degrees around the projection axis
            zoom : float
                ratio of the angle of view of the camera's virtual lens
                (1.0 is 30 degrees, 2.0 is 15 degrees)
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        cam = self.renderer.GetActiveCamera()
        cam.Azimuth(azimuth)
        cam.Elevation(elevation)
        cam.Roll(roll)
        cam.Zoom(zoom)

        self.interactor.Render()

    def set_camera(self, azimuth=0, elevation=0, roll=0, zoom=0):
        """Sets the camera's position around the object

        Parameters
        ----------
            azimuth : float
                angle in degrees around the Y axis of the object
            elevation : float
                angle in degrees relative to the Y axis of the object
            roll : float
                angle in degrees around the projection axis
            zoom : float
                ratio of the angle of view of the camera's virtual lens
                (1.0 is 30 degrees, 2.0 is 15 degrees)
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        self.set_camera_angle("axial")
        self.move_camera(azimuth, elevation, roll, zoom)

    def set_camera_transfo(self, transfo):
        """Sets the camera's position with vtk transformation

        Parameters
        ----------
            transfo : vtk.vtkTransform
                from a vtk.vtkMatrix4x4
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        cam = self.renderer.GetActiveCamera()
        cam.ApplyTransform(transfo)

    def set_camera_angle(self, angle):
        """Sets the camera to a standard angle

        Parameters
        ----------
            angle : string ("sagital", "coronal" or "axial")
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        cam = self.renderer.GetActiveCamera()

        if angle == "sagital":
            cam.SetPosition(1, 0, 0)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 0, 1)
        elif angle == "coronal":
            cam.SetPosition(0, -1, 0)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 0, 1)
        elif angle == "axial":
            cam.SetPosition(0, 0, 1)
            cam.SetFocalPoint(0, 0, 0)
            cam.SetViewUp(0, 1, 0)
        else:
            print "Not a valid angle"
            return

        cam.SetViewAngle(30)
        self.renderer.ResetCamera()
        self.plane_widget.InvokeEvent("InteractionEvent")

    def set_view_angle(self, angle):
        """Sets the plane and camera to a standard angle

        Parameters
        ----------
            angle : string ("sagital", "coronal" or "axial")
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        self.set_camera_angle(angle)
        self.set_plane_angle(angle)

    def toggle_axes(self):
        """Toggles the display of the axes
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        self.axes.SetXAxisVisibility(not self.axes.GetXAxisVisibility())
        self.axes.SetYAxisVisibility(not self.axes.GetYAxisVisibility())
        self.axes.SetZAxisVisibility(not self.axes.GetZAxisVisibility())

        self.interactor.Render()

    """
    RENDERING METHODS
    -----------------
        Guillotine's display and output functions
    """

    def show(self):
        """Displays an interactive window of the current view
            Once the guillotine as been built and the view optionally changed,
            this function shows the result in an interactive view, where a
            widget allows you to pla with the displayed data. The key actions
            are described in th _callback function.
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        self.renderer.Render()
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

    def snapshot(self, filename=None, render_size=None):
        """Takes a snapshot of the current view
            Instead of showing an interactive window, a screen capture is taken
            from the built guillotine. This is convenient for scripting or
            when you know exactly which view of the data you need. The
            magnification can be used to render the scene high quality thanks
            to a virtual multi-screen rendering.

        Parameters
        ----------
            filename : string
                Output filename ending with ".png" extension
            render_size : tuple of size 2
                Width and height of the captured image
        """
        
        assert (self.is_built), "Error ! Call build() method before."

        number = 0
        if filename is None:
            filename = "guillotine_snapshot_"
            while os.path.isfile(filename + str(number) + ".png"):
                number += 1
            filename = filename + str(number) + ".png"
        elif os.path.isfile(filename):
            print "Warning : Overwriting existing file."

        # Adjust the magnification and render size to optimize the rendering
        magnification = 1
        if render_size is None:
            new_render_size = self.render_window.GetSize()
        else:
            new_render_size = render_size
            while new_render_size[0] > 1000 or new_render_size[1] > 1000:
                magnification += 1
                new_render_size = (render_size[0] / magnification,
                                   render_size[1] / magnification)
            self.render_window.SetSize(new_render_size)

        large_renderer = vtk.vtkRenderLargeImage()
        large_renderer.SetInput(self.renderer)
        large_renderer.SetMagnification(magnification)
        large_renderer.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetInput(large_renderer.GetOutput())
        writer.SetFileName(filename)
        writer.Write()
