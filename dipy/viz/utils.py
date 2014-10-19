import numpy as np
import vtk


def rescale_to_uint8(data):
    """ Rescales value of a ndarray to 8 bits unsigned integer
            This function rescales the values of the input between 0 and 255,
            then copies it to a new 8 bits unsigned integer array.

        Parameters
        ----------
            data : ndarray

        Return
        ------
            uint8 ndarray

        Note
        ----
            NAN are clipped to 0. If min equals max, result will be all 0.
    """

    temp = np.array(data, dtype=np.float64)
    temp[np.isnan(temp)] = 0
    temp -= np.min(temp)
    if np.max(temp) != 0.0:
        temp /= np.max(temp)
        temp *= 255.0
    temp = np.array(np.round(temp), dtype=np.uint8)

    return temp


def ndarray2vtkImageData(data):
    """ Transforms ndarray into VTK tangible object
            Transforms a Numpy NdArray into a uint8 vtkImageData object. This
            function uses rescale_to_uint8 so it can be streamed as a unsigned
            char string to vtk.

        Parameters
        ----------
            data : ndarray

        Return
        ------
            vtkImageData
    """

    nb_composites = len(data.shape)

    if nb_composites == 3:
        [sx, sy, sz] = data.shape
        nb_channels = 1
    elif nb_composites == 4:
        [sx, sy, sz, nb_channels] = data.shape
    else:
        exit()

    # Convert data to uint8 properly
    uint8_data = rescale_to_uint8(data)
    uint8_data = np.swapaxes(uint8_data, 0, 2)
    string_data = uint8_data.tostring()

    # Set data importer
    data_source = vtk.vtkImageImport()
    data_source.SetNumberOfScalarComponents(nb_channels)
    data_source.SetDataScalarTypeToUnsignedChar()
    data_source.SetWholeExtent(0, sx-1, 0, sy-1, 0, sz-1)
    data_source.SetDataExtentToWholeExtent()
    data_source.CopyImportVoidPointer(string_data, len(string_data))
    data_source.Update()

    return data_source.GetOutput()
