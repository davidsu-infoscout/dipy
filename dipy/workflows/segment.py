import os
from dipy.utils.six import string_types
from glob import glob
from dipy.align.streamlinear import whole_brain_slr
from dipy.segment.bundles import recognize_bundles
from dipy.tracking.streamline import transform_streamlines
from nibabel import trackvis as tv


def load_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams], hdr


def save_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr_dict = {key: hdr[key] for key in hdr.dtype.names}
        tv.write(fname, streams, hdr_mapping=hdr_dict, points_space='rasmm')
    else:
        tv.write(fname, streams, points_space='rasmm')


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):

    from dipy.viz import fvtk
    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def recognize_bundles_proxy(streamline_files, model_bundle_files,
                            model_streamlines_file, out_dir='same',
                            close_centroids_thr=str(20),
                            clean_thr=str(5.),
                            local_slr=True):

    verbose = True
    disp = True

    if isinstance(streamline_files, string_types):
        sfiles = glob(streamline_files)
    else:
        raise ValueError('# Streamline_files not a string')

    if isinstance(model_bundle_files, string_types):
        mbfiles = glob(model_bundle_files)

    if out_dir == 'same':
        pass

    print('### Recognition of bundles ###')
    model_streamlines, hrd_model = load_trk(model_streamlines_file)

    print('# Streamline files')
    for sf in sfiles:
        print(sf)
        streamlines, hdr = load_trk(sf)
        ret = whole_brain_slr(model_streamlines, streamlines,
                              maxiter=150, select_random=50000,
                              verbose=verbose)
        moved_streamlines, mat, centroids1, centroids2 = ret
        print(mat)
        if disp:
            show_bundles(model_streamlines, streamlines)
            show_bundles(model_streamlines, moved_streamlines)

        print('# Model_bundle files')
        for mb in mbfiles:
            print(mb)
            model_bundle, hdr_model_bundle = load_trk(mb)
            extracted_bundle, mat2 = recognize_bundles(
                model_bundle, moved_streamlines,
                close_centroids_thr=float(close_centroids_thr),
                clean_thr=float(clean_thr),
                local_slr=bool(local_slr),
                verbose=verbose)

            #bundle_in_model = transform_streamlines(extracted, np.dot(mat2, mat))
            if disp:
                show_bundles(model_bundle, extracted_bundle)

                #show_bundles(model_streamlines, moved_streamlines)

    print('# Model whole brain streamlines file')
    print(model_streamlines_file)