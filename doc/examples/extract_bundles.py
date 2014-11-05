"""
===========================
Automatic bundle extraction
===========================
"""
from os.path import basename
import nibabel.trackvis as tv
from glob import glob
from dipy.viz import fvtk
from time import time
from copy import deepcopy
from dipy.segment.extractbundles import whole_brain_registration


def read_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams], hdr


def write_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr2 = deepcopy(hdr)
        tv.write(fname, streams, hdr_mapping=hdr2, points_space='rasmm')
    else:
        tv.write(fname, streams, points_space='rasmm')


def show_bundles(static, moving,  linewidth=0.15, tubes=False, fname=None):

    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(1900, 1200))
    if fname is not None:
        fvtk.record(ren, size=(1900, 1200), out_path=fname)


def janice_next_subject(dname_whole_streamlines, verbose=False):

    for wb_trk2 in glob(dname_whole_streamlines + '*.trk'):

        wb2, hdr = read_trk(wb_trk2)

        if verbose:
            print(wb_trk2)

        tag = basename(wb_trk2).split('_')[0]

        if verbose:
            print(tag)

        yield (wb2, tag)


# Read janice model streamlines
initial_dir = '/home/eleftherios/Data/Hackethon_bdx/'

dname_model_bundles = initial_dir + 'bordeaux_tracts_and_stems/'

model_bundle_trk = dname_model_bundles + \
    't0337/tracts/IFOF_R/t0337_IFOF_R_GP.trk'

model_bundle, _ = read_trk(model_bundle_trk)

dname_whole_brain = initial_dir + \
    'bordeaux_whole_brain_DTI/whole_brain_trks_60sj/'

model_streamlines_trk = dname_whole_brain + \
    't0337_dti_mean02_fact-45_splined.trk'

model_streamlines, hdr = read_trk(model_streamlines_trk)


for (streamlines, tag) in janice_next_subject(dname_whole_brain):

    moved_streamlines, mat, _, _ = whole_brain_registration(model_streamlines,
                                                            streamlines)

    print(tag)
    write_trk(tag + '_moved_streamlines.trk', streamlines, hdr=hdr)

    show_bundles(model_streamlines, streamlines)
    show_bundles(model_streamlines, moved_streamlines)

    break
