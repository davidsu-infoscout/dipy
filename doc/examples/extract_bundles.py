"""
===========================
Automatic bundle extraction
===========================
"""
from os.path import basename
import nibabel.trackvis as tv
from dipy.segment.extractbundles import ExtractBundles
from glob import glob
from dipy.viz import fvtk
from time import time
from copy import deepcopy


def read_trk(fname):
    streams, hdr = tv.read(fname, points_space='rasmm')
    return [i[0] for i in streams]


def write_trk(fname, streamlines, hdr=None):
    streams = ((s, None, None) for s in streamlines)
    if hdr is not None:
        hdr2 = deepcopy(hdr)
        tv.write(fname, streams, hdr2, points_space='rasmm')
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


def janice_data():

    initial_dir = '/home/eleftherios/Data/Hackethon_bdx/'

    dname_model_bundles = initial_dir + 'bordeaux_tracts_and_stems/'

    model_bundle_trk = dname_model_bundles + \
        't0337/tracts/IFOF_R/t0337_IFOF_R_GP.trk'

    model_bundle = read_trk(model_bundle_trk)

    dname_whole_tracks = initial_dir + \
        'bordeaux_whole_brain_DTI/whole_brain_trks_60sj/'

    wb_trk1 = dname_whole_tracks + 't0337_dti_mean02_fact-45_splined.trk'

    #wb_trk2 = dname_whole_tracks + 't0126_dti_mean02_fact_45.trk'

    wb1 = read_trk(wb_trk1)

    dname_results = initial_dir + 'automatic_extraction_for_ifof/'

    return wb1, model_bundle, dname_whole_tracks, dname_results


def janice_validate(tag):

    initial_dir = '/home/eleftherios/Data/Hackethon_bdx/'

    dname_model_bundles = initial_dir + 'bordeaux_tracts_and_stems/'

    manual_bundle_trk = dname_model_bundles + \
        tag + '/tracts/IFOF_R/' + tag + '_IFOF_R_GP.trk'

    manual_bundle = read_trk(manual_bundle_trk)

    return manual_bundle


def janice_next_subject(dname_whole_streamlines, verbose=False):

    for wb_trk2 in glob(dname_whole_streamlines + '*.trk'):

        wb2 = read_trk(wb_trk2)

        if verbose:
            print(wb_trk2)

        tag = basename(wb_trk2).split('_')[0]

        if verbose:
            print(tag)

        yield (wb2, tag)


out = janice_data()
model_streamlines, model_bundle, dname_whole_streamlines, dname_results = out

for (streamlines, tag) in janice_next_subject(dname_whole_streamlines):

    print(tag)

    eb = ExtractBundles(strategy='B', min_thr=5.)

    t0 = time()

    extracted_bundle = eb.extract(streamlines, model_streamlines,
                                  model_bundle)
    print('Duration %f' % (time() - t0, ))

    #show_bundles(streamlines, model_streamlines)
    show_bundles(eb.moved_model_bundle, extracted_bundle)

    manual_bundle = janice_validate(tag)
    show_bundles(manual_bundle, extracted_bundle)


