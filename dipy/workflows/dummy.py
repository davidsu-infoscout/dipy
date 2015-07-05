from glob import glob
from os.path import dirname, join
from dipy.utils.six import string_types


def dummy_func_interface(input1, input2, output,
                       param1, param2):

    if isinstance(input1, string_types):
        g1 = glob(input1)
        g2 = glob(input2)
    else:
        g1 = input1
        g2 = input2

    for (f1, f2) in zip(g1, g2):

        print('Input files')
        print(f1)
        print(f2)

        if output == 'same':
            f3 = join(dirname(f1), 'out.nii.gz')
            print('Output files')
            print(f3)

        else:

            f3 == join(output, 'out.nii.gz')
            print('Output files')
            print(f3)

        # dummy_func function is called here from the corresponding dipy module
        # load any needed images from f1, f2
        # dummy_func(im1, im2, im3.)
        # save results in f3


if __name__ == '__main__':

    input1 = ['/home/eleftherios/Data/exp1/subj_01/fa_1x1x1.nii.gz',
              '/home/eleftherios/Data/exp1/subj_02/fa_1x1x1.nii.gz']

    input2 = ['/home/eleftherios/Data/exp1/subj_01/t1_warped.nii.gz',
              '/home/eleftherios/Data/exp1/subj_02/t1_warped.nii.gz']

    print('Case 1')
    # lists of file paths and no output dir is given (assuming out_dir same as input_dir)
    dummy_func_interface(input1, input2, 'same', 10, 20)

    print('Case 2')
    # only single file paths are given and no output dir is given
    dummy_func_interface(input1[0], input2[0], 'same', 10, 20)

    print('Case 3')
    # single input files with
    output_dir = '/home/eleftherios/Data/out_dir'
    dummy_func_interface(input1[0], input2[0], output_dir, 10, 20)

    print('Case 4')
    # Use wild chars to include multiple subjects
    dummy_func_interface('/home/eleftherios/Data/exp1/sub*/fa_*.nii.gz',
                         '/home/eleftherios/Data/exp1/sub*/t1_*.nii.gz',
                         'same', 10, 20)

