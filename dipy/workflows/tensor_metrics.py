from glob import glob
from os.path import dirname, join


def tensor_metrics(fgrad, fraw, fout_dir):
    pass

def workflow_example(fraw1, fraw2, fout_dir='same'):

   for (f1, f2) in zip(glob(fraw1), glob(fraw2)):

        print('Input files')
        print(f1)
        print(f2)
        print('\n')

        if fout_dir == 'same':

            dname = dirname(f1)
            f3 = join(dname, 'dummy.txt')
            print('Output files')
            print(f3)
            print('\n')





