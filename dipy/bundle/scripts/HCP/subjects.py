from os.path import expanduser, join

id_ = 0
base_dirs = ['100307', '111312', '194140', '865363', '889579']

home = expanduser("~")

from dipy.external import pipe

for id_ in range(1, 5):

    dname = join(home, 'Data', 'HCP', 'Q1', base_dirs[id_])
    print(dname)

    cmd = 'python upsample.py ' + dname
    print(cmd)
    pipe(cmd)

    cmd = 'python create_streamlines.py ' + dname
    print(cmd)
    pipe(cmd)

    cmd = 'python ants.py ' + dname
    print(cmd)
    pipe(cmd)

    cmd = 'python warp_wmparc.py ' + dname
    print(cmd)
    pipe(cmd)

    cmd = 'python queries.py ' + dname
    print(cmd)
    pipe(cmd)

