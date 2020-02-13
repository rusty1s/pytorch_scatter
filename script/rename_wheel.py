import sys
import os
import os.path as osp
import glob
import shutil

idx = sys.argv[1]
assert idx in ['cpu', 'cu92', 'cu100', 'cu101']

dist_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'dist')
wheels = glob.glob(osp.join('dist', '**', '*.whl'), recursive=True)

for wheel in wheels:
    if idx in wheel:
        continue

    paths = wheel.split(osp.sep)
    names = paths[-1].split('-')

    name = '-'.join(names[:-4] + ['latest+' + idx] + names[-3:])
    shutil.copyfile(wheel, osp.join(*paths[:-1], name))

    name = '-'.join(names[:-4] + [names[-4] + '+' + idx] + names[-3:])
    os.rename(wheel, osp.join(*paths[:-1], name))
