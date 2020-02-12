import os
import os.path as osp
import glob

dist_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'dist')
wheels = glob.glob(osp.join('dist', '**', '*.whl'), recursive=True)

for wheel in wheels:
    idx = wheel.split(osp.sep)[-2]
    if idx not in ['cpu', 'cu92', 'cu100', 'cu101']:
        continue
    name = wheel.split(osp.sep)[-1]
    if idx in name:
        continue

    names = name.split('-')
    name = '-'.join(names[:-4] + [idx] + names[-4:])
    new_wheel = osp.join(*wheel.split(osp.sep)[:-1], name)
    os.rename(wheel, new_wheel)
