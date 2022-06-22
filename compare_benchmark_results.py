import sys

import numpy as np
import itertools
import argparse


datasets = [
    ('DIMACS10', 'citationCiteseer'),
    ('SNAP', 'web-Stanford'),
    ('Janna', 'StocF-1465'),
    ('GHS_psdef', 'ldoor'),
]

reductions = ['sum', 'mean', 'min', 'max']
with_backwards = [False, True]
sizes = [1, 16, 32, 64, 128, 256, 512]


def compare():
    benchmark_results_after = np.load(args.after, allow_pickle=True)[()]
    benchmark_results_before = np.load(args.before, allow_pickle=True)[()]
    original_stdout = sys.stdout
    for with_backward, reduce in itertools.product(with_backwards, reductions):
        with open(args.filename, 'a+') as f:
            sys.stdout = f
            print(f"## {reduce}, backward={with_backward}", end='  \n')
            print()
            sys.stdout = original_stdout
        for d in datasets:
            group, name = d
            name = f'{group}/{name}'
            key_prepend = f'{name}_{reduce}_{with_backward}'
            t1_before = benchmark_results_before[key_prepend + 'SCA1_ROW']
            t2_before = benchmark_results_before[key_prepend + 'SCA1_COL']
            t3_before = benchmark_results_before[key_prepend + 'SCA2_ROW']
            t4_before = benchmark_results_before[key_prepend + 'SCA2_COL']
            t6_before = benchmark_results_before[key_prepend + 'SEG_CSR']
            t1_after = benchmark_results_after[key_prepend + 'SCA1_ROW']
            t2_after = benchmark_results_after[key_prepend + 'SCA1_COL']
            t3_after = benchmark_results_after[key_prepend + 'SCA2_ROW']
            t4_after = benchmark_results_after[key_prepend + 'SCA2_COL']
            t6_after = benchmark_results_after[key_prepend + 'SEG_CSR']

            ts = [t1_before, t2_before, t3_before, t4_before, t6_before,
                  t1_after, t2_after, t3_after, t4_after, t6_after]
            for t in ts:
                np.nan_to_num(t, copy=False)

            t1 = np.divide(t1_before, t1_after)
            t2 = np.divide(t2_before, t2_after)
            t3 = np.divide(t3_before, t3_after)
            t4 = np.divide(t4_before, t4_after)
            t6 = np.divide(t6_before, t6_after)

            with open(args.filename, 'a+') as f:
                sys.stdout = f
                print(f'**{name}**', end='  \n')
                print('\t'.join(['|       |'] +
                                [f'{size:>5}|' for size in sizes]), end='  \n')
                print('----'.join(['|------|'] +
                      ['-------|' for _ in sizes]), end='  \n')
                print('\t'.join(['|**SCA1_ROW**|'] +
                                [f'{t:.5f}|' for t in t1]), end='  \n')
                print('\t'.join(['|**SCA1_COL**|'] +
                                [f'{t:.5f}|' for t in t2]), end='  \n')
                print('\t'.join(['|**SCA2_ROW**|'] +
                                [f'{t:.5f}|' for t in t3]), end='  \n')
                print('\t'.join(['|**SCA2_COL**|'] +
                                [f'{t:.5f}|' for t in t4]), end='  \n')
                print('\t'.join(['|**SEG_CSR**|'] +
                                [f'{t:.5f}|' for t in t6]), end='  \n')
                print()
                sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=str, required=True)
    parser.add_argument('--after', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()
    assert args.before.endswith('.npy'), "before must be a .npy file"
    assert args.after.endswith('.npy'), "after must be a .npy file"
    assert args.filename.endswith('.md'), "filename must be a .md file"

    compare()
