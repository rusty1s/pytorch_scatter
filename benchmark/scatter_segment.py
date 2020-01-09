# flake8: noqa

import time
import os.path as osp
import itertools

import argparse
import wget
import torch
from scipy.io import loadmat

import torch_scatter
from torch_scatter import scatter_add, scatter_mean, scatter_min, scatter_max
from torch_scatter import segment_coo, segment_csr

iters = 20
sizes = [1, 16, 32, 64, 128, 256, 512]

short_rows = [
    ('DIMACS10', 'citationCiteseer'),
    ('SNAP', 'web-Stanford'),
]
long_rows = [
    ('Janna', 'StocF-1465'),
    ('GHS_psdef', 'ldoor'),
]


def download(dataset):
    url = 'https://sparse.tamu.edu/mat/{}/{}.mat'
    for group, name in itertools.chain(long_rows, short_rows):
        if not osp.exists(f'{name}.mat'):
            print(f'Downloading {group}/{name}:')
            wget.download(url.format(group, name))
            print('')


def bold(text, flag=True):
    return f'\033[1m{text}\033[0m' if flag else text


@torch.no_grad()
def correctness(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(args.device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(args.device, torch.long)
    dim_size = rowptr.size(0) - 1

    for size in sizes:
        try:
            x = torch.randn((row.size(0), size), device=args.device)
            x = x.squeeze(-1) if size == 1 else x

            out1 = scatter_add(x, row, dim=0, dim_size=dim_size)
            out2 = segment_coo(x, row, dim_size=dim_size, reduce='add')
            out3 = segment_csr(x, rowptr, reduce='add')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

            out1 = scatter_mean(x, row, dim=0, dim_size=dim_size)
            out2 = segment_coo(x, row, dim_size=dim_size, reduce='mean')
            out3 = segment_csr(x, rowptr, reduce='mean')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

            x = x.abs_().mul_(-1)

            out1, arg_out1 = scatter_min(x, row, 0, torch.zeros_like(out1))
            out2, arg_out2 = segment_coo(x, row, reduce='min')
            out3, arg_out3 = segment_csr(x, rowptr, reduce='min')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

            x = x.abs_()

            out1, arg_out1 = scatter_max(x, row, 0, torch.zeros_like(out1))
            out2, arg_out2 = segment_coo(x, row, reduce='max')
            out3, arg_out3 = segment_csr(x, rowptr, reduce='max')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

        except RuntimeError:
            torch.cuda.empty_cache()


@torch.no_grad()
def time_func(func, x):
    try:
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            func(x)
        torch.cuda.synchronize()
        return time.perf_counter() - t
    except RuntimeError:
        torch.cuda.empty_cache()
        return float('inf')


@torch.no_grad()
def timing(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(args.device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(args.device, torch.long)
    row_perm = row[torch.randperm(row.size(0))]
    dim_size = rowptr.size(0) - 1
    avg_row_len = row.size(0) / dim_size

    sca_row = lambda x: getattr(torch_scatter, f'scatter_{args.reduce}')(
        x, row, dim=0, dim_size=dim_size)
    sca_col = lambda x: getattr(torch_scatter, f'scatter_{args.reduce}')(
        x, row_perm, dim=0, dim_size=dim_size)
    seg_coo = lambda x: segment_coo(x, row, reduce=args.reduce)
    seg_csr = lambda x: segment_csr(x, rowptr, reduce=args.reduce)
    dense1 = lambda x: getattr(torch, args.dense_reduce)(x, dim=-2)
    dense2 = lambda x: getattr(torch, args.dense_reduce)(x, dim=-1)

    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []

    for size in sizes:
        try:
            x = torch.randn((row.size(0), size), device=args.device)
            x = x.squeeze(-1) if size == 1 else x

            t1 += [time_func(sca_row, x)]
            t2 += [time_func(sca_col, x)]
            t3 += [time_func(seg_coo, x)]
            t4 += [time_func(seg_csr, x)]

            del x

        except RuntimeError:
            torch.cuda.empty_cache()
            for t in (t1, t2, t3, t4):
                t.append(float('inf'))

        try:
            x = torch.randn((dim_size, int(avg_row_len + 1), size),
                            device=args.device)

            t5 += [time_func(dense1, x)]
            x = x.view(dim_size, size, int(avg_row_len + 1))
            t6 += [time_func(dense2, x)]

            del x

        except RuntimeError:
            torch.cuda.empty_cache()
            for t in (t5, t6):
                t.append(float('inf'))

    ts = torch.tensor([t1, t2, t3, t4, t5, t6])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    name = f'{group}/{name}'
    print(f'{bold(name)} (avg row length: {avg_row_len:.2f}):')
    print('\t'.join(['       '] + [f'{size:>5}' for size in sizes]))
    print('\t'.join([bold('SCA_ROW')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t1, winner[0])]))
    print('\t'.join([bold('SCA_COL')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t2, winner[1])]))
    print('\t'.join([bold('SEG_COO')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t3, winner[2])]))
    print('\t'.join([bold('SEG_CSR')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t4, winner[3])]))
    print('\t'.join([bold('DENSE1 ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t5, winner[4])]))
    print('\t'.join([bold('DENSE2 ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t6, winner[5])]))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce', type=str, required=True,
                        choices=['add', 'mean', 'min', 'max'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    args.dense_reduce = 'sum' if args.reduce == 'add' else args.reduce

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=args.device).sum()
    for dataset in itertools.chain(short_rows, long_rows):
        download(dataset)
        correctness(dataset)
        timing(dataset)
