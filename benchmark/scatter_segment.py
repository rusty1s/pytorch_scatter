import time
import os.path as osp
import itertools

import argparse
import wget
import torch
from scipy.io import loadmat

from torch_scatter import scatter, segment_coo, segment_csr

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

            out1 = scatter(x, row, dim=0, dim_size=dim_size, reduce='add')
            out2 = segment_coo(x, row, dim_size=dim_size, reduce='add')
            out3 = segment_csr(x, rowptr, reduce='add')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

            out1 = scatter(x, row, dim=0, dim_size=dim_size, reduce='mean')
            out2 = segment_coo(x, row, dim_size=dim_size, reduce='mean')
            out3 = segment_csr(x, rowptr, reduce='mean')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

            out1 = scatter(x, row, dim=0, dim_size=dim_size, reduce='min')
            out2 = segment_coo(x, row, reduce='min')
            out3 = segment_csr(x, rowptr, reduce='min')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

            out1 = scatter(x, row, dim=0, dim_size=dim_size, reduce='max')
            out2 = segment_coo(x, row, reduce='max')
            out3 = segment_csr(x, rowptr, reduce='max')

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
            torch.cuda.empty_cache()


def time_func(func, x):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()

        if not args.with_backward:
            with torch.no_grad():
                for _ in range(iters):
                    func(x)
        else:
            x = x.requires_grad_()
            for _ in range(iters):
                out = func(x)
                out = out[0] if isinstance(out, tuple) else out
                torch.autograd.grad(out, x, out, only_inputs=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t
    except RuntimeError as e:
        if 'out of memory' not in str(e):
            raise RuntimeError(e)
        torch.cuda.empty_cache()
        return float('inf')


def timing(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(args.device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(args.device, torch.long)
    row2 = row[torch.randperm(row.size(0))]
    dim_size = rowptr.size(0) - 1
    avg_row_len = row.size(0) / dim_size

    def sca_row(x):
        return scatter(x, row, dim=0, dim_size=dim_size, reduce=args.reduce)

    def sca_col(x):
        return scatter(x, row2, dim=0, dim_size=dim_size, reduce=args.reduce)

    def seg_coo(x):
        return segment_coo(x, row, reduce=args.reduce)

    def seg_csr(x):
        return segment_csr(x, rowptr, reduce=args.reduce)

    def dense1(x):
        return getattr(torch, args.reduce)(x, dim=-2)

    def dense2(x):
        return getattr(torch, args.reduce)(x, dim=-1)

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

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
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

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
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
                        choices=['sum', 'add', 'mean', 'min', 'max'])
    parser.add_argument('--with_backward', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    iters = 1 if args.device == 'cpu' else 20
    sizes = [1, 16, 32, 64, 128, 256, 512]
    sizes = sizes[:3] if args.device == 'cpu' else sizes

    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=args.device).sum()
    for dataset in itertools.chain(short_rows, long_rows):
        download(dataset)
        correctness(dataset)
        timing(dataset)
