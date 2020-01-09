import time
import os.path as osp
import itertools

import wget
import torch
from scipy.io import loadmat

from torch_scatter import scatter_add, scatter_mean, scatter_min, scatter_max
from torch_scatter import segment_coo, segment_csr

iters = 20
device = 'cuda'
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
    rowptr = torch.from_numpy(mat.indptr).to(device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(device, torch.long)
    dim_size = rowptr.size(0) - 1

    for size in sizes:
        try:
            x = torch.randn((row.size(0), size), device=device)
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

            # out1, arg_out1 = scatter_max(x, row, dim=0, dim_size=dim_size)
            # out3, arg_out3 = segment_csr(x, rowptr, reduce='max')

            # print(out1[:5])
            # print(out3[:5])

            # nnz = (out1 != out3).nonzero().flatten()

            # nnz1 = nnz[0].item()
            # print(rowptr[nnz1], rowptr[nnz1 + 1])

            # print(x[rowptr[nnz1]:rowptr[nnz1 + 1]])
            # print(x[rowptr[nnz1]:rowptr[nnz1 + 1]])

            # print(out1[nnz1])
            # print(out3[nnz1])

            # assert torch.allclose(out1, out3, atol=1e-4)
            # assert torch.all(arg_out1 == arg_out3)
        except RuntimeError:
            torch.cuda.empty_cache()


@torch.no_grad()
def timing(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(device, torch.long)
    row_perm = row[torch.randperm(row.size(0))]
    dim_size = rowptr.size(0) - 1
    avg_row_len = row.size(0) / dim_size

    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []
    for size in sizes:
        try:
            x = torch.randn((row.size(0), size), device=device)
            x = x.squeeze(-1) if size == 1 else x

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = scatter_add(x, row, dim=0, dim_size=dim_size)
                    del out
                torch.cuda.synchronize()
                t1.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t1.append(float('inf'))

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = scatter_add(x, row_perm, dim=0, dim_size=dim_size)
                    del out
                torch.cuda.synchronize()
                t2.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t2.append(float('inf'))

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = segment_coo(x, row, dim_size=dim_size, reduce='any')
                    del out
                torch.cuda.synchronize()
                t3.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t3.append(float('inf'))

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = segment_csr(x, rowptr, reduce='any')
                    del out
                torch.cuda.synchronize()
                t4.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t4.append(float('inf'))

            del x

        except RuntimeError:
            torch.cuda.empty_cache()
            for t in (t1, t2, t3, t4):
                t.append(float('inf'))

        try:
            x = torch.randn((dim_size, int(avg_row_len + 1), size),
                            device=device)
            x = x.squeeze(-1) if size == 1 else x

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = x.sum(dim=1)
                    del out
                torch.cuda.synchronize()
                t5.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t5.append(float('inf'))

            x = x.view(dim_size, size, int(avg_row_len + 1))
            x = x.squeeze(-2) if size == 1 else x

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = x.sum(dim=-1)
                    del out
                torch.cuda.synchronize()
                t6.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t6.append(float('inf'))

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
    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=device).sum()
    for dataset in itertools.chain(short_rows, long_rows):
        download(dataset)
        correctness(dataset)
        timing(dataset)
