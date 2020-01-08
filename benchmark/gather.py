import time
import itertools

import torch
from scipy.io import loadmat

from torch_scatter import gather_coo, gather_csr

from scatter_segment import iters, device, sizes
from scatter_segment import short_rows, long_rows, download, bold


@torch.no_grad()
def correctness(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(device, torch.long)
    dim_size = rowptr.size(0) - 1

    for size in sizes[1:]:
        try:
            x = torch.randn((dim_size, size), device=device)
            x = x.squeeze(-1) if size == 1 else x

            out1 = x.index_select(0, row)
            out2 = gather_coo(x, row)
            out3 = gather_csr(x, rowptr)

            assert torch.allclose(out1, out2, atol=1e-4)
            assert torch.allclose(out1, out3, atol=1e-4)
        except RuntimeError:
            torch.cuda.empty_cache()


@torch.no_grad()
def timing(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(device, torch.long)
    dim_size = rowptr.size(0) - 1
    avg_row_len = row.size(0) / dim_size

    t1, t2, t3, t4 = [], [], [], []
    for size in sizes:
        try:
            x = torch.randn((dim_size, size), device=device)
            row_expand = row.view(-1, 1).expand(-1, x.size(-1))
            x = x.squeeze(-1) if size == 1 else x
            row_expand = row_expand.squeeze(-1) if size == 1 else row_expand

            try:
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    out = x.index_select(0, row)
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
                    out = x.gather(0, row_expand)
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
                    out = gather_coo(x, row)
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
                    out = gather_csr(x, rowptr)
                    del out
                torch.cuda.synchronize()
                t4.append(time.perf_counter() - t)
            except RuntimeError:
                torch.cuda.empty_cache()
                t4.append(float('inf'))

            del x

        except RuntimeError:
            torch.cuda.empty_cache()
            for t in (t1, t2, t3):
                t.append(float('inf'))

    ts = torch.tensor([t1, t2, t3, t4])
    winner = torch.zeros_like(ts, dtype=torch.bool)
    winner[ts.argmin(dim=0), torch.arange(len(sizes))] = 1
    winner = winner.tolist()

    name = f'{group}/{name}'
    print(f'{bold(name)} (avg row length: {avg_row_len:.2f}):')
    print('\t'.join(['       '] + [f'{size:>5}' for size in sizes]))
    print('\t'.join([bold('SELECT ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t1, winner[0])]))
    print('\t'.join([bold('GAT    ')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t2, winner[1])]))
    print('\t'.join([bold('GAT_COO')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t3, winner[2])]))
    print('\t'.join([bold('GAT_CSR')] +
                    [bold(f'{t:.5f}', f) for t, f in zip(t4, winner[3])]))
    print()


if __name__ == '__main__':
    for _ in range(10):  # Warmup.
        torch.randn(100, 100, device=device).sum()
    for dataset in itertools.chain(short_rows, long_rows):
        download(dataset)
        correctness(dataset)
        timing(dataset)
