import time
import itertools

import argparse
import torch
from scipy.io import loadmat

from torch_scatter import gather_coo, gather_csr

from scatter_segment import short_rows, long_rows, download, bold


@torch.no_grad()
def correctness(dataset):
    group, name = dataset
    mat = loadmat(f'{name}.mat')['Problem'][0][0][2].tocsr()
    rowptr = torch.from_numpy(mat.indptr).to(args.device, torch.long)
    row = torch.from_numpy(mat.tocoo().row).to(args.device, torch.long)
    dim_size = rowptr.size(0) - 1

    for size in sizes[1:]:
        try:
            x = torch.randn((dim_size, size), device=args.device)
            x = x.squeeze(-1) if size == 1 else x

            out1 = x.index_select(0, row)
            out2 = gather_coo(x, row)
            out3 = gather_csr(x, rowptr)

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
    dim_size = rowptr.size(0) - 1
    avg_row_len = row.size(0) / dim_size

    def select(x):
        return x.index_select(0, row)

    def gather(x):
        return x.gather(0, row.view(-1, 1).expand(-1, x.size(1)))

    def gat_coo(x):
        return gather_coo(x, row)

    def gat_csr(x):
        return gather_csr(x, rowptr)

    t1, t2, t3, t4 = [], [], [], []
    for size in sizes:
        try:
            x = torch.randn((dim_size, size), device=args.device)

            t1 += [time_func(select, x)]
            t2 += [time_func(gather, x)]
            t3 += [time_func(gat_coo, x)]
            t4 += [time_func(gat_csr, x)]

            del x

        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise RuntimeError(e)
            torch.cuda.empty_cache()
            for t in (t1, t2, t3, t4):
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
    parser = argparse.ArgumentParser()
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
