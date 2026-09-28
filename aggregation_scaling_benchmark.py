"""Synthetic aggregation scaling benchmark, NOT the projection test."""
import statistics
import time
from types import SimpleNamespace
import torch
from operators.aggregate import evaluate
from parsing import get_id


def main():
    print('Synthetic aggregation evaluator; GPU:', torch.cuda.get_device_name(0), flush=True)
    print('rows,groups,two_step,original_median_ms,reuse_median_ms,original_min_ms,original_max_ms,reuse_min_ms,reuse_max_ms', flush=True)
    with torch.inference_mode():
        for rows in (65536, 262144, 1048576):
            for groups in (4, 64):
                ids = torch.arange(rows, device='cuda', dtype=torch.int64) % groups
                values = (torch.arange(rows, device='cuda', dtype=torch.int64) % 17 + 1).double()
                columns = {get_id('x#1'): SimpleNamespace(tensor=values)}

                def run(mode):
                    cache = {} if mode else None
                    return [evaluate('cuda', columns, [op, ['x#1']], rows, groups, ids, cache)
                            for op in ('sum', 'avg', 'sum', 'avg')]

                expected_sum = torch.stack([values[ids == g].sum() for g in range(groups)])
                expected_mean = expected_sum / (rows // groups)
                for mode in (0, 1):
                    output = run(mode)
                    for actual, expected in zip(output, (expected_sum, expected_mean, expected_sum, expected_mean)):
                        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                    del output, actual, expected
                    for _ in range(3):
                        output = run(mode)
                        torch.cuda.synchronize()
                        del output
                samples = {0: [], 1: []}
                for repeat in range(8):
                    for mode in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = run(mode)
                        torch.cuda.synchronize()
                        samples[mode].append((time.perf_counter() - start) * 1000)
                        del output
                record = [rows, groups, int(rows / groups >= 65536)]
                record += [round(statistics.median(samples[m]), 6) for m in (0, 1)]
                for mode in (0, 1):
                    record += [round(min(samples[mode]), 6), round(max(samples[mode]), 6)]
                print(','.join(map(str, record)), flush=True)
                del columns, ids, values, expected_sum, expected_mean
    print('PASS: all synthetic results matched reference sums/means.', flush=True)
    print('Synthetic GPU-resident inputs; synchronized host timings; not TPC-H scale factors.', flush=True)


if __name__ == '__main__':
    main()
