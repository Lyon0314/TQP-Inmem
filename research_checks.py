"""Focused aggregation correctness checks and whole-query GPU memory checks."""
import argparse
import gc
import os
from types import SimpleNamespace

import torch


def correctness():
    from operators.aggregate import evaluate
    from parsing import get_id
    import constants

    assert torch.cuda.is_available()
    device = 'cuda'
    key = get_id('x#1')
    cases = [(32, 4, False), (131070, 2, False),
             (131072, 2, False), (262144, 2, True), (131072, 1, False)]
    comparisons = 0
    with torch.inference_mode():
        for rows, groups, skew in cases:
            ids_cpu = torch.arange(rows, dtype=torch.int64) % groups
            if skew:
                ids_cpu[:] = 0
                ids_cpu[::97] = 1
            ids = ids_cpu.to(device)
            for dtype in (torch.int64, torch.float64):
                values_cpu = (torch.arange(rows) % 17 + 1).to(dtype)
                if dtype == torch.float64:
                    values_cpu /= 4
                values = values_cpu.to(device)
                columns = {key: SimpleNamespace(tensor=values)}
                cache = {}
                for repetition in range(2):
                    for operation in ('sum', 'avg', 'count', 'min', 'max'):
                        tree = [operation, ['x#1']]
                        original = evaluate(device, columns, tree, rows, groups, ids, None)
                        reused = evaluate(device, columns, tree, rows, groups, ids, cache)
                        expected = []
                        for group in range(groups):
                            selected = values_cpu[ids_cpu == group]
                            if operation == 'sum':
                                answer = selected.sum().item()
                            elif operation == 'avg':
                                answer = selected.double().mean().item()
                            elif operation == 'count':
                                answer = selected.numel()
                            elif operation == 'min':
                                answer = selected.min().item()
                            else:
                                answer = selected.max().item()
                            expected.append(answer)
                        reference = torch.tensor(expected, dtype=reused.dtype, device=device)
                        torch.testing.assert_close(reused, original, rtol=1e-10, atol=1e-10)
                        torch.testing.assert_close(reused, reference, rtol=1e-10, atol=1e-10)
                        comparisons += 1
                expected_entries = 1 if rows / groups >= 65536 else 0
                assert len(cache) == expected_entries, (rows, groups, list(cache))
                # Changed values with unchanged grouping must NOT reuse old answers.
                changed = values.clone()
                changed[::7] = constants.null
                columns[key].tensor = changed
                result = evaluate(device, columns, ['count', ['x#1']], rows, groups, ids, cache)
                expected = torch.tensor([
                    ((changed[ids == group]) != constants.null).sum().item()
                    for group in range(groups)
                ], dtype=result.dtype, device=device)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                comparisons += 1
            print(f'PASS rows={rows}, groups={groups}, skew={skew}', flush=True)
    print(f'PASS: {comparisons} full-vector checks; integer/float, threshold, skew, null counts, cache reuse.')
    print('Scope: nonempty groups, positive inputs; not a complete SQL semantics test.')


def memory():
    import main as tqp
    print('Reuse:', os.environ.get('TQP_AGG_REUSE', '0'), flush=True)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    records = []
    with torch.inference_mode():
        # Match the existing two whole-suite warmups, outside memory measurement.
        tqp.warmup()
        tqp.warmup()
        for repetition in range(1, 4):
            for qid in (1, 13, 18):
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                start = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                tqp.exec(1, [qid], None, None, True, None, None, None)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                records.append((repetition, qid, start, peak, peak-start, reserved))
    print('MEMORY_RESULTS_BYTES', flush=True)
    print('run,qid,start_allocated,peak_allocated,extra_above_start,peak_reserved')
    for record in records:
        print(','.join(map(str, record)))
    print('Includes the preallocated GPU pool; PyTorch allocator figures, not total device usage.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('check', choices=('correctness', 'memory'))
    args = parser.parse_args()
    if args.check == 'correctness':
        correctness()
    else:
        memory()
