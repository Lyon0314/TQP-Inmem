"""Exercise actual projection and pool-copy paths on small GPU inputs."""
import os
import torch
from IO.pinned_mem import GPUMemory
from operators.project import tqp_project
from parsing import get_id
from variable import Variable, torch_to_type
from utility.logger import set_perf_logger, set_message_logger


def main():
    set_perf_logger(None, True)
    set_message_logger(None, False)
    assert torch.cuda.is_available(), 'CUDA is required'
    pool = GPUMemory(capacity_gb=0.016, block_size_mb=1)
    checks = 0
    with torch.inference_mode():
        # The layout that previously failed: two columns sliced from wider rows.
        full = torch.arange(120, dtype=torch.int8, device='cuda').reshape(20, 6)
        sliced = full[:, :2]
        assert not sliced.is_contiguous()
        _, storage = pool.malloc_like(sliced)
        storage[:] = sliced
        torch.testing.assert_close(storage, sliced, rtol=0, atol=0)
        assert storage.dtype == sliced.dtype
        assert storage.data_ptr() != sliced.data_ptr()
        checks += 1
        pool.free_all()

        source_id, output_id = get_id('x#1'), get_id('out#2')

        def project(value, expression, expected, native):
            nonlocal checks
            os.environ['TQP_PROJECT_NATIVE'] = str(native)
            columns = {source_id: Variable(value, torch_to_type(value.dtype))}
            args = {
                'rename map': {},
                'used expr': [[expression, 'AS', 'out#2']],
                'materialized in': [source_id],
                'materialized out': [output_id],
            }
            tqp_project(True, columns, args, None, pool)
            actual = columns[output_id].tensor
            assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.cuda.synchronize()
            checks += 1
            del columns, actual
            pool.free_all()

        integers = torch.arange(30, dtype=torch.int64, device='cuda')[::2]
        decimals = integers.double() / 4
        cases = [
            ('substring slice', full, ['substring', ['x#1', '1', '2']], full[:, :2]),
            ('integer addition', integers, ['x#1', '+', '1'], integers + 1),
            ('decimal multiplication', decimals, ['x#1', '*', '2'], decimals * 2),
            ('boolean comparison', integers, ['x#1', '>', '10'], integers > 10),
        ]
        for name, value, expression, expected in cases:
            for native in (0, 1):
                project(value, expression, expected, native)
            print('PASS original and native:', name, flush=True)

        # Native mode should preserve integers beyond float64's exact range.
        large = torch.tensor([2**53 + 1, 2**53 + 3], dtype=torch.int64, device='cuda')
        project(large, ['x#1', '+', '0'], large, 1)
        print('PASS native: large-integer precision', flush=True)
    print(f'PASS: {checks} full-array checks, including non-contiguous allocation/copy.')
    print('Scope: nonempty projection results; not exhaustive SQL or empty-input coverage.')


if __name__ == '__main__':
    main()
