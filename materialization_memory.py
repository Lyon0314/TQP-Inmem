"""Q22 whole-query PyTorch allocated-memory comparison, aggregation reuse off."""
import gc
import os
import torch
import main

if __name__ == '__main__':
    os.environ['TQP_AGG_REUSE'] = '0'
    records = []
    with torch.inference_mode():
        for mode in ('0', '1'):
            os.environ['TQP_PROJECT_NATIVE'] = mode
            main.warmup()
            main.warmup()
            for run in range(1, 4):
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                main.exec(1, [22], None, None, True, None, None, None)
                torch.cuda.synchronize()
                records.append((mode, run, torch.cuda.max_memory_allocated()))
    print('mode,run,peak_allocated_bytes')
    for record in records:
        print(','.join(map(str, record)))
