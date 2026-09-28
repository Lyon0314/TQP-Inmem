# Reproducing the aggregation and materialization experiments

## Scope and tested machine

The experiments use Linux x86_64, an NVIDIA RTX 5080 (16303 MiB reported by
nvidia-smi), driver 590.48.01, and an Intel Xeon w5-2455X (12 cores / 24 threads).
Python was 3.12.14; GPU PyTorch was 2.14.0+cu130, with CUDA runtime 13.0.
The 13.1 shown by nvidia-smi is the driver's reported CUDA support, not the
PyTorch runtime. Runtime dependencies observed during setup were pandas 3.0.6,
pyspark 4.2.0, pyparsing 3.3.3 and pytest 9.1.1.

Base TQP commit: `934139d9c8abfe2dc982d64c2a390860133af90c`.
Data-generator commit: `83b538594b1af1e9a60b0fc9b715adf8cb09d23a`.
The submitted branch also includes an 8 GB decimal GPU pool in main.py instead
of the upstream 40 GB pool. All compared configurations use that same pool
and the same allocator sizing fix; neither is counted as an aggregation speedup.

## Fresh environment and data

Run these steps on the Linux GPU machine, not on a Mac. Conda must already be
installed. Use a common parent directory for TQP-Inmem and tpch-dbgen-tensors.
Check out the submitted TQP branch there using the repository/branch supplied
with the submission. Do not use the upstream base commit alone: it lacks the
changes and experiment scripts.

```bash
conda create -y -n tqp-review python=3.12.14 pip -c conda-forge
conda activate tqp-review
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install pandas==3.0.6 pyspark==4.2.0 pyparsing==3.3.3 pytest==9.1.1
```

The original cu126 instructions did not provide kernels for this RTX 5080 in
the installed build. This branch uses the tested cu130 build. Do not change a
shared machine's driver without administrator approval.

Build the data generator in a separate CPU environment. This avoids requiring
a CUDA compiler matching the GPU PyTorch runtime. From the common parent:

```bash
git clone https://github.com/hfhongzy/tpch-dbgen-tensors.git
git -C tpch-dbgen-tensors checkout 83b538594b1af1e9a60b0fc9b715adf8cb09d23a
conda create -y -n tpch-data-review python=3.12 pip cmake=3.31 ninja gcc_linux-64=11 gxx_linux-64=11 -c conda-forge
conda activate tpch-data-review
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
cd tpch-dbgen-tensors
mkdir -p data
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 2
cd build-cpu
./dbgen -s 1
cd ../../TQP-Inmem
conda activate tqp-review
```

The generator writes `../data/SF1-tensor-*.pth` relative to build-cpu.
config.json uses `../tpch-dbgen-tensors/data`, resolved from the TQP project
root. For a different layout, edit that setting. Run all following commands
from the TQP root. Existing generated data may be reused for code-only checkout
verification, but that is not a verification of fresh data generation.

```bash
nvidia-smi -L
# Replace GPU-UUID-HERE with the RTX 5080 UUID printed above.
export CUDA_VISIBLE_DEVICES=GPU-UUID-HERE
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.ones(3, device='cuda') * 2)"
mkdir -p results
```

The original machine's RTX UUID was GPU-47f261f4-a20b-afa9-e0ba-17bce50f44e7.
Do not assume GPU number 1 has the same meaning in CUDA and nvidia-smi.

## Correctness and switches

Both experiments default off. `TQP_AGG_REUSE=1` enables per-aggregation subgroup
index reuse; `TQP_PROJECT_NATIVE=1` preserves projection result dtypes through
the pool copy. Baseline means both switches are zero.

```bash
python research_checks.py correctness
python materialization_checks.py
for mode in 00 10 01 11
do
  TQP_AGG_REUSE=${mode%?} TQP_PROJECT_NATIVE=${mode#?} python main.py --test
done
```

The existing SF1 test checks formatted leading rows, not every output row.
The supplemental tests compare complete synthetic arrays but exclude full SQL
semantics and empty-group/input coverage. They must not be described as an
exhaustive correctness proof.

## Four-way runtime comparison

Use bash for this loop. It rotates configuration order over five rounds.
Warmup runs the entire SF1 suite twice in each process, not just the selected
queries. Profiling is disabled. The CSV timer subtracts scan and output time;
it is not process startup time or full end-to-end ingestion latency.

```bash
task_final_dir=$(mktemp -d results/final-compare-XXXXXX)
for run in 1 2 3 4 5
do
  case "$run" in
    1|5) modes="00 10 01 11" ;;
    2) modes="10 01 11 00" ;;
    3) modes="01 11 00 10" ;;
    4) modes="11 00 10 01" ;;
  esac
  for mode in $modes
  do
    if ! TQP_AGG_REUSE=${mode%?} TQP_PROJECT_NATIVE=${mode#?} python main.py --SF 1 --q 1 13 18 22 --warmup_iter 2 --no_debug --time_output_file "$task_final_dir/mode-$mode.csv" > "$task_final_dir/run-$run-mode-$mode.log" 2>&1
    then
      tail -n 20 "$task_final_dir/run-$run-mode-$mode.log"
      break 2
    fi
  done
done
```

Preserve CSVs and logs. Summarize medians and ranges; do not select only favorable
runs. Record `nvidia-smi` before/after experiments to help identify shared load.

## Profiling, memory and scaling

```bash
TQP_AGG_REUSE=0 TQP_PROJECT_NATIVE=0 python main.py --SF 1 --q 1 --no_debug --torch_profile=results/profile-original
TQP_AGG_REUSE=1 TQP_PROJECT_NATIVE=0 python main.py --SF 1 --q 1 --no_debug --torch_profile=results/profile-reuse
TQP_AGG_REUSE=0 TQP_PROJECT_NATIVE=0 python research_checks.py memory
TQP_AGG_REUSE=1 TQP_PROJECT_NATIVE=0 python research_checks.py memory
python aggregation_scaling_benchmark.py
python materialization_memory.py
```

Profiles add overhead and are diagnostic, not runtime comparison data. Profiler
allocation totals are not peak memory. Memory scripts report PyTorch allocated
memory including the 8 GB pool, not total driver/device usage. The scaling script
uses resident synthetic tensors, 3 warmups per mode and 8 alternating timed
invocations per mode; each invocation computes sum/avg twice. It times GPU work
with host clocks and explicit synchronization. Its row/group grid is not a set
of larger TPC-H scale factors.

## Capture the final environment

```bash
git rev-parse HEAD
git status --short
python --version
python -m pip freeze
conda list --explicit
nvidia-smi
lscpu
```

Record the final submitted commit alongside raw measurements. A new environment
may resolve different transitive dependencies unless the captured environment
is replayed. Fresh-checkout/environment verification must be recorded separately
from the original working-directory test results.
