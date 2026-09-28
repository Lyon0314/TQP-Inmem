# Setup

Environment setup for TQP-Inmem. Once these steps are done, see `README.md` for how to run the system.

Choose your favorite Python environment manager — we use Conda as an example.

## Conda environment

Create a Conda environment and install torch.

```
conda create -y -n torch126 python=3.12 cmake ninja gcc_linux-64=11 gxx_linux-64=11 cuda-toolkit=12.6 -c conda-forge
conda activate torch126
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The `cmake`, `ninja`, `gcc`, and `cuda-toolkit` packages are not needed by TQP-Inmem itself (this fork is pure Python on top of PyTorch), but they are required by the TPC-H dbgen step below.

## TPCH-DBGEN with Tensors

We wrote a torch-native dbgen for generating TPC-H data linked here: https://github.com/hfhongzy/tpch-dbgen-tensors/tree/main

Clone the repo and use the following instructions to configure the tpch-dbgen repo under its root directory.

```
mkdir -p build/ data/
cd build/
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux
make -j
./dbgen -s 1
```

Substitute the `-s` flag with dbgen with the desired scale factor to generate the corresponding set of data tables in tensor form. The tensors will be generated under `./data` from the tpch-dbgen repo home directory.

## Install Python Dependencies

From the TQP-Inmem home directory:

```
pip install -r requirements.txt
```

## Point config.json at the data

Edit `config.json` so `tensors path` points to the `data/` directory produced by tpch-dbgen above:

```
"tensors path": "/path/to/tpch-dbgen-tensors/data"
```


## RTX 5080 research branch

For this branch on the tested RTX 5080, follow [REPRODUCE.md](REPRODUCE.md)
for the cu130 GPU environment, separate CPU data-generator environment,
relative data path, correctness checks and four-way experiments. The original
cu126 commands above did not support this GPU in the tested installation.
The GPU pool in this branch is 8 GB, not the upstream 40 GB.
