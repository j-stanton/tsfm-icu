# Setting up the TimesFM environment on mscluster

This is the setup that worked for me (September 2026).

## Getting into the environment

Start a session on a compute node. The `pytorch` module is not available on the login node.

```bash
srun --partition=stampede -w mscluster40 --pty bash
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00
conda-activate tsfm-adapter        # opens an `Apptainer>` shell
```

Inside the `Apptainer>` shell, run:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate tsfm-adapter
```

The prompt should now read `(tsfm-adapter) Apptainer>`. Check that the GPU is visible:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.6.0+cu126 True
```

You need to repeat the two `source` and `activate` lines every time you open a new `Apptainer>` shell.

`pip install` and `conda install` write to your own `~/.conda/envs/tsfm-adapter`. `timesfm` is installed from `external/timesfm` with `pip install -e .`.

## Why this setup

- The `anaconda` module never passes `--nv` to Apptainer, so `torch.cuda.is_available()` was always `False` there, even on nodes with working GPUs. The `pytorch` module passes it automatically.
- The `pytorch` module's own environment (`ai`) is read-only, so `pip install timesfm` fails with `Read-only file system`. The fix is `conda create -n tsfm-adapter --clone ai`, which makes a writable copy in your home directory.
- The module's `conda-activate` script points at `/opt/anaconda/etc/profile.d/conda.sh`, which does not exist. The real file is `/opt/conda/etc/profile.d/conda.sh`, and it is only visible from inside the `Apptainer>` shell. That is why you source it by hand.

## Other problems I hit

- **Module names change.** `anaconda/2025-12-2` did not exist. The real name was `anaconda/anaconda-2025-12-2`. Check with `module spider <name>`.
- **`pytorch-cuda=12.0` is not a conda package.** Use a versioned one such as `12.1`.
- **`undefined symbol: iJIT_NotifyEvent` on `import torch`.** Newer `mkl` and `intel-openmp` (2024.1 and later) don't work with that PyTorch build. Fix: `conda install -y "mkl<2024.1" "intel-openmp<2024.1"`. See [pytorch/pytorch#123097](https://github.com/pytorch/pytorch/issues/123097).
- **`--gres=gpu:1` fails** with "Invalid generic resource (gres) specification". Stampede has no GPU scheduling, so don't request GPUs through Slurm.
