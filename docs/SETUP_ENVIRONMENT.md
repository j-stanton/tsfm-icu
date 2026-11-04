# Setting Up the TSFM Adapter Environment on mscluster

## TL;DR (confirmed working end-to-end, 2026-09-09)

```bash
srun --partition=stampede -w mscluster40 --pty bash   # must be inside an
                                                         # srun allocation on
                                                         # a compute node --
                                                         # the pytorch module
                                                         # is not available
                                                         # on the login node
module load pytorch/pytorch-cuda12.9-ubuntu24.04-00
conda-activate tsfm-adapter        # drops you into an `Apptainer>` shell
```

Inside that `Apptainer>` shell (this exact sequence, in this order --
`conda-activate`'s own hook script points at the wrong path, see
"Known issue" below):

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate tsfm-adapter
```

Prompt should now read `(tsfm-adapter) Apptainer>`. Verify:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.6.0+cu126 True
```

From here, `pip install`/`conda install` write into your OWN environment
at `~/.conda/envs/tsfm-adapter` (writable), not the shared read-only
`/opt/conda/envs/ai`. `timesfm` is already installed as of this doc's last
update -- confirmed via `pip install -e .` succeeding from
`~/tsfm-icu-early-warning/external/timesfm`.

## Why this path, not `conda run -n tsfm-adapter` from a plain shell

Earlier attempts (see "Full troubleshooting history" below) got torch+CUDA
working via the `anaconda` module + hand-built `tsfm-adapter` env, but
GPU access silently failed (`torch.cuda.is_available()` was always
`False`) because that module's Apptainer wrapper never requests `--nv`
passthrough. Switching to the `pytorch` module fixed CUDA access
(`--nv` is automatic there), but its base environment (`ai`, at
`/opt/conda/envs/ai`) is READ-ONLY -- baked into the container image, not
meant to be modified. `pip install timesfm` into it fails with
`OSError: Read-only file system`.

The fix: `conda create -n tsfm-adapter --clone ai` makes a full,
independent, WRITABLE copy of `ai` (torch/CUDA/pandas/etc. all already
present) under your own home directory
(`~/.conda/envs/tsfm-adapter`), leaving the shared `ai` env untouched for
every other user. This is the environment referenced everywhere else in
this doc and in `run_env_check.sbatch`.

## Known issue: `conda-activate` alone doesn't work

```
Apptainer> conda-activate tsfm-adapter
/usr/bin/bash: line 1: /opt/anaconda/etc/profile.d/conda.sh: No such file or directory
CondaError: Run 'conda init' before 'conda activate'
```

The module's `conda-activate` wrapper script has a hardcoded path
(`/opt/anaconda/etc/profile.d/conda.sh`) that doesn't match where conda
actually lives in this container image. Confirmed via `find / -name
"conda.sh" 2>/dev/null` **from inside the Apptainer shell** (this path
does not exist from the normal host shell -- it's inside the container's
own filesystem view):

```
/opt/conda/etc/profile.d/conda.sh
```

Workaround (source the real path manually, then activate) -- see the
TL;DR above. This needs to be redone every time you enter a fresh
`Apptainer>` shell; it does not persist, and running `source`/`conda
activate` from OUTSIDE the `Apptainer>` shell fails too, since that path
only resolves inside the container's filesystem view.

## Installing additional packages later

Same pattern: get into `(tsfm-adapter) Apptainer>` via the TL;DR sequence
above, then:

```bash
pip install <package>
# or
conda install -c conda-forge <package>
```

Both write into your own `~/.conda/envs/tsfm-adapter`, safely, without
touching the shared `ai` environment or affecting other users.

## Full troubleshooting history

Kept for reference in case any of this resurfaces. Skip this section for
normal day-to-day use -- the TL;DR above is the confirmed working path.

1. **Module name drift**: `readme.txt`'s documented
   `module load anaconda/2025-12-2` doesn't exist. Real name (confirmed
   via `module spider anaconda`): `anaconda/anaconda-2025-12-2`. Same
   pattern hit again with `hpc-examples/06-default-pytorch`'s
   `module load pytorch/2026-02-24` -- real name (confirmed via
   `module spider pytorch`): `pytorch/pytorch-cuda12.9-ubuntu24.04-00`.
   **Lesson**: always confirm exact module names with `module spider
   <name>` rather than trusting an example script's exact version string,
   since these drift as the cluster is updated.

2. **`pytorch-cuda=12.0` doesn't exist** as a conda package (the official
   PyTorch channel uses versioned tags like `12.1`/`12.4`, never a bare
   `12.0`). Fixed by using `pytorch-cuda=12.1` in the original hand-built
   conda env.

3. **`undefined symbol: iJIT_NotifyEvent`** on `import torch` in the
   hand-built env -- a known conda-ecosystem issue
   ([pytorch/pytorch#123097](https://github.com/pytorch/pytorch/issues/123097)):
   newer `mkl`/`intel-openmp` (2024.1+) is ABI-incompatible with that
   PyTorch build. Fixed with `conda install -y "mkl<2024.1"
   "intel-openmp<2024.1"`.

4. **`torch.cuda.is_available()` returned `False`** even after fixing (2)
   and (3), even on a node `nvidia-smi` confirmed had 2 working GPUs.
   Chased `CUDA_VISIBLE_DEVICES` (turned out unset, not the cause),
   Slurm GRES config (confirmed `Gres=(null)` cluster-wide on stampede --
   no GRES-based GPU scheduling exists on this cluster at all -- so
   `--gres=gpu:1` / `--gpus=1` both fail with "Invalid generic resource
   (gres) specification", also not the cause), and device file
   permissions (`/dev/nvidia*` were world read/write, not the issue).

5. **Root cause of (4)**: the `anaconda` module's Apptainer wrapper never
   passes `--nv` (confirmed by comparing its `module spider` help text,
   which says nothing about `--nv`, against the `pytorch` module's, which
   explicitly says "Automatically uses 'apptainer exec --nv' when NVIDIA
   devices are detected"). Structural limitation of that module, not
   fixable by anything inside the environment. A commented-out line in
   this project's own `hpc-examples/06-default-pytorch` example
   (`apptainer exec --nv ... pytorch-2026-02-24.sif ...`) was the hint
   that GPU work here needs `--nv`, sitting there the whole time.

6. **Switching to the `pytorch` module fixed GPU access**
   (`torch.cuda.is_available()` -> `True`, confirmed on `mscluster40`) but
   introduced the read-only-filesystem problem when installing `timesfm`
   into the module's own base `ai` environment -- see "Why this path"
   above for the `--clone` fix.

7. **`conda activate` failing with `CondaError: Run 'conda init' before
   'conda activate'`**, and `conda-activate`'s own wrapper pointing at a
   wrong path -- see "Known issue" above for the resolution.

**Meta-lesson for next time**: when a module is delivered via
Apptainer/Singularity on this cluster, check `module spider <name>`'s
help text FIRST for an explicit `--nv`-passthrough mention before
spending time debugging CUDA/driver/env-variable issues inside that
module's environment. If the module doesn't mention `--nv`, it
structurally cannot see the GPU, and no amount of package-level fixing
will change that.

