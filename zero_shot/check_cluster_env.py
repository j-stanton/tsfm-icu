"""Print the Python and package versions available in the cluster environment."""

import importlib
import platform
import subprocess
import sys


def try_import(pkg_name: str, version_attr: str = "__version__"):
    try:
        mod = importlib.import_module(pkg_name)
        version = getattr(mod, version_attr, "unknown version")
        return True, version
    except ImportError:
        return False, None


def main():
    print("=" * 60)
    print("Cluster environment check for TSFM adapter (Phase 2)")
    print("=" * 60)

    print(f"\nPython: {sys.version}")
    print(f"Platform: {platform.platform()}")

    import os
    loaded_modules = os.environ.get("LOADEDMODULES", "(none detected)")
    print(f"\nLoaded modules: {loaded_modules}")

    print("\n--- Core packages needed by the adapter (yaib_data.py, zero_shot_pipeline.py, forecast_to_label.py) ---")
    for pkg in ["numpy", "pandas", "pyarrow", "sklearn"]:
        ok, version = try_import(pkg)
        print(f"  {pkg:12s} {'OK  ' if ok else 'MISSING'} {version or ''}")

    print("\n--- Torch / GPU ---")
    ok, version = try_import("torch")
    print(f"  torch        {'OK  ' if ok else 'MISSING'} {version or ''}")
    if ok:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version (torch built against): {torch.version.cuda}")
            print(f"  GPU device(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        else:
            print("  No CUDA GPU visible in this session.")
            print("  If you're inside the pytorch module (module load")
            print("  pytorch/pytorch-cuda12.9-ubuntu24.04-00) and still see")
            print("  False here, something has regressed since this was last")
            print("  confirmed working (2026-09-09) -- see SETUP_ENVIRONMENT.md's")
            print("  troubleshooting history before re-debugging from scratch.")
            print("  If you're using the plain `anaconda` module instead, this")
            print("  is EXPECTED -- that module does not pass --nv through to")
            print("  Apptainer and structurally cannot see the GPU. Use the")
            print("  pytorch module instead.")

    print("\n--- TimesFM itself ---")
    ok, version = try_import("timesfm")
    print(f"  timesfm      {'OK  ' if ok else 'MISSING'} {version or ''}")

    print("\n--- nvidia-smi (raw driver/CUDA info, if visible from this node) ---")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout[:600])
        else:
            print("  `nvidia-smi` ran but returned an error — no GPU visible "
                  "from this node/session (may need an interactive GPU "
                  "allocation, e.g. srun/salloc, rather than a login node).")
    except FileNotFoundError:
        print("  `nvidia-smi` not found on PATH from this node/session — "
              "either no GPU here, or it needs a module load / different "
              "node (check your cluster's documentation).")

    print("=" * 60)
    print("If numpy/pandas/pyarrow/scikit-learn/timesfm show MISSING above,")
    print("install them with pip3 (this module provides its own pip3")
    print("wrapper) -- see SETUP_ENVIRONMENT.md's 'Installing the remaining")
    print("packages' section.")
    print("=" * 60)


if __name__ == "__main__":
    main()
