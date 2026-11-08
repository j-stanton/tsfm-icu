# splits — one patient-level split for every model

| File | Purpose |
|---|---|
| `shared_split.csv` | `stay_id,split` for all 38,129 stays: train 26,662 / val 5,705 / test 5,762 (about 70/15/15) |
| `apply_fixed_split.py` | Monkeypatches YAIB's `make_single_split_polars` to use this file instead of computing its own split |
| `fix_class_weight_bug.py` | Monkeypatch for the YAIB bug where scikit-learn models never receive balanced class weights |

The same split is used by the baselines and by TimesFM, so all models are evaluated on identical test patients. Verified to have zero patient overlap across splits.

## Usage

Apply the patches before YAIB runs, in the same Python process:

```python
import sys; sys.path.insert(0, "splits")
import apply_fixed_split, fix_class_weight_bug
apply_fixed_split.patch(split_path="splits/shared_split.csv")
fix_class_weight_bug.patch()
from icu_benchmarks.run import main
main(my_args=[...])
```

Patching from this repo keeps `external/YAIB` an unmodified, upstream-trackable submodule. The trade-off is that it depends on YAIB's internal function signatures.