# splits

One patient-level split, used by every model.

| File | What it is |
|---|---|
| `shared_split.csv` | `stay_id,split` for all 38,129 stays: train 26,662, validation 5,705, test 5,762 (about 70/15/15) |
| `apply_fixed_split.py` | Makes YAIB use this file instead of computing its own split |
| `fix_class_weight_bug.py` | Fixes the YAIB bug where scikit-learn models never get class weights |

The baselines and TimesFM use the same split, so they are all scored on the same test patients. No patient appears in more than one split.

## Using the patches

Apply them in the same Python process that runs YAIB:

```python
import sys; sys.path.insert(0, "splits")
import apply_fixed_split, fix_class_weight_bug
apply_fixed_split.patch(split_path="splits/shared_split.csv")
fix_class_weight_bug.patch()
from icu_benchmarks.run import main
main(my_args=[...])
```

This leaves `external/YAIB` unmodified. The catch is that the patches depend on YAIB's internal function names.
