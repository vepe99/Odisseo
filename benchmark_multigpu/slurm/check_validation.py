"""Gate the HoreKa sweep on its validation rows; exit non-zero on any failure.

* every row ran and no capacity flag fired;
* 2 and 4 cards: the cross error at p4/5/6 decreases with order (the coverage check --
  a missing piece of the cross field does not improve with p) and at p6 stays within
  1.25x of the one-card error measured IN THE SAME JOB (on the development box: 6.77e-4
  on one card, 6.95e-4 on two, at N = 2e5, theta 0.8, leaf 64).
"""

import json
import os
import sys

out = sys.argv[1]
fail = []


def row(tag):
    path = os.path.join(out, f"{tag}.json")
    if not os.path.exists(path):
        fail.append(f"{tag}: no row")
        return None
    r = json.load(open(path))
    if r.get("overflow_local") or r.get("overflow_cross"):
        fail.append(f"{tag}: a capacity flag fired")
    return r


one = row("val_1card_p6")
for ndev in (2, 4):
    errs = []
    for p in (4, 5, 6):
        r = row(f"val_{ndev}card_p{p}")
        errs.append(None if r is None else r.get("err_cross"))
    if None in errs:
        continue
    if not (errs[0] > errs[1] > errs[2]):
        fail.append(f"{ndev} cards: error not monotone in p {errs}")
    if one and one.get("err_local") and errs[2] > 1.25 * one["err_local"]:
        fail.append(
            f"{ndev} cards: p6 error {errs[2]:.3e} above 1.25x the one-card "
            f"{one['err_local']:.3e}"
        )
    print(f"{ndev} cards: errors p4/5/6 = {errs}")
if fail:
    print("\n".join(fail))
    sys.exit(1)
print("validation passed")
