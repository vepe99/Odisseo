"""HOST-ONLY. Strays: particles owned by device d (frozen RCB at the IC) that have drifted
outside d's box by step k. Per step: how many, on which device, how far out. If a repro json
with per-target errors exists for that step, join: are wrong targets on devices with strays?

A frozen partition meeting drift is the one thing the identical-input test (clean) and the
moving-position test (wrong) differ on structurally. A handful of strays cannot move a median
directly -- but if a stray corrupts its device's coarse-tree extents, the cross-domain far field
is wrong for EVERY particle on that device, which can.

Usage: stray_analysis.py <snapshot_stepK.npz> [repro_json]
"""
import sys, json, numpy as np
sys.path.insert(0, "/export/home/tbuck/Odisseo")
IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_17m8.npz"; NDEV, LEAF = 4, 1024
from odisseo import mesh_coupling
snap = np.load(sys.argv[1]); step = int(snap["step"]) if "step" in snap.files else -1
Xk = np.asarray(snap["state0"])[:, 0, :].astype(np.float64)
ic = np.load(IC); X0 = np.asarray(ic["state0"])[:, 0, :].astype(np.float64); mass = np.asarray(ic["mass"])
part = mesh_coupling.build_mesh_partition(X0.astype(np.float32), mass, ndev=NDEV, leaf_size=LEAF, partitioner="rcb")
cap = part.cap; order = np.asarray(part.order_ix)[: len(mass)]          # row -> original index
owner_of_row = np.arange(len(order)) // cap
print(f"step {step}: N={len(mass):,} cap={cap:,} ndev={NDEV}")
print(f"  displacement since IC: median {np.median(np.linalg.norm(Xk-X0,axis=1)):.3e}  max {np.linalg.norm(Xk-X0,axis=1).max():.3e}")
boxes = []
for d in range(NDEV):
    rows = np.arange(d*cap, (d+1)*cap); idx = order[rows]
    lo0, hi0 = X0[idx].min(0), X0[idx].max(0)          # the device's box, at partition time
    xk = Xk[idx]
    out = np.any((xk < lo0) | (xk > hi0), axis=1)
    dist = np.maximum(np.maximum(lo0 - xk, xk - hi0), 0).max(axis=1)
    boxes.append((lo0, hi0))
    print(f"  dev {d}: box {np.round(lo0,3)}..{np.round(hi0,3)}  strays {int(out.sum()):,} ({100*out.mean():.4f} %)  "
          f"max excursion {dist.max():.3e}  n(excursion>1e-3)={int((dist>1e-3).sum())}")
if len(sys.argv) > 2:
    r = json.load(open(sys.argv[2])); c0 = r["calls"][0]
    per = np.array(c0["per_target_rel"]); dev = np.array(c0["target_device"])
    print(f"\n  repro call 0 rel_l2={c0['rel_l2']:.4e}; per-target rel error by device:")
    for d in range(NDEV):
        m = dev == d
        if m.any(): print(f"    dev {d}: n={m.sum():3d}  median {np.median(per[m]):.3e}  p90 {np.percentile(per[m],90):.3e}  frac>0.1: {np.mean(per[m]>0.1):.3f}")
