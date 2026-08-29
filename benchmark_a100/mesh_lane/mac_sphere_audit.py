"""S4: is the near field inflated because the MAC radius is a BOX circumsphere
about the BOX CENTRE rather than a COM-centred particle radius?

Pure measurement. Builds the real shard-0 radix tree the distributed lane builds,
then re-runs the Dehnen accept test under five radius definitions and counts the
near list each would produce. No library change.
"""
import argparse, json, math, time
import numpy as np


def disc(n, radius=10.0, thickness=0.2, seed=9):
    """The harness's own IC (bench/distributed_ceiling_ladder.py::_disc)."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    th = rng.random(n) * 2.0 * np.pi
    z = (rng.random(n) - 0.5) * thickness
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=10_485_760)
    ap.add_argument("--ndev", type=int, default=5)
    ap.add_argument("--leaf", type=int, default=512)
    ap.add_argument("--thetas", default="0.4,0.7")
    ap.add_argument("--out", default="mac_sphere_audit.json")
    a = ap.parse_args()

    import jax, jax.numpy as jnp
    from jaccpot.distributed.fmm import partition_for_devices
    from yggdrax.tree import Tree
    from yggdrax.geometry import compute_tree_geometry

    pos = disc(a.n_total)
    mass = np.full(a.n_total, 1.0 / a.n_total, np.float32)
    t0 = time.perf_counter()
    part = partition_for_devices(pos, mass, a.ndev, leaf_size=a.leaf, partitioner="rcb")
    cap = part["cap"]
    print(f"# partitioned {a.n_total:,} -> {a.ndev} x cap {cap:,} in {time.perf_counter()-t0:.1f} s", flush=True)

    p0 = np.asarray(part["pos_flat"]).reshape(a.ndev, cap, 3)[0]
    m0 = np.asarray(part["mass_flat"]).reshape(a.ndev, cap)[0]

    tree = Tree.from_particles(jnp.asarray(p0), jnp.asarray(m0),
                              tree_type="radix", leaf_size=a.leaf)
    psort = np.asarray(tree.positions_sorted)
    msort = np.asarray(tree.masses_sorted)
    geom = compute_tree_geometry(tree, jnp.asarray(psort), max_leaf_size=a.leaf)
    print(f"# tree built: {int(tree.num_nodes)} nodes", flush=True)

    # leaves + the shipped radius
    from yggdrax._interactions_impl import get_leaf_nodes
    leaf_ids = np.asarray(get_leaf_nodes(tree))
    ranges = np.asarray(tree.node_ranges)[leaf_ids]
    centres_box = np.asarray(geom.center)[leaf_ids]
    r_box = np.asarray(geom.radius)[leaf_ids]

    nl = len(leaf_ids)
    print(f"# {nl} leaves", flush=True)

    com = np.zeros((nl, 3)); r_com = np.zeros(nl); r_true = np.zeros(nl)
    r_rms = np.zeros(nl); half_max = np.zeros(nl); dims = np.zeros((nl, 3))
    for i, (s, e) in enumerate(ranges):
        q = psort[s:e]
        w = msort[s:e]
        c = (q * w[:, None]).sum(0) / w.sum()
        com[i] = c
        d = np.linalg.norm(q - c, axis=1)
        r_com[i] = d.max(); r_rms[i] = np.sqrt((d * d).mean())
        r_true[i] = np.linalg.norm(q - centres_box[i], axis=1).max()
        lo, hi = q.min(0), q.max(0)
        dims[i] = hi - lo
        half_max[i] = 0.5 * float((hi - lo).max())

    defs = {"r_box(SHIPPED)": (centres_box, r_box), "r_true_boxcentre": (centres_box, r_true),
            "r_com": (com, r_com), "r_rms_com": (com, r_rms), "half_max_box": (centres_box, half_max)}

    out = {"n_total": a.n_total, "ndev": a.ndev, "leaf": a.leaf, "cap": int(cap),
           "num_leaves": int(nl),
           "leaf_dims_median": dims.mean(0).round(4).tolist(),
           "ratios": {"r_box/r_com_median": float(np.median(r_box / r_com)),
                      "r_box/r_true_median": float(np.median(r_box / r_true)),
                      "r_box/r_rms_median": float(np.median(r_box / r_rms))},
           "near_pairs": {}}
    print(f"# median leaf dims {dims.mean(0).round(4)}  r_box/r_com median "
          f"{np.median(r_box/r_com):.3f}", flush=True)

    # all-pairs Dehnen accept test on GPU, per radius definition
    for th in [float(x) for x in a.thetas.split(",")]:
        out["near_pairs"][str(th)] = {}
        for name, (c, r) in defs.items():
            C = jnp.asarray(np.asarray(c, np.float64))
            R = jnp.asarray(np.asarray(r, np.float64))
            d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            rs = (R[:, None] + R[None, :]) ** 2
            near = np.asarray(jnp.sum(rs > (th ** 2) * d2))  # MAC FAILS -> near
            out["near_pairs"][str(th)][name] = int(near)
            print(f"  theta {th}  {name:22s} near leaf pairs {int(near):>12,}"
                  f"   ({near/nl:7.1f}/leaf)", flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
