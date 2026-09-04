"""FORWARD-ONLY churn repro for jax.lax.ragged_all_to_all under jit(shard_map).

jaccpot's bench/repro_jax_ragged_all_to_all_grad.py triggers XLA:GPU's stale
peer-address cache (RaggedAllToAllStartThunk::Initialize caches output-buffer
addresses from the FIRST execution; fixed in XLA 4e0cc7e356 / jax 0.9.1) with a
gradient. This one uses NO gradient: fresh input buffers each call, optional
donation, and -- the churn -- live junk allocations of varying size held across
the call so the executable's temporary buffer lands at a different address.
That is what a donating KDK loop does to the allocator. One config per process
(the upstream caveat: cases perturb each other within a process).
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

NDEV, CAP, FILL = 2, 4, -1.0
EXPECTED = np.array([1., 2., 5., 6., 3., 4., 7., 8.])

def build(donate: bool):
    mesh = Mesh(np.array(jax.devices()[:NDEV]), ("gpus",))
    def body(x, sizes, in_off, out_off, rec):
        out = jnp.full((CAP,), FILL, x.dtype)
        return jax.lax.ragged_all_to_all(x, out, in_off[0], sizes[0], out_off[0], rec[0], axis_name="gpus")
    sm = shard_map(body, mesh=mesh, in_specs=(P("gpus"),) * 5, out_specs=P("gpus"), check_vma=False)
    return jax.jit(sm, donate_argnums=(0,) if donate else ())

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--donate", action="store_true")
    ap.add_argument("--churn", action="store_true", help="hold varying-size junk alive across each call")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if len(jax.devices()) < NDEV:
        raise SystemExit(f"needs >= {NDEV} devices, found {len(jax.devices())}")
    run = build(a.donate)
    sizes = jnp.asarray(np.array([[2, 2], [2, 2]], np.int32))
    in_off = jnp.asarray(np.array([[0, 2], [0, 2]], np.int32))
    out_off = jnp.asarray(np.array([[0, 0], [2, 2]], np.int32))
    rec = jnp.asarray(np.array([[2, 2], [2, 2]], np.int32))
    base = np.arange(1.0, NDEV * CAP + 1.0)
    rng = np.random.default_rng(a.seed)
    bad, pattern = 0, []
    for i in range(a.iters):
        junk = None
        if a.churn:
            junk = [jnp.ones((int(rng.integers(1 << 12, 1 << 24)),), jnp.float64) for _ in range(3)]
            jax.block_until_ready(junk)
        x = jnp.asarray(base.copy())          # a fresh device buffer every call
        y = np.asarray(run(x, sizes, in_off, out_off, rec))
        ok = np.array_equal(y, EXPECTED)
        pattern.append("." if ok else "X")
        if not ok:
            bad += 1
            if bad <= 3:
                print(f"  iter {i:3d}: {y}")
        del junk
    print(f"jax {jax.__version__}  donate={a.donate} churn={a.churn}  iters={a.iters}")
    print("  " + "".join(pattern))
    print(f"  -> {'CORRUPT' if bad else 'CLEAN'} ({bad}/{a.iters} calls returned the fill value or garbage)")
    raise SystemExit(1 if bad else 0)

if __name__ == "__main__":
    main()
