# jaccpot: the FORWARD LET halo exchange is unguarded against the jax < 0.9.1 `ragged_all_to_all` defect

**Where:** `jaccpot/distributed/fmm.py::_grad_halo_exchange` / `resolve_grad_halo_exchange`;
`yggdrax/distributed/let.py` → `comm.ragged_all_to_all_exchange(method="auto")` (→ `"native"` on GPU).

**Claim in the code today:** the stale-peer-address defect in XLA's `RaggedAllToAllStartThunk`
(fixed in jax 0.9.1) is triggered by *executing a gradient*, so only the gradient path is pinned to
`"buf"` and "the forward keeps the native path".

**Measured (2026-09-04, jax/jaxlib 0.9.0, 2×A100, `ragged_forward_churn_repro.py` — the upstream
repro with the gradient removed):** the trigger is *any* movement of the exchange's buffers, and
**input donation alone is sufficient**:

| forward-only config, 40 calls | corrupt calls |
|---|---|
| `jax.jit(..., donate_argnums=(0,))`, fresh input each call | **36/40** (first call clean) |
| donation + live junk allocations of varying size across the call | **35/40** |
| no donation, junk allocations only | **3/40** (intermittent) |
| no donation, no churn, identical buffers | 0/40 |

Corrupt calls return the output buffer's fill value on one device (`[-1 -1 -1 -1 3 4 7 8]`).

**Effect on the distributed FMM:** a KDK loop that donates its state (any sane integrator) gets
`halo_posm = 0` / `halo_gid = -1` on most steps after the first, i.e. the cross-domain near field is
silently dropped: rel-L2 0.45–0.50 against an fp64 direct sum on a 17.8M-particle disc+bulge on 4
GPUs, for both the geometric and the mass-dependent MAC, while angular momentum, KE, COM and every
overflow flag stay clean (the dropped pairs are mutual). Intermittently the stale writes land in
memory now owned by another buffer → NaN (observed once in 10 steps, then clean for 14).
Identical-input reproducibility tests pass (row 4), so they cannot catch it.

**Requested fix:**
1. Gate the forward `"auto"` the same way as the gradient path: on GPU with jax < 0.9.1 resolve to
   `"buf"`, or refuse to run the distributed forward with `"native"` unless explicitly requested.
   `resolve_grad_halo_exchange` already encodes the version boundary; rename it and apply it in
   `make_force_evaluator` before yggdrax builds the LET.
2. Expose the forward exchange choice in `DistributedFMMConfig` (currently `halo_exchange` is
   documented "differentiable mode only").
3. Add a moving-position test: evaluate the distributed force in a donating loop on ≥2 devices for
   ≥3 steps and compare EACH step against a direct sum — not the same inputs twice.
4. Docs: `_grad_halo_exchange`'s "the forward keeps the native path" is wrong on affected JAX.

Attached: `ragged_forward_churn_repro.py`, `ragged_forward_churn_repro.log`, and (pending) the
full-pipeline confirmation with the forward pinned to `"buf"` and the jax 0.9.1 control.
