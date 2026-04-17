# jaccpot traced-mode leaf-cap performance patch (proposal)

## Problem
When `prepare_state(...)` is called under an outer `jax.jit` (as in ODISSEO's `build_jitted_leapfrog_jaccpot_active` path), jaccpot disables stateful cache and currently inflates leaf-cap hints to `N` particles.

Relevant current code:
- `/Users/buck/Documents/Nexus/Projects/jaccpot/jaccpot/runtime/_fmm_impl.py:1053-1057`
- `/Users/buck/Documents/Nexus/Projects/jaccpot/jaccpot/runtime/_fmm_impl.py:1079`

This can make nearfield work scale badly because chunk sizing depends on `max_leaf_size`:
- `/Users/buck/Documents/Nexus/Projects/jaccpot/jaccpot/runtime/_fmm_impl.py:3310`

## Effect
Prepared-state evaluation can be fast in eager mode, but the same workload becomes much slower when wrapped by outer `jit` because traced mode uses an overly conservative `leaf_cap`.

## Minimal safe patch
Prefer `fixed_max_leaf_size` when available in traced mode, and propagate the resolved cap into prepared state metadata.

```diff
diff --git a/jaccpot/runtime/_fmm_impl.py b/jaccpot/runtime/_fmm_impl.py
@@ -1051,11 +1051,20 @@ class FastMultipoleMethod:
-        # Use the true tree leaf cap in eager mode for performance. Under outer
-        # tracing/JIT we still need a conservative static upper bound.
-        leaf_cap_hint = (
-            int(build_artifacts.max_leaf_size)
-            if allow_stateful_cache
-            else int(positions_arr.shape[0])
-        )
+        # Use true cap in eager mode. Under traced/outer-jit mode, prefer
+        # configured fixed_max_leaf_size (if present) to avoid pathological N-sized
+        # nearfield buffers/chunks. Fall back to N only when no fixed cap exists.
+        if allow_stateful_cache:
+            leaf_cap_hint = int(build_artifacts.max_leaf_size)
+        elif self.fixed_max_leaf_size is not None:
+            leaf_cap_hint = int(self.fixed_max_leaf_size)
+        else:
+            leaf_cap_hint = int(positions_arr.shape[0])
@@ -1076,7 +1085,7 @@ class FastMultipoleMethod:
         return _PrepareStateTreeUpwardArtifacts(
             tree_mode=tree_config.mode,
             tree=tree,
             positions_sorted=pos_sorted,
             masses_sorted=mass_sorted,
             inverse_permutation=build_artifacts.inverse_permutation,
-            leaf_cap=build_artifacts.max_leaf_size,
+            leaf_cap=leaf_cap_hint,
             leaf_parameter=build_artifacts.cache_leaf_parameter,
             upward=upward,
             locals_template=locals_template,
```

## Notes
- This patch keeps traced-mode correctness conservative unless the user explicitly provides `fixed_max_leaf_size`.
- ODISSEO already passes `fmm_fixed_max_leaf_size`, so this patch should immediately improve its outer-jit path.
- If `fixed_max_leaf_size` is too small for a given tree, existing checks should still raise.
