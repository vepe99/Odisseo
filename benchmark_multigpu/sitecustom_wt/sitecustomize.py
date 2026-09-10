"""Repoint the editable jaccpot / yggdrax finders at a worktree (process-local).

Use:  PYTHONPATH=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt \
      JACCPOT_WORKTREE=<jaccpot worktree> [YGGDRAX_WORKTREE=<yggdrax worktree>] python ...
The finders' MAPPING wins over PYTHONPATH, so this is the only way to test a branch
without switching the shared checkouts (see memory jaccpot-worktree-isolation). The
shared /export/home/tbuck/yggdrax checkout sits on a paper branch, so any yggdrax
measurement for the record must name YGGDRAX_WORKTREE explicitly.
"""
import os

_wt = os.environ.get("JACCPOT_WORKTREE", "/export/home/tbuck/jaccpot-strict-wt")
try:
    import __editable___jaccpot_0_0_1_finder as _jf

    _jf.MAPPING["jaccpot"] = os.path.join(_wt, "jaccpot")
except Exception as _exc:  # pragma: no cover
    print("sitecustomize: could not repoint jaccpot finder:", _exc)

_ywt = os.environ.get("YGGDRAX_WORKTREE")
if _ywt:
    try:
        import __editable___yggdrax_0_0_1_finder as _yf

        _yf.MAPPING["yggdrax"] = os.path.join(_ywt, "yggdrax")
    except Exception as _exc:  # pragma: no cover
        print("sitecustomize: could not repoint yggdrax finder:", _exc)
