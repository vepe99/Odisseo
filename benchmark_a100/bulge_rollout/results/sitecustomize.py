"""Pin `yggdrax` to a fixed worktree, out of band, for this project's runs.

`yggdrax` is installed EDITABLE, and the generated finder hard-codes

    MAPPING = {'yggdrax': '/export/home/tbuck/yggdrax/yggdrax'}

so every run imports whatever branch that SHARED checkout happens to be sitting on.
On 2026-09-02 15:38 it was switched from `main` (a5262ae, which has yggdrax PR #54)
to `paper/differentiable-applications` (954ecaf, which does not), and the next run
correctly refused to start because the cross-walk pair policy had vanished underneath
it. A measurement whose library can change branch between two runs is not reproducible,
whatever the code says.

PYTHONPATH cannot fix this: the editable finder is installed into `sys.meta_path` by a
.pth file during `site.py` and takes precedence over path entries. `sitecustomize` is
imported by `site.py` AFTER the .pth files are processed, which is exactly the window
where the MAPPING can be repointed -- so this file is the hook, not an ordinary module.

Set PYTHONPATH to this directory and yggdrax resolves to YGGDRAX_PIN (or the default
below) regardless of the shared checkout. Nothing here writes to the shared repo.
"""
import os

_PIN = os.environ.get("YGGDRAX_PIN", "/export/home/tbuck/yggdrax-main-wt/yggdrax")

try:
    import __editable___yggdrax_0_0_1_finder as _f
except Exception:  # not installed editable in this env; nothing to pin
    pass
else:
    if os.path.isdir(_PIN):
        _f.MAPPING["yggdrax"] = _PIN
        if os.environ.get("YGGDRAX_PIN_VERBOSE"):
            print(f"[sitecustomize] yggdrax pinned to {_PIN}", flush=True)
    else:
        raise RuntimeError(
            f"YGGDRAX_PIN={_PIN!r} is not a directory; refusing to run against an "
            f"unpinned shared checkout"
        )
