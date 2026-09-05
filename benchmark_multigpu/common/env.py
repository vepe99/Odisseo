"""Provenance capture: git SHAs + library/GPU versions embedded in every artifact.

Every benchmark script calls :func:`capture_provenance` and stores the result in
its ``.json``/``.npz`` output so a figure can always be traced back to the exact
code + hardware that produced it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

REPOS = {
    "jaccpot": "/export/home/tbuck/jaccpot",
    "yggdrax": "/export/home/tbuck/yggdrax",
    "jztree": "/export/home/tbuck/jztree",
    "odisseo_bench": "/export/home/tbuck/Odisseo-bench-multigpu",
}


def _git(repo: str, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _repo_state(repo: str) -> dict:
    return {
        "sha": _git(repo, "rev-parse", "--short", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
    }


def capture_provenance(extra: dict | None = None) -> dict:
    """Return a provenance dict (repo SHAs, library + GPU versions, timestamp)."""
    prov: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repos": {name: _repo_state(path) for name, path in REPOS.items()},
    }
    try:
        import jax

        prov["jax_version"] = jax.__version__
        prov["jax_backend"] = jax.default_backend()
        prov["devices"] = [
            {"id": d.id, "kind": d.device_kind, "platform": d.platform}
            for d in jax.devices()
        ]
    except Exception as exc:  # pragma: no cover
        prov["jax_error"] = str(exc)

    try:
        gpus = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        prov["nvidia_smi"] = gpus
    except Exception:
        pass

    if extra:
        prov.update(extra)
    return prov


def print_provenance(prov: dict) -> None:
    print(json.dumps(prov, indent=2))
