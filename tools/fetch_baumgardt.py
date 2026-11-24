#!/usr/bin/env python3
"""Download Holger Baumgardt's Galactic globular cluster tables (for odisseo.globular_clusters).

Fetches the orbit table and the structural table from
https://people.smp.uq.edu.au/HolgerBaumgardt/globular/ into ``--out`` and writes
``SOURCE.json`` next to them: URL, UTC date, size and sha256 of each file. The tables
are the database's (Baumgardt & Hilker 2018; Vasiliev & Baumgardt 2021; Baumgardt &
Vasiliev 2021; Baumgardt et al. 2023): cite those when you use them. The site states
no licence; Odisseo's repository carries only a four-row test fixture
(``tests/data/baumgardt_*_fixture.txt``), never the tables.

usage: python tools/fetch_baumgardt.py --out DIR   (then ODISSEO_GC_DATA=DIR)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from odisseo.globular_clusters import (  # noqa: E402
    BAUMGARDT_BASE_URL,
    BAUMGARDT_FILES,
    parse_baumgardt_orbits,
    parse_baumgardt_structure,
)


def fetch(out: str, base: str = BAUMGARDT_BASE_URL, timeout: float = 60.0) -> dict:
    """Download both tables into ``out``; return the source record (also written to SOURCE.json)."""
    os.makedirs(out, exist_ok=True)
    record = dict(base_url=base, fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  files={})
    for kind, name in BAUMGARDT_FILES.items():
        url = base + name
        req = urllib.request.Request(url, headers={"User-Agent": "odisseo-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        text = data.decode("utf-8", errors="replace")
        rows = (parse_baumgardt_orbits if kind == "orbits" else parse_baumgardt_structure)(text)
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(data)
        record["files"][kind] = dict(url=url, file=name, bytes=len(data),
                                     sha256=hashlib.sha256(data).hexdigest(), clusters=len(rows))
        print(f"[fetch] {url}: {len(data)} B, {len(rows)} clusters", flush=True)
    with open(os.path.join(out, "SOURCE.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    return record


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="directory for the tables (ODISSEO_GC_DATA)")
    p.add_argument("--base-url", default=BAUMGARDT_BASE_URL)
    a = p.parse_args(argv)
    fetch(a.out, a.base_url)


if __name__ == "__main__":
    main()
