#!/usr/bin/env python3
"""Encode the density-grid stacks a mesh rollout wrote into labelled movies.

``tools/mesh_galaxy_run.py --render-every N --projection xy,xz`` leaves one
``<prefix>_frames_<proj>.npz`` per projection. Those are raw surface-density grids,
not pictures: they carry no axes, no scale and no time. This turns each into a movie
through ``odisseo.render_callback.FrameSink.encode``, which is the encoder the
single-GPU lane already uses -- same colormap, same log normalisation held fixed
across frames so the movie does not flicker, same time stamp.

The projection decides the axis labels, and getting them from the file NAME rather
than from a flag is deliberate: the two stacks differ only in which components were
histogrammed, so a hand-passed label is one copy-paste away from an edge-on movie
labelled as face-on.

Example
-------
    python tools/mesh_frames_to_movie.py \
        --prefix /export/scratch/tbuck/odisseo_runs/quarter_orbit/qorbit \
        --extent 1.2 --length-unit-kpc 10 --dt 2.5e-4 --render-every 10 \
        --time-unit-myr 149.1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_AXIS_LABELS = {
    "xy": ("x", "y"),
    "xz": ("x", "z"),
    "yz": ("y", "z"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True,
                   help="The rollout's --out-prefix; <prefix>_frames_<proj>.npz is read.")
    p.add_argument("--projections", default="xy,xz",
                   help="Comma-separated projections to encode.")
    p.add_argument("--extent", type=float, required=True,
                   help="Half-width of the render window in CODE units (the run's "
                        "--render-extent).")
    p.add_argument("--length-unit-kpc", type=float, default=10.0,
                   help="Physical length of one code unit, for the axis labels.")
    p.add_argument("--dt", type=float, default=0.0,
                   help="Integrator dt in code units (the run's --dt).")
    p.add_argument("--render-every", type=int, default=1,
                   help="Steps between frames (the run's --render-every), so the "
                        "time stamp counts real steps and not frame index.")
    p.add_argument("--time-unit-myr", type=float, default=149.1,
                   help="Physical duration of one code time unit.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--cmap", default="magma")
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--format", default="mp4",
                   help="Output container extension (mp4 needs imageio+ffmpeg; gif "
                        "always works).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from odisseo.render_callback import FrameSink

    ext_kpc = float(args.extent) * float(args.length_unit_kpc)
    wrote = []
    for proj in [t.strip().lower() for t in args.projections.split(",") if t.strip()]:
        src = pathlib.Path(f"{args.prefix}_frames_{proj}.npz")
        if not src.exists():
            print(f"[skip] {src} does not exist")
            continue
        frames = np.load(src)["frames"]
        sink = FrameSink()
        for k, grid in enumerate(frames):
            # The real step number, so the time stamp is physical rather than a
            # frame counter.
            sink.push(k * max(int(args.render_every), 1), grid)
        xl, yl = _AXIS_LABELS.get(proj, ("x", "y"))
        out = f"{args.prefix}_{proj}.{args.format}"
        try:
            path = sink.encode(
                out,
                fps=int(args.fps),
                cmap=args.cmap,
                percentile=float(args.percentile),
                extent=(-ext_kpc, ext_kpc, -ext_kpc, ext_kpc),
                xlabel=f"{xl} [kpc]",
                ylabel=f"{yl} [kpc]",
                dt_time=float(args.dt) * float(args.time_unit_myr),
                time_label="Myr",
                title=f"21M disc + bulge, {proj} projection",
            )
        except Exception as exc:  # noqa: BLE001
            # imageio/ffmpeg is optional; a GIF needs only Pillow.
            print(f"[warn] {args.format} encode failed ({exc}); falling back to gif")
            path = sink.encode(
                f"{args.prefix}_{proj}.gif",
                fps=int(args.fps), cmap=args.cmap, percentile=float(args.percentile),
                extent=(-ext_kpc, ext_kpc, -ext_kpc, ext_kpc),
                xlabel=f"{xl} [kpc]", ylabel=f"{yl} [kpc]",
                dt_time=float(args.dt) * float(args.time_unit_myr),
                time_label="Myr",
                title=f"21M disc + bulge, {proj} projection",
            )
        wrote.append((path, len(frames)))
        print(f"[ok] {path}  ({len(frames)} frames)")
    if not wrote:
        raise SystemExit("no frame stacks found for the given prefix/projections")


if __name__ == "__main__":
    main()
