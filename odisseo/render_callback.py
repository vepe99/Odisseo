"""Minimal-sync callback rendering for the fused FMM execution lane.

Streams frames out of the device-resident ``strict_run_v2`` velocity-Verlet scan
via ``jax.debug.callback`` (mirrors the astronomix snapshot pattern). The key
property: the 2D projection is computed **on device**, and only the small
``(res, res)`` density grid crosses to the host, fire-and-forget, every ``K``
steps -- so the GPU keeps running the fused scan and rendering does not stall it.

Usage (host side):

    sink = FrameSink()
    cb = make_density_step_callback(sink, bounds_min, bounds_max, res=256, stride=20)
    integrate_leapfrog_jaccpot_active(..., return_history=False,
                                      step_callback=cb, step_callback_stride=20)
    jax.block_until_ready(final_state)   # flush pending debug callbacks
    sink.encode("movie.gif", fps=30)

Only ``num_steps // stride`` small grids are transferred (e.g. 100 x 256x256xf32
= ~25 MB total for a 2000-step run), never the full particle trajectory.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def project_density_2d(
    positions: jnp.ndarray,
    bounds_min: jnp.ndarray,
    bounds_max: jnp.ndarray,
    res: int,
    axes: tuple[int, int] = (0, 1),
) -> jnp.ndarray:
    """On-device 2D particle density projection -> ``(res, res)`` float32 grid.

    A jittable 2D histogram: positions are binned by scatter-add, so this stays
    on the device and only the small grid is later shipped to the host.
    """
    ax0, ax1 = axes
    span0 = bounds_max[ax0] - bounds_min[ax0]
    span1 = bounds_max[ax1] - bounds_min[ax1]
    u = (positions[:, ax0] - bounds_min[ax0]) / span0
    v = (positions[:, ax1] - bounds_min[ax1]) / span1
    r = jnp.int32(res)
    iu = jnp.clip((u * res).astype(jnp.int32), 0, r - 1)
    iv = jnp.clip((v * res).astype(jnp.int32), 0, r - 1)
    flat = iu * r + iv
    grid = jnp.zeros((res * res,), dtype=jnp.float32).at[flat].add(1.0)
    return grid.reshape(res, res)


class FrameSink:
    """Host-side frame buffer. ``push`` is the (cheap) target of the callback."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.frames: list[np.ndarray] = []

    def push(self, step_index, grid) -> None:
        # Runs on the host off the debug-callback stream. Keep it cheap
        # (append only) so the device is never back-pressured; encode later.
        self.steps.append(int(step_index))
        self.frames.append(np.asarray(grid, dtype=np.float32))

    def _radial_profile(self, grid: np.ndarray, nbins: int = 80):
        """Azimuthally-averaged surface density Sigma(R) from a density grid.

        Bins pixels by distance from the density centroid; divides by annulus
        area. Returns (centers, Sigma)."""
        res = grid.shape[0]
        yy, xx = np.mgrid[0:res, 0:res].astype(np.float64)
        total = float(grid.sum()) or 1.0
        cx = float((grid * xx).sum() / total)
        cy = float((grid * yy).sum() / total)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
        w = grid.ravel().astype(np.float64)
        rmax = float(np.percentile(rr, 99.0))
        bins = np.linspace(0.0, max(rmax, 1e-6), nbins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
        h, _ = np.histogram(rr, bins=bins, weights=w)
        return centers, h / np.maximum(area, 1e-12)

    def ring_metric(self) -> dict:
        """Quantify radial ring structure from the density grids.

        Mirrors the sweep tool's ring score: RMS of the fractional residual of
        Sigma(R) about a smoothed profile, over a mid-radius band, for the first
        and last frame. Rising ring_rms_end indicates rings forming."""
        if len(self.frames) < 2:
            raise RuntimeError("need >=2 frames for ring_metric")
        order = np.argsort(self.steps)

        def frame_ring(grid: np.ndarray) -> float:
            centers, prof = self._radial_profile(grid)
            smooth = np.convolve(prof, np.ones(9) / 9.0, mode="same")
            resid = (prof - smooth) / np.maximum(smooth, 1e-12)
            rmax = centers[-1]
            mask = (centers > 0.15 * rmax) & (centers < 0.85 * rmax)
            return float(np.sqrt(np.mean(resid[mask] ** 2)))

        start = frame_ring(self.frames[order[0]])
        end = frame_ring(self.frames[order[-1]])
        return {"ring_rms_start": start, "ring_rms_end": end}

    def to_rgb(self, cmap: str = "magma", percentile: float = 99.5) -> np.ndarray:
        """Return time-ordered ``(frames, res, res, 3)`` uint8 RGB (log1p+percentile)."""
        if not self.frames:
            raise RuntimeError("FrameSink is empty; nothing to encode")
        import matplotlib.cm as cm

        # Defensive time-order (jax.debug.callback is unordered).
        order = np.argsort(self.steps)
        stacked = np.stack([np.log1p(self.frames[i]) for i in order])
        vmax = float(np.percentile(stacked, percentile)) or 1.0
        norm = np.clip(stacked / vmax, 0.0, 1.0)
        colormap = cm.get_cmap(cmap)
        return (colormap(norm)[..., :3] * 255).astype(np.uint8)

    def _save_gif(self, path: str, frames_rgb: list, fps: int) -> str:
        duration_ms = int(1000 / max(fps, 1))
        if path.lower().endswith(".gif"):
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames_rgb]
            imgs[0].save(
                path,
                save_all=True,
                append_images=imgs[1:],
                duration=duration_ms,
                loop=0,
            )
        else:
            import imageio.v3 as iio

            iio.imwrite(path, list(frames_rgb), fps=fps)
        return path

    def encode(
        self,
        path: str,
        fps: int = 30,
        cmap: str = "magma",
        percentile: float = 99.5,
        *,
        extent: Sequence[float] | None = None,
        xlabel: str = "x",
        ylabel: str = "y",
        dt_time: float | None = None,
        time_label: str = "",
        cbar_label: str = "log$_{10}$(1 + N per cell)",
        title: str | None = None,
        dpi: int = 110,
    ) -> str:
        """Encode buffered density grids to a movie (GIF via Pillow; other
        containers via imageio if installed).

        With ``extent`` (x0, x1, y0, y1 in display units, e.g. kpc) the frames are
        rendered scientifically via matplotlib: labelled axes, a colorbar, and a
        per-frame time stamp (``t = step * dt_time`` with ``time_label``). Without
        ``extent`` it falls back to bare colormapped rasters."""
        if not self.frames:
            raise RuntimeError("FrameSink is empty; nothing to encode")
        if extent is None:
            return self._save_gif(
                path, list(self.to_rgb(cmap=cmap, percentile=percentile)), fps
            )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = np.argsort(self.steps)
        # Consistent log-density normalization across frames (no flicker).
        logs = [np.log10(1.0 + self.frames[i]) for i in order]
        vmax = float(np.percentile(np.stack(logs), percentile)) or 1.0
        times = None
        if dt_time is not None:
            times = [float(self.steps[i]) * float(dt_time) for i in order]

        frames_rgb: list[np.ndarray] = []
        for k, lg in enumerate(logs):
            fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=dpi)
            # grid[i, j] = density at (x-bin i, y-bin j); transpose so x is
            # horizontal, y vertical, with origin at lower-left.
            im = ax.imshow(
                lg.T,
                origin="lower",
                extent=list(extent),
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ttl = title or ""
            if times is not None:
                ttl = (ttl + "   " if ttl else "") + f"t = {times[k]:.2f} {time_label}"
            if ttl:
                ax.set_title(ttl.strip())
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cbar_label)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            frames_rgb.append(buf)
            plt.close(fig)
        return self._save_gif(path, frames_rgb, fps)


class PositionSink:
    """Host buffer of subsampled particle positions per emitted step.

    Particle-based (not grid-based), so azimuthal density profiles are far less
    shot-noise-limited than a sparse density grid -- suitable for a reliable ring
    metric and for repairing the sweep's snapshot-based scoring on the fused lane."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.positions: list[np.ndarray] = []

    def push(self, step_index, pos) -> None:
        self.steps.append(int(step_index))
        self.positions.append(np.asarray(pos, dtype=np.float32))

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.steps)
        return (
            np.asarray([self.steps[i] for i in order]),
            np.stack([self.positions[i] for i in order]),
        )

    def ring_metric(self, nbins: int = 161) -> dict:
        """RMS fractional residual of Sigma(R) (mirrors the sweep scorer), for the
        first and last emitted frame, from the actual particle radii."""
        if len(self.positions) < 2:
            raise RuntimeError("need >=2 frames for ring_metric")
        _, sp = self.stack()  # [T, Ns, 3]
        rxy = np.linalg.norm(sp[:, :, :2], axis=2)
        rmax = float(np.percentile(rxy[0], 99.9))
        bins = np.linspace(0.0, max(rmax, 1e-6), nbins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
        mask = (centers > 0.3 * rmax) & (centers < 0.95 * rmax)

        def frame_ring(rt: np.ndarray) -> float:
            h, _ = np.histogram(rt, bins=bins)
            prof = h / np.maximum(area, 1e-12)
            smooth = np.convolve(prof, np.ones(9) / 9.0, mode="same")
            resid = (prof - smooth) / np.maximum(smooth, 1e-12)
            return float(np.sqrt(np.mean(resid[mask] ** 2)))

        start = frame_ring(rxy[0])
        end = frame_ring(rxy[-1])
        r99_growth = float(
            np.percentile(rxy[-1], 99) / max(np.percentile(rxy[0], 99), 1e-12)
        )
        return {"ring_rms_start": start, "ring_rms_end": end, "r99_growth": r99_growth}


def make_position_step_callback(
    sink: PositionSink,
    sample_indices,
) -> Callable[[jnp.ndarray, jnp.ndarray], None]:
    """Traced step_callback shipping only subsampled particle positions to host."""
    idx = jnp.asarray(sample_indices, dtype=jnp.int32)

    def _step_callback(step_index: jnp.ndarray, state: jnp.ndarray) -> None:
        jax.debug.callback(sink.push, step_index, state[idx, 0, :])

    return _step_callback


def make_density_step_callback(
    sink: FrameSink,
    bounds_min: Sequence[float] | jnp.ndarray,
    bounds_max: Sequence[float] | jnp.ndarray,
    res: int = 256,
    axes: tuple[int, int] = (0, 1),
) -> Callable[[jnp.ndarray, jnp.ndarray], None]:
    """Build a traced ``step_callback(step_index, state)`` for ``strict_run_v2``.

    Computes the density projection on-device and ships only the grid to
    ``sink.push`` via a fire-and-forget ``jax.debug.callback`` (returns nothing,
    does not touch the scan carry). The caller controls cadence via
    ``step_callback_stride``.
    """
    bmin = jnp.asarray(bounds_min, dtype=jnp.float32)
    bmax = jnp.asarray(bounds_max, dtype=jnp.float32)

    def _step_callback(step_index: jnp.ndarray, state: jnp.ndarray) -> None:
        positions = state[:, 0, :]
        grid = project_density_2d(positions, bmin, bmax, res, axes)
        jax.debug.callback(sink.push, step_index, grid)

    return _step_callback
