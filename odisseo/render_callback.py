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

    def encode(
        self,
        path: str,
        fps: int = 30,
        cmap: str = "magma",
        percentile: float = 99.5,
    ) -> str:
        """Encode buffered density grids to a GIF (log1p + percentile norm).

        Uses Pillow (always available) for GIF; falls back to imageio for other
        containers (e.g. .mp4) when installed.
        """
        rgb = self.to_rgb(cmap=cmap, percentile=percentile)
        duration_ms = int(1000 / max(fps, 1))
        if path.lower().endswith(".gif"):
            from PIL import Image

            imgs = [Image.fromarray(frame) for frame in rgb]
            imgs[0].save(
                path,
                save_all=True,
                append_images=imgs[1:],
                duration=duration_ms,
                loop=0,
            )
        else:
            import imageio.v3 as iio

            iio.imwrite(path, list(rgb), fps=fps)
        return path


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
