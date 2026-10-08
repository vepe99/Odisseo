"""Gravitational softening kernels for ODISSEO's direct sums.

The same three kernels, with the same coefficients and conventions, as
``jaccpot.softening`` (ODISSEO does not depend on jaccpot, so the formulas are
mirrored here; ``tests/test_softening.py`` checks the two agree when jaccpot is
installed):

``"ferrers3"`` (default)
    Density ``315 / (64 pi h^3) (1 - r^2/h^2)^3`` on ``r < h``; polynomial in ``r^2``.
``"wendland_c2"``
    Wendland's C2 function ``21 / (2 pi h^3) (1 - r/h)^4 (1 + 4 r/h)``.
``"plummer"``
    ``(r^2 + eps^2)^(-1/2)``, the historical convention.

``softening`` is the Plummer-EQUIVALENT length for every kernel (equal central
potential): the support is ``h = 315/128 eps`` (ferrers3) or ``h = 3 eps``
(wendland_c2). Both compact kernels are exactly Newtonian beyond ``h``.

With ``s = 1 / max(r, h)`` and ``c = min(r^2/h^2, 1)`` (``min(r/h, 1)`` for
Wendland), the pair acceleration is ``-G m x G(c) s^3`` and the pair potential
``-G m Psi(c) s``; each polynomial is written as ``1 + (1 - c) R(c)``, so past
``h`` the factors are exactly ``r^-3`` and ``r^-1``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax.numpy as jnp

SOFTENING_KERNELS = ("ferrers3", "wendland_c2", "plummer")
DEFAULT_SOFTENING_KERNEL = "ferrers3"

_SUPPORT_FACTOR = {"ferrers3": 315.0 / 128.0, "wendland_c2": 3.0, "plummer": 0.0}
_FERRERS3_G = (89.0 / 16.0, -25.0 / 4.0, 35.0 / 16.0)
_FERRERS3_PSI = (187.0 / 128.0, -233.0 / 128.0, 145.0 / 128.0, -35.0 / 128.0)
_WENDLAND_G = (13.0, 13.0, -71.0, 69.0, -21.0)
_WENDLAND_PSI = (2.0, 2.0, -5.0, -5.0, 16.0, -12.0, 3.0)


def resolve_softening_kernel(kernel: Optional[str]) -> str:
    """Validate a kernel name; ``None`` gives :data:`DEFAULT_SOFTENING_KERNEL`.

    Args:
        kernel: A name from :data:`SOFTENING_KERNELS` (case-insensitive), or ``None``.

    Returns:
        The canonical name.

    Raises:
        ValueError: For any other name.
    """
    if kernel is None:
        return DEFAULT_SOFTENING_KERNEL
    name = str(kernel).strip().lower()
    if name not in SOFTENING_KERNELS:
        raise ValueError(f"softening_kernel={kernel!r}; expected one of {SOFTENING_KERNELS}")
    return name


def support_radius(kernel: Optional[str], softening: float) -> float:
    """The kernel's support ``h`` for a Plummer-equivalent ``softening`` (0 for Plummer).

    Args:
        kernel: A kernel name.
        softening: The Plummer-equivalent softening length.

    Returns:
        ``h``.
    """
    return _SUPPORT_FACTOR[resolve_softening_kernel(kernel)] * float(softening)


def _horner(c: Any, coeffs: Tuple[float, ...]) -> Any:
    out = coeffs[-1]
    for a in coeffs[-2::-1]:
        out = a + c * out
    return out


def softened_inverse_powers(
    r2: Any, softening: Any, kernel: Optional[str], xp: Any = jnp
) -> Tuple[Any, Any]:
    """``(g, psi)``: the kernel's stand-ins for ``r^-3`` and ``r^-1`` at squared separation ``r2``.

    ``r2`` carries no softening. The pair acceleration is ``-G m x g`` and the pair
    potential ``-G m psi``. Finite at ``r2 = 0`` for every kernel with ``softening > 0``
    (``psi(0) = 1 / softening`` for all three), so callers mask self pairs as before.

    Args:
        r2: Squared separations.
        softening: Plummer-equivalent softening length (scalar).
        kernel: A kernel name; ``None`` gives the default.
        xp: ``jax.numpy`` (default) or ``numpy`` (host-side fp64 references).

    Returns:
        ``(g, psi)``, elementwise.
    """
    name = resolve_softening_kernel(kernel)
    if isinstance(softening, (int, float)) and float(softening) == 0.0:
        name = "plummer"  # Newton for every kernel; keeps 0/0 out of the clip
    eps = xp.asarray(softening, dtype=xp.asarray(r2).dtype)
    if name == "plummer":
        d2 = r2 + eps * eps
        return d2**-1.5, d2**-0.5
    h = _SUPPORT_FACTOR[name] * eps
    s = 1.0 / xp.sqrt(xp.maximum(r2, h * h))
    if name == "ferrers3":
        c = xp.minimum(r2 / (h * h), 1.0)
        rg, rpsi = _FERRERS3_G, _FERRERS3_PSI
    else:
        c = xp.minimum(xp.sqrt(r2) / h, 1.0)
        rg, rpsi = _WENDLAND_G, _WENDLAND_PSI
    one_minus = 1.0 - c
    g = (1.0 + one_minus * _horner(c, rg)) * s * s * s
    psi = (1.0 + one_minus * _horner(c, rpsi)) * s
    return g, psi
