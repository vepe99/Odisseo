"""odisseo.globular_clusters: the catalogue parser, the frame, the King sampler (CPU, numpy)."""

import os

import numpy as np
import pytest

from odisseo import globular_clusters as gc

DATA = os.path.join(os.path.dirname(__file__), "data")
ORBITS = os.path.join(DATA, "baumgardt_orbits_fixture.txt")
STRUCTURE = os.path.join(DATA, "baumgardt_structure_fixture.txt")


@pytest.fixture(scope="module")
def catalogue():
    return gc.load_baumgardt(ORBITS, STRUCTURE)


def test_parser_reads_the_fixture(catalogue):
    assert sorted(catalogue) == ["NGC_104", "NGC_4590", "NGC_5466", "PAL_5"]
    pal5 = catalogue["PAL_5"]
    assert pal5.name == "Pal_5"
    assert (pal5.ra, pal5.dec, pal5.distance, pal5.rv) == (229.019167, -0.121, 21.94, -58.61)
    assert (pal5.pmra_cosdec, pal5.pmdec) == (-2.754, -2.643)
    assert pal5.mass == 1.34e4 and pal5.r_half == 27.63
    assert (pal5.r_core, pal5.r_tidal) == (15.35, 49.73)
    assert pal5.concentration == pytest.approx(np.log10(49.73 / 15.35))
    assert pal5.published == dict(X=-7.10, Y=0.22, Z=15.74, U=49.28, V=-144.14, W=-10.20)
    assert catalogue["NGC_104"].mass == 8.53e5


@pytest.mark.parametrize(
    "name,key",
    [("Pal 5", "PAL_5"), ("pal5", "PAL_5"), ("NGC 104", "NGC_104"), ("47 Tuc", "NGC_104"), ("M68", "NGC_4590")],
)
def test_names(name, key):
    assert gc.normalize_name(name) == key


def test_parser_refuses_a_short_row():
    with pytest.raises(ValueError):
        gc.parse_baumgardt_orbits("Pal_5 1 2 3\n")
    with pytest.raises(ValueError):
        gc.parse_baumgardt_structure("Pal_5 1 2 3\n")


def test_unknown_cluster(catalogue):
    with pytest.raises(KeyError):
        gc.cluster_table(["NGC 9999"], catalogue)


def test_transform_matches_the_published_galactocentric_values(catalogue):
    """Pal 5 (and the other fixture rows) against the catalogue's own X, Y, Z, U, V, W.

    The catalogue's axes: X from the centre to the Sun (= -x), Y along rotation, Z to
    the NGP; its frame is BAUMGARDT_FRAME. Tolerances: the table's rounding (0.01 kpc
    in position; proper motions to 0.001 mas/yr, i.e. 0.1 km/s at 22 kpc).
    """
    recs = [catalogue[k] for k in ("PAL_5", "NGC_5466", "NGC_4590", "NGC_104")]
    pos, vel = gc.observed_to_galactocentric(recs, gc.galactocentric_frame(**gc.BAUMGARDT_FRAME))
    for k, r in enumerate(recs):
        p = r.published
        np.testing.assert_allclose(pos[k], [-p["X"], p["Y"], p["Z"]], atol=0.02)
        np.testing.assert_allclose(vel[k], [-p["U"], p["V"], p["W"]], atol=0.3)
    # astropy's default frame (R0 8.122 kpc, v_sun (12.9, 245.6, 7.78)) is a different frame
    pos_d, vel_d = gc.observed_to_galactocentric(recs[:1])
    assert abs(pos_d[0, 0] - pos[0, 0]) > 0.03 and abs(vel_d[0, 1] - vel[0, 1]) > 3.0


def test_frame_parameters_are_recorded():
    rec = gc.frame_parameters(gc.galactocentric_frame(**gc.BAUMGARDT_FRAME))
    assert rec["galcen_distance_kpc"] == pytest.approx(8.178)
    assert rec["galcen_v_sun_kms"] == pytest.approx([11.1, 252.24, 7.25])
    assert rec["z_sun_pc"] == 0.0


def test_code_units():
    x, v = gc.to_code_units([[10.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]])
    assert x[0, 0] == pytest.approx(1.0)
    # 10 kpc, 1e10 Msun, G = 1: v_unit = sqrt(G 1e10 Msun / 10 kpc) = 65.58 km/s
    assert v[0, 0] == pytest.approx(100.0 / 65.58, rel=1e-3)


@pytest.mark.parametrize("w0,c", [(3.0, 0.67), (6.0, 1.26), (9.0, 2.12)])
def test_king_concentration_table(w0, c):
    """King (1966) c = log10(r_t / r_0) against the standard table (e.g. Binney & Tremaine fig 4.9)."""
    assert gc.KingModel(w0).concentration == pytest.approx(c, abs=0.01)
    assert gc.king_w0_from_concentration(gc.KingModel(w0).concentration) == pytest.approx(w0, abs=1e-3)


def test_king_sampler_density_profile():
    """The sampled radial density against the analytic King profile, shell by shell."""
    model = gc.KingModel(5.0)
    rng = np.random.default_rng(1)
    n = 40000
    x, _ = model.sample(n, rng)
    r = np.linalg.norm(x, axis=1)
    assert r.max() <= model.r_t * (1 + 1e-9)
    edges = np.geomspace(0.05, model.r_t, 16)
    counts, _ = np.histogram(r, edges)
    # expected counts: the analytic density integrated over each shell
    fine = np.geomspace(edges[0], edges[-1], 4000)
    rho = np.interp(fine, model.r, model.density)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] * fine[1:] ** 2 + rho[:-1] * fine[:-1] ** 2) * np.diff(fine))])
    shell = np.diff(np.interp(edges, fine, cum)) * 4 * np.pi / model.mass * n
    ok = shell > 20
    z = (counts[ok] - shell[ok]) / np.sqrt(shell[ok])
    assert np.all(np.abs(z) < 5.0), z
    # the density the ODE gives integrates to its mass profile
    assert 4 * np.pi * np.trapezoid(model.density * model.r**2, model.r) == pytest.approx(model.mass, rel=2e-3)


def test_king_sampler_is_in_virial_equilibrium():
    x, v, m = gc.sample_king(3000, w0=6.0, mass=2.0, r_half=0.5, G=1.0, rng=np.random.default_rng(3))
    assert m.sum() == pytest.approx(2.0)
    np.testing.assert_allclose((m[:, None] * x).sum(0), 0.0, atol=1e-12)
    np.testing.assert_allclose((m[:, None] * v).sum(0), 0.0, atol=1e-12)
    assert np.median(np.linalg.norm(x, axis=1)) == pytest.approx(0.5, rel=0.06)
    kin = 0.5 * np.sum(m * np.sum(v * v, 1))
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    iu = np.triu_indices(len(m), 1)
    pot = -np.sum(m[iu[0]] * m[iu[1]] / d[iu])
    assert 2 * kin / abs(pot) == pytest.approx(1.0, abs=0.06)


def test_clusters_from_catalogue(catalogue):
    xc, vc, mc, table = gc.clusters_from_catalogue(["Pal 5", "NGC 5466"], M=200, catalogue=catalogue,
                                                   return_table=True)
    assert xc.shape == (2, 200, 3) and vc.shape == (2, 200, 3) and mc.shape == (2, 200)
    for k, row in enumerate(table):
        assert mc[k].sum() == pytest.approx(row["mass"])
        np.testing.assert_allclose((mc[k, :, None] * xc[k]).sum(0) / mc[k].sum(), row["x"], atol=1e-12)
        np.testing.assert_allclose((mc[k, :, None] * vc[k]).sum(0) / mc[k].sum(), row["v"], atol=1e-12)
    assert table[0]["mass"] == pytest.approx(1.34e-6)  # 1.34e4 Msun in 1e10 Msun
    assert table[0]["r_half"] == pytest.approx(2.763e-3)  # 27.63 pc in 10 kpc
    xp, _, _ = gc.clusters_from_catalogue(["Pal 5"], M=100, catalogue=catalogue, model="plummer")
    assert xp.shape == (1, 100, 3)
    cen = (np.zeros((1, 3)), np.ones((1, 3)))
    xs, vs, _ = gc.clusters_from_catalogue(["Pal 5"], M=50, catalogue=catalogue, centres=cen)
    np.testing.assert_allclose(vs[0].mean(0), 1.0, atol=1e-12)
    with pytest.raises(ValueError):
        gc.clusters_from_catalogue(["Pal 5"], catalogue=catalogue, model="sersic")


def test_defaults():
    assert gc.default_reserve(1, 1000) == 1024
    assert gc.default_reserve(150, 1000) == 19456
    assert gc.default_cluster_softening([dict(r_half=1.3048), dict(r_half=2.0)]) == pytest.approx(0.05)


def test_integrate_orbit_is_reversible():
    def kepler(x):
        r = np.linalg.norm(x, axis=-1, keepdims=True)
        return -x / r**3

    x0, v0 = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
    x1, v1, track = gc.integrate_orbit(x0, v0, kepler, -2 * np.pi, 1e-3, n_out=10)
    assert track.shape == (11, 1, 2, 3)
    np.testing.assert_allclose(x1, x0, atol=1e-5)  # one circular period back
    x2, v2 = gc.integrate_orbit(x1, v1, kepler, 2 * np.pi, 1e-3)
    np.testing.assert_allclose(x2, x0, atol=1e-10)


def test_clusters_scheme_defaults(catalogue):
    """The scheme helper builds nornax's ClustersRunnerScheme with the documented defaults (no run)."""
    pytest.importorskip("nornax.adapters.jaccpot_runner", exc_type=ImportError)
    pytest.importorskip("jaccpot.production", exc_type=ImportError)
    from nornax.hybrid_clusters import StrippingPolicy

    xc, vc, mc, table = gc.clusters_from_catalogue(["Pal 5", "NGC 5466"], M=64, catalogue=catalogue,
                                                   return_table=True)
    eps_c = gc.default_cluster_softening(table)
    gal, clu, settings = gc.build_solvers(0.01, eps_c, backend="cpu")
    assert settings.mac_type == "dehnen"  # the CPU route's geometric MAC
    s = gc.clusters_scheme((xc, vc, mc), eps_c=eps_c, cluster_solver=clu)
    assert s.reserve == gc.default_reserve(2, 64)
    assert s.policy == StrippingPolicy()
    assert s.options == dict(cross="fmm", coupling="fmm", cluster_leaf_size=64)
    assert s.scheme.n_sub == 16 and s.scheme.order == 4 and s.scheme.cluster_batch == 1
    assert s.scheme.split.internal.softening == pytest.approx(eps_c)
    np.testing.assert_array_equal(np.asarray(s.clusters[2]), mc)
    assert gc.clusters_scheme((xc, vc, mc), eps_c=eps_c, cluster_solver=clu, policy=None).policy is None


def test_repark_empty_slots():
    """Empty slots move inside their cluster's member radius, co-moving; filled ones do not move."""
    from typing import Any, NamedTuple

    import jax.numpy as jnp

    class Stars(NamedTuple):
        positions: Any
        velocities: Any
        masses: Any

    class State(NamedTuple):
        clusters: Any

    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 8, 3)) * 0.01
    x[:, :, 0] += np.array([1.0, 2.0, 3.0])[:, None]
    v = rng.normal(size=(3, 8, 3)) * 0.1
    m = np.full((3, 8), 0.5)
    m[0, :3] = 0.0  # three stripped stars, left 5 units away at rest
    x[0, :3] += 5.0
    v[0, :3] = 0.0
    m[2] = 0.0  # a dissolved cluster
    state = State(Stars(jnp.asarray(x), jnp.asarray(v), jnp.asarray(m)))
    out = gc.repark_empty_slots(state).clusters
    xo, vo = np.asarray(out.positions), np.asarray(out.velocities)
    filled = m > 0
    np.testing.assert_array_equal(xo[filled], x[filled])
    np.testing.assert_array_equal(vo[filled], v[filled])
    np.testing.assert_array_equal(xo[2], x[2])  # dissolved: untouched
    X = (m[0, 3:, None] * x[0, 3:]).sum(0) / m[0, 3:].sum()
    V = (m[0, 3:, None] * v[0, 3:]).sum(0) / m[0, 3:].sum()
    R = np.linalg.norm(x[0, 3:] - X, axis=1).max()
    r_new = np.linalg.norm(xo[0, :3] - X, axis=1)
    np.testing.assert_allclose(r_new, R, rtol=1e-12)
    np.testing.assert_allclose(vo[0, :3], np.broadcast_to(V, (3, 3)), rtol=1e-12)
    assert len({tuple(np.round(p, 12)) for p in xo[0, :3]}) == 3  # distinct points
    again = gc.repark_empty_slots(State(out)).clusters  # idempotent
    np.testing.assert_allclose(np.asarray(again.positions), xo, rtol=0, atol=1e-15)


def test_default_n_sub(catalogue):
    table = gc.cluster_table(["Pal 5", "NGC 104"], catalogue)
    t = gc.crossing_times(table)
    assert t[0] == pytest.approx(np.sqrt(table[0]["r_half"] ** 3 / table[0]["mass"]))
    assert gc.default_n_sub(1e-3, table[:1]) == 8  # Pal 5 alone: the minimum
    assert gc.default_n_sub(1e-3, table) == int(np.ceil(20e-3 / t.min()))  # 47 Tuc sets it


# ---------------------------------------------------------------------------
# repark_empty (TEMPORARY workaround): physics-neutral, and it bounds the star tree's box.
# The runner's loop reduced, on nornax's direct-sum reference force (as nornax's own
# tests/unit/test_jaccpot_runner_scheme.py), float64, no jaccpot: an orbiting cluster that
# loses stars at several edges.
# ---------------------------------------------------------------------------


def _orbiting_system(m=32, n=64, seed=3):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.05, 0.9, n)
    r = 1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0)
    d = rng.normal(size=(n, 3))
    xg = r[:, None] * d / np.linalg.norm(d, axis=1)[:, None]
    vg = 0.3 * rng.normal(size=(n, 3))
    mg = np.full(n, 1.0 / n)
    xc = 0.05 * rng.normal(size=(1, m, 3)) + np.array([1.0, 0.0, 0.0])
    vc = 0.05 * rng.normal(size=(1, m, 3)) + np.array([0.0, 0.7, 0.0])
    mc = np.full((1, m), 2e-3 / m)
    return xg, vg, mg, xc, vc, mc


def _drive_scheme(repark, segments=10, dt=0.01, steps=5):
    import jax
    import jax.numpy as jnp

    from nornax.hybrid_clusters import DirectClustersForce, StrippingPolicy

    jax.config.update("jax_enable_x64", True)
    slow = DirectClustersForce(softening=0.05)

    def factory(rs, solver, settings, xg, mg, xc, mc, *, reserve, n_live=None, galaxy_aux=None):
        return slow, None

    xg, vg, mg, xc, vc, mc = _orbiting_system()
    rs = gc.clusters_scheme(
        (xc, vc, mc), eps_c=0.01, cluster_solver=None, n_sub=4, reserve=64, block_size=None, cluster_batch=None,
        policy=StrippingPolicy(r_factor=0.6, min_members=4), slow_factory=factory, repark_empty=repark,
    )
    carry = rs.start(None, None, jnp.stack([jnp.asarray(xg), jnp.asarray(vg)], 1), jnp.asarray(mg),
                     dt=dt, t0=0.0, escapers=None)
    takes, boxes, members = [], [], []
    for k in range(segments):
        carry = rs.segment(carry, dt, steps)
        assert not rs.verdict(carry)
        carry, info = rs.edge(carry, index=k + 1, escapers_due=False, energy=False)
        takes.append(info["n_take"])
        cl = carry.state.clusters
        x = np.asarray(cl.positions)[0]
        filled = np.asarray(cl.masses)[0] > 0
        boxes.append(float(np.max(x.max(0) - x.min(0))))  # the star tree's slots: all of them
        com = (np.asarray(cl.masses)[0][filled, None] * x[filled]).sum(0) / np.asarray(cl.masses)[0][filled].sum()
        members.append(float(np.max(np.linalg.norm(x[filled] - com, axis=1))))  # R: the farthest member
    return carry.state, takes, boxes, members


@pytest.fixture(scope="module")
def repark_runs():
    pytest.importorskip("nornax.adapters.jaccpot_runner", exc_type=ImportError)
    return _drive_scheme(True), _drive_scheme(False)


def test_repark_empty_is_physics_neutral(repark_runs):
    (on, takes_on, _, _), (off, takes_off, _, _) = repark_runs
    assert sum(takes_on) > 0 and sum(1 for t in takes_on if t) >= 2  # several stripping edges
    assert takes_on == takes_off  # the same stars leave at the same edges
    for name in ("galaxy_positions", "galaxy_velocities", "galaxy_masses", "galaxy_acc", "n_live",
                 "galaxy_star_id", "star_id", "dwell", "ledger"):
        np.testing.assert_allclose(np.asarray(getattr(on, name)), np.asarray(getattr(off, name)),
                                   rtol=1e-13, atol=0, err_msg=name)
    filled = np.asarray(on.clusters.masses) > 0
    np.testing.assert_array_equal(filled, np.asarray(off.clusters.masses) > 0)
    for name in ("positions", "velocities", "masses"):
        np.testing.assert_allclose(np.asarray(getattr(on.clusters, name))[filled],
                                   np.asarray(getattr(off.clusters, name))[filled], rtol=1e-13, atol=0,
                                   err_msg=name)
    np.testing.assert_allclose(np.asarray(on.cluster_acc)[filled], np.asarray(off.cluster_acc)[filled],
                               rtol=1e-13, atol=0)


def test_repark_empty_bounds_the_star_tree_box(repark_runs):
    """The cause of the per-edge star-tree re-plan: without the re-park the slots' box grows with the
    cluster's travel; with it the box stays within the members' own extent."""
    (_, _, boxes_on, members_on), (_, _, boxes_off, members_off) = repark_runs
    assert all(b <= 2 * r * (1 + 1e-12) for b, r in zip(boxes_on, members_on))  # inside the members' sphere
    assert boxes_off[-1] > 2.0 * 2 * members_off[-1]  # left-behind slots stretch the box
    assert boxes_off[-1] > 1.5 * boxes_on[-1]


def test_repark_empty_fails_loud_on_a_changed_nornax():
    class NoHooks:
        def edge(self, carry, **kw):
            return carry, {}

    with pytest.raises(RuntimeError, match="repark_empty=False"):
        gc.install_empty_slot_repark(NoHooks())

    class NoTransfer:
        def edge(self, carry, **kw):
            return carry, {}

        def _edge_progs(self, slow):
            return object()

    s = gc.install_empty_slot_repark(NoTransfer())
    with pytest.raises(RuntimeError, match="repark_empty=False"):
        s._edge_progs(None)
