import pytest
import jax.numpy as jnp
from odisseo import potentials
from odisseo.option_classes import (
    SimulationConfig, SimulationParams, 
    NFW_POTENTIAL, POINT_MASS, MN_POTENTIAL, PSP_POTENTIAL, LOGARITHMIC_POTENTIAL,
    NFWParams, PointMassParams, MNParams, PSPParams, LogarithmicParams
)

class DummyParams:
    G = 4.30091e-6  # (kpc * (km/s)^2) / Msun

    class NFW_params:
        Mvir = 1e12
        r_s = 20.0

    class PointMass_params:
        M = 1e11

    class MN_params:
        M = 5e10
        a = 6.5
        b = 0.26

    class PSP_params:
        M = 1e11
        alpha = 2.0
        r_c = 20.0

    class Logarithmic_Params:
        v0 = 220.0
        q = 0.9


@pytest.fixture
def state():
    # 3 particles, shape (3, 2, 3) -> positions & velocities
    pos = jnp.array([[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]])
    vel = jnp.zeros_like(pos)
    return jnp.stack([pos, vel], axis=1)


@pytest.fixture
def config():
    return SimulationConfig(N_particles=3, 
                            external_accelerations=(NFW_POTENTIAL, 
                                                    POINT_MASS, 
                                                    MN_POTENTIAL,
                                                    PSP_POTENTIAL, 
                                                    LOGARITHMIC_POTENTIAL, ))


@pytest.fixture
def params():
    return SimulationParams(
        G=DummyParams.G,
        NFW_params=NFWParams(),
        PointMass_params=PointMassParams(), 
        MN_params=MNParams(),
        PSP_params=PSPParams(),
        Logarithmic_params=LogarithmicParams(),
    )


@pytest.mark.parametrize("potential_func", [
    potentials.NFW,
    potentials.point_mass,
    potentials.MyamotoNagai,
    potentials.PowerSphericalPotentialwCutoff,
    potentials.logarithmic_potential,
])
def test_potential_is_negative(state, config, params, potential_func):
    """Test that potential values are negative or zero."""
    _, pot = potential_func(state, config, params, return_potential=True)
    print(f"Testing {potential_func.__name__} potential values: {pot}")
    assert jnp.all(pot <= 0.0), f"{potential_func.__name__} has positive potential values!"


def test_combined_potential_is_sum_of_individuals(state, config, params):
    """Test that combined external potential equals sum of individual potentials."""
    total_acc, total_pot = potentials.combined_external_acceleration_vmpa_switch(state, config, params, return_potential=True)

    acc1, pot1 = potentials.NFW(state, config, params, return_potential=True)
    acc2, pot2 = potentials.point_mass(state, config, params, return_potential=True)
    acc3, pot3 = potentials.MyamotoNagai(state, config, params, return_potential=True)
    acc4, pot4 = potentials.PowerSphericalPotentialwCutoff(state, config, params, return_potential=True)
    acc5, pot5 = potentials.logarithmic_potential(state, config, params, return_potential=True)

    acc_sum = acc1 + acc2 + acc3 + acc4 + acc5
    pot_sum = pot1 + pot2 + pot3 + pot4 + pot5

    assert jnp.allclose(total_acc, acc_sum, rtol=1e-5), "Total acceleration does not match sum of components."
    assert jnp.allclose(total_pot, pot_sum, rtol=1e-5), "Total potential does not match sum of components."


def test_output_shape(state, config, params):
    acc, pot = potentials.NFW(state, config, params, return_potential=True)
    assert acc.shape == (state.shape[0], 3)
    assert pot.shape == (state.shape[0],)
