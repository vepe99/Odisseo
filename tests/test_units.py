import pytest
from astropy import units as u
from odisseo.units import CodeUnits
from astropy.constants import G as physical_G


def test_codeunits_with_time():
    unit_length = 1 * u.kpc
    unit_mass = 1e10 * u.Msun
    unit_time = 1e8 * u.yr
    G_sim = 1.0  # arbitrary for test

    units = CodeUnits(unit_length, unit_mass, G_sim, unit_time)

    assert 1. * units.code_length.to(u.kpc) == 1.
    assert 1. * units.code_mass.to(u.Msun) == 1e10
    assert 1 * units.code_time.to(u.yr) == 1e8
    assert isinstance(units.G, float)
    assert 1 * units.code_velocity.to(u.kpc/u.yr) == 1/1e8


# def test_codeunits_without_time():
#     unit_length = 1 * u.kpc
#     unit_mass = 1e10 * u.Msun
#     G_sim = 1.0  # user provides G in code units

#     units = CodeUnits(unit_length, unit_mass, G_sim)

#     # Code time should be derived
#     assert 1. * units.code_length.to(u.kpc) == 1.
#     assert 1. * units.code_mass.to(u.Msun) == 1e10
#     assert 1 * units.code_time.to(u.yr) ==  ()**(1/2)
#     assert isinstance(units.G, float)
#     assert 1 * units.code_velocity.to(u.kpc/u.yr) == 1/1e8

#     # Check that G in code units is close to provided G_sim
#     G_in_code_units = physical_G.to(units.code_length**3 / (units.code_mass * units.code_time**2)).value
#     assert pytest.approx(G_in_code_units, rel=1e-5) == units.G
