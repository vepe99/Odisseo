from typing import NamedTuple
from jdgsim.integrators import LEAPFROG
from jdgsim.dynamics import DIRECT_ACC, DIRECT_ACC_LAXMAP
from jdgsim.potentials import NFW_POTENTIAL, POINT_MASS, MN_POTENTIAL
from astropy import units as u
from astropy import constants as c
from math import log


class PlummerParams(NamedTuple):
    """
    NamedTuple containing the parameters for the Plummer profile
    """
    
    a: float = 7 * u.kpc #kpc
    
    Mtot: float = 1.0 * u.Msun #M☉

class NFWParams(NamedTuple):
    """
    NamedTuple containing the parameters for the NFW profile
    """
    
    Mvir: float = 1.62*1e11 * u.Msun #M☉
    
    r_s: float = 15.3 * u.kpc #kpc
    
    c: float = 10

    d_c: float = log(1+c) - c/(1+c)

class PointMassParams(NamedTuple):
    """
    NamedTuple containing the parameters for the point mass
    """

    M: float = 1.0 * u.Msun #M☉
    
class MNParams(NamedTuple):
    """
    NamedTuple containing the parameters for the Myamoto Nagai profile
    
    """

    M: float = 6.5e10 * u.Msun #M☉

    a: float = 3.0 * u.kpc #kpc

    b: float = 0.28 * u.kpc #kpc


class SimulationParams(NamedTuple):
    """
    NamedTuple containing the parameters for the simulation. This parameter do not require recompilation
    """
    
    G: float = c.G.to(u.kpc**3 / u.Msun / u.Gyr**2)
    
    t_end: float = 1.0 * u.Gyr #Gyr

    Plummer_params: PlummerParams = PlummerParams()

    NFW_params: NFWParams = NFWParams()

    PointMass_params: PointMassParams = PointMassParams()

    MN_params: MNParams = MNParams()
    
class SimulationConfig(NamedTuple):
    """
    NamedTuple containing the configuration for the simulation. This parameter require recompilation
    """
    
    N_particles: int = 1000
    
    dimensions: int = 3
    
    return_snapshots: bool = False
    
    num_snapshots: int = 10
    
    fixed_timestep: bool = True
    
    num_timesteps: int = 1000
    
    softening:float = 1e-10
    
    integrator: int = LEAPFROG
    
    acceleration_scheme: int = DIRECT_ACC

    batch_size: int = 10_000

    double_map: bool = False

    external_accelerations: tuple = ()

    
