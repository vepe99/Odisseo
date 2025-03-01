from typing import NamedTuple
from jdgsim.integrators import LEAPFROG
from jdgsim.dynamics import DIRECT_ACC
from jdgsim.potentials import NFW_POTENTIAL, POINT_MASS
from astropy import units as u
from astropy import constants as c
from math import log


class NFWParams(NamedTuple):
    """
    NamedTuple containing the parameters for the NFW profile
    """
    
    Mvir: float = 1.62*1e11 * u.Msun #M☉
    
    r_s: float = 15.3 * u.kpc #kpc
    
    c: float = 10

    d_c: float = log(1+c) - c/(1+c)

class PlummerParams(NamedTuple):
    """
    NamedTuple containing the parameters for the Plummer profile
    """
    
    a: float = 7 * u.kpc #kpc
    
    Mtot: float = 1.0 * u.Msun #M☉

class PointMassParams(NamedTuple):
    """
    NamedTuple containing the parameters for the point mass
    """

    M: float = 1.0 * u.Msun #M☉
    

class SimulationParams(NamedTuple):
    """
    NamedTuple containing the parameters for the simulation. This parameter do not require recompilation
    """
    
    G: float = c.G.to(u.kpc**3 / u.Msun / u.Gyr**2)
    
    t_end: float = 1.0 * u.Gyr #Gyr
    
    NFW_params: NFWParams = NFWParams()
    
    Plummer_params: PlummerParams = PlummerParams()

    PointMass_params: PointMassParams = PointMassParams()
    

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
    
    external_accelerations: tuple = (NFW_POTENTIAL,)
    
