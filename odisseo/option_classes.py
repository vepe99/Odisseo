from typing import NamedTuple
from astropy import units as u
from astropy import constants as c
from math import log


# differentiation modes
FORWARDS = 0
BACKWARDS = 1

#integratiopn schemes
LEAPFROG = 0
RK4 = 1
DIFFRAX_BACKEND = 2

#diffrax solvers
DOPRI5 = 0
TSIT5 = 1
SEMIIMPLICITEULER = 2
REVERSIBLEHEUN = 3
LEAPFROGMIDPOINT = 4

#acceleartion schemes
DIRECT_ACC = 0
DIRECT_ACC_LAXMAP = 1
DIRECT_ACC_MATRIX = 2
DIRECT_ACC_FOR_LOOP = 3
DIRECT_ACC_SHARDING = 4

#external potential 
NFW_POTENTIAL = 0
POINT_MASS = 1
MN_POTENTIAL = 2
PSP_POTENTIAL = 3
LOGARITHMIC_POTENTIAL = 4

class PlummerParams(NamedTuple):
    """
    NamedTuple containing the parameters for the Plummer profile
    """
    
    a: float = 7 #kpc
    
    Mtot: float = 1.0 #M☉

class NFWParams(NamedTuple):
    """
    NamedTuple containing the parameters for the NFW profile
    """
    
    Mvir: float = 1.62*1e11 #M☉
    
    r_s: float = 15.3 #kpc
    
    # c: float = 10

    # d_c: float = log(1+c) - c/(1+c)

class PointMassParams(NamedTuple):
    """
    NamedTuple containing the parameters for the point mass
    """

    M: float = 1.0 #M☉
    
class MNParams(NamedTuple):
    """
    NamedTuple containing the parameters for the Myamoto Nagai profile
    
    """

    M: float = 6.5e10 #M☉

    a: float = 3.0 #kpc

    b: float = 0.28 #kpc

class PSPParams(NamedTuple):

    M: float = 4501365375.06545 #M☉

    alpha: float = 1.8 

    r_c: float   = 1.9 #kpc

class LogarithmicParams(NamedTuple):
    """
    NamedTuple containing the parameters for the logarithmic potential
    """

    v0: float = 220.0 #km/s
    
    q: float = 0.9 #flattening parameter


class SimulationParams(NamedTuple):
    """
    NamedTuple containing the parameters for the simulation. This parameter do not require recompilation
    """
    
    G: float = 1.0
    
    t_end: float = 1.0  #In code_units by setting G=1

    Plummer_params: PlummerParams = PlummerParams()

    NFW_params: NFWParams = NFWParams()

    PointMass_params: PointMassParams = PointMassParams()

    MN_params: MNParams = MNParams()

    PSP_params: PSPParams = PSPParams()

    Logarithmic_Params: LogarithmicParams = LogarithmicParams()
        
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
    
    softening: float = 1e-10
    
    integrator: int = LEAPFROG

    diffrax_solver: int = DOPRI5
    
    acceleration_scheme: int = DIRECT_ACC

    batch_size: int = 10_000

    double_map: bool = False

    external_accelerations: tuple = ()

    differentation_mode: int = BACKWARDS

    num_checkpoints: int = 100

    progress_bar: bool = True

    
