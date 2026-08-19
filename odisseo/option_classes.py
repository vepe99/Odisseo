from typing import NamedTuple, Optional
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
HERMITE = 3  # nornax higher-order Hermite integrators (see odisseo.nornax_coupling)

#diffrax solvers
DOPRI5 = 0
TSIT5 = 1
SEMIIMPLICITEULER = 2
REVERSIBLEHEUN = 3
LEAPFROGMIDPOINT = 4
DOPRI8 = 5

#diffrax adjoint methods
RECURSIVECHECKPOINTADJOING = 0
FORWARDMODE = 1

#acceleartion schemes
DIRECT_ACC = 0
DIRECT_ACC_LAXMAP = 1
DIRECT_ACC_MATRIX = 2
DIRECT_ACC_FOR_LOOP = 3
DIRECT_ACC_SHARDING = 4
NO_SELF_GRAVITY = 5
FMM_ACC = 6

#external potential 
NFW_POTENTIAL = 0
POINT_MASS = 1
MN_POTENTIAL = 2
PSP_POTENTIAL = 3
LOGARITHMIC_POTENTIAL = 4
TRIAXIAL_NFW_POTENTIAL = 5
THIN_MN3_DISK = 6
THICK_MN3_DISK = 7
TWO_POWER_TRIAXIAL = 8

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
    NamedTuple containing the parameters for the Miyamoto-Nagai profile
    
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

class TriaxialNFWParams(NamedTuple):
    Mvir: float = 1.62*1e11      # Total mass
    r_s: float = 15.3    # Scale radius
    q1: float = 1.0  # y-axis flattening (q1=1 is spherical)
    q2: float = 1.0  # z-axis flattening (q2=1 is spherical)


class ThinMN3DiskParams(NamedTuple):
    """
    NamedTuple containing the parameters for the thin Miyamoto-Nagai 3 disk potential
    """

    M: float = 1e10  #M☉

    hr: float = 3.0  #kpc

    hz: float = 0.3  #kpc

class ThickMN3DiskParams(NamedTuple):
    """
    NamedTuple containing the parameters for the thick Miyamoto-Nagai 3 disk potential
    """

    M: float = 5e9  #M☉

    hr: float = 3.0  #kpc

    hz: float = 1.0  #kpc

class TwoPowerTriaxialParams(NamedTuple):
    """
    NamedTuple containing the parameters for the two power-law triaxial potential
    """

    rho: float = 0.015  #Density normalization in M☉/pc^3

    a: float = 20.0  #Scale radius in kpc

    b: float = 1.0  #Intermediate axis ratio

    c: float = 1.0  #Minor axis ratio

    alpha: float = 1.0  #Inner slope

    beta: float = 3.0  #Outer slope

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

    Logarithmic_params: LogarithmicParams = LogarithmicParams()

    TriaxialNFW_params: TriaxialNFWParams = TriaxialNFWParams()

    ThinMN3Disk_params: ThinMN3DiskParams = ThinMN3DiskParams()

    ThickMN3Disk_params: ThickMN3DiskParams = ThickMN3DiskParams()

    TwoPowerTriaxial_params: TwoPowerTriaxialParams = TwoPowerTriaxialParams()
        
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

    # nornax Hermite integrator tuning (used when integrator == HERMITE).
    # The self-gravity backend is chosen by acceleration_scheme: FMM_ACC uses
    # jaccpot's FMM (forward-only), any direct scheme uses nornax's
    # differentiable DirectSumGravity. hermite_order 6/8 requires the
    # direct-sum backend today (jaccpot adapter is capped at order 4).
    hermite_order: int = 4
    hermite_eta: float = 0.02
    hermite_atol: float = 1e-5
    hermite_min_dt: float = 1e-8
    hermite_max_dt: float = 1e-1
    hermite_jerk_mode: str = "fast_approx"

    # Jaccpot-FMM backend tuning (used by integrate API).
    fmm_refresh_every: int = 1
    fmm_leaf_size: int = 16
    fmm_max_order: int = 4
    fmm_refresh_after_position_update: bool = False

    # Jaccpot solver tuning knobs exposed to ODISSEO.
    fmm_preset: str = "fast"
    # Real (Dehnen) harmonics is the production default: the radix large-N fast
    # lane runs pure-real end to end (no complex<->real conversion). Use
    # "solidfmm"/"complex" only for cross-checking.
    fmm_basis: str = "real"
    fmm_theta: float = 0.6
    fmm_runtime_path: str = "auto"
    fmm_mac_type: str = "dehnen"
    fmm_farfield_mode: str = "auto"
    fmm_m2l_chunk_size: Optional[int] = None
    fmm_nearfield_mode: str = "auto"
    fmm_nearfield_edge_chunk_size: int = 256
    # static_radix is the production tree build for the large-N GPU fast lane.
    fmm_tree_build_mode: str = "static_radix"
    fmm_tree_leaf_target: int = 32
    # Pallas fused near-field/M2L kernels. None => auto (ON for Ampere sm_80+,
    # pure-JAX on sm_75/CPU). The ODISSEO_FMM_USE_PALLAS env var overrides this.
    fmm_use_pallas: Optional[bool] = None
    fmm_fixed_order: Optional[int] = None
    fmm_jit_tree: Optional[bool] = None
    fmm_jit_traversal: Optional[bool] = True
    fmm_max_pair_queue: Optional[int] = None
    fmm_pair_process_block: Optional[int] = None
    fmm_max_interactions_per_node: Optional[int] = None
    fmm_max_neighbors_per_leaf: Optional[int] = None
    fmm_prepare_stage_memory_split_enabled: Optional[bool] = None
    fmm_upward_leaf_batch_size: Optional[int] = None
    fmm_auto_large_n_profile: bool = True
    fmm_large_n_min_particles: int = 200_000
    fmm_large_n_force_fp32: bool = True
    fmm_large_n_target_block_size: Optional[int] = None
    fmm_large_n_static_target_blocks: Optional[bool] = None
    fmm_large_n_static_target_blocks_max_per_leaf: Optional[int] = None
    fmm_large_n_environment_overrides_enabled: bool = True
    # Static-shape/compile-stability experiment knobs.
    fmm_enforce_static_shape_contract: bool = False
    fmm_static_shape_warmup_prepares: int = 0
    fmm_rematerialize_between_refresh: bool = True
    # Differentiable FMM: gradients w.r.t. external-potential parameters, the
    # initial state and masses. Routes FMM_ACC through odisseo.differentiable
    # instead of the forward-throughput coupler -- the tree topology is frozen
    # for the whole call and self-gravity is re-evaluated from the live
    # positions, so jax.grad flows. See odisseo/differentiable.py.
    fmm_differentiable: bool = False
    # jaccpot GradConfig knobs worth surfacing here. "auto" takes the bucketed
    # near-field reverse below 100k particles and the leaf-major fast lane at or
    # above it (the bucketed reverse OOMs at galaxy scale).
    fmm_grad_nearfield_lane: str = "auto"
    fmm_grad_fused_m2l_pallas: Optional[bool] = None
    # Adaptive FMM (diffrax) controls.
    fmm_adaptive_refresh_rhs_calls: int = 1
    fmm_adaptive_refresh_displacement_threshold: Optional[float] = None
    fmm_adaptive_max_dt: Optional[float] = None
    fmm_adaptive_min_dt: Optional[float] = None
    fmm_adaptive_rtol: float = 1e-3
    fmm_adaptive_atol: float = 1e-6
    fmm_adaptive_use_dense_output: bool = False

    batch_size: int = 10_000

    double_map: bool = False

    external_accelerations: tuple = ()

    differentation_mode: int = BACKWARDS

    diffrax_adjoint_method: int = RECURSIVECHECKPOINTADJOING

    num_checkpoints: int = 100

    progress_bar: bool = False

    gradient_horizon: int = 0

    sech2_z: bool = False  #whether to use sech^2 (True) or exponential vertical profile (False, default) for MN3 disk potential

    MN3_positive_density: bool = True  #whether to enforce positive density everywhere for MN3 disk potential

    glorder: int = 50 #order of Gauss-Legendre quadrature for MN3 disk potential
