from typing import Optional, Tuple, Callable, Union, List
import numpy as np
import numpy.typing as npt
# from ..particles import Particles

class Particles:
    """
    Simple class to store the properties position, velocity, mass of the particles.
    Example:

    >>> from fireworks.particles import Particles
    >>> position=np.array([[1.,1.,1.],[2.,2.,2.],[3.,3.,3.]])
    >>> velocity=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    >>> mass=np.array([1.,1.,1.])
    >>> P=Particles(position,velocity,mass)
    >>> P.pos # particles'positions
    >>> P.vel # particles'velocities
    >>> P.mass # particles'masses
    >>> P.ID # particles'unique IDs

    The class contains also methods to estimate the radius of all the particles (:func:`~Particles.radius`),
    the module of the velociy of all the particles (:func:`~Particles.vel_mod`), and the module the positition and
    velocity of the centre of mass (:func:`~Particles.com_pos` and :func:`~Particles.com_vel`)

    >>> P.vel_mod() # return a Nx1 array with the module of the particle's velocity
    >>> P.com() # array with the centre of mass position (xcom,ycom,zcom)
    >>> P.com() # array with the centre of mass velocity (vxcom,vycom,vzcom)

    It is also possibile to set an acceleration for each particle, using the method set_acc
    Example:

    >>> acc= some_method_to_estimate_acc(P.position)
    >>> P.set_acc(acc)
    >>> P.acc # particles's accelerations

    Notice, if never initialised, P.acc is equal to None

    The class can be used also to estimate the total, kinetic and potential energy of the particles
    using the methods :func:`~Particles.Etot`, :func:`~Particles.Ekin`, :func:`~Particles.Epot`
    **NOTICE:** these methods need to be implemented by you!!!

    The method :func:`~Particles.copy` can be used to be obtaining a safe copy of the current
    Particles instances. Safe means that changing the members of the copied version will not
    affect the members or the original instance
    Example

    >>> P=Particles(position,velocity,mass)
    >>> P2=P.copy()
    >>> P2.pos[0] = np.array([10,10,10]) # P.pos[0] will not be modified!

    """
    def __init__(self, position: npt.NDArray[np.float64], 
                 velocity: npt.NDArray[np.float64], 
                 mass: npt.NDArray[np.float64]):
        """
        Class initialiser.
        It assigns the values to the class member pos, vel, mass and ID.
        ID is just a sequential integer number associated to each particle.

        :param position: A Nx3 numpy array containing the positions 
            of the N particles
        :param velocity: A Nx3 numpy array containing the velocity 
            of the N particles
        :param mass: A Nx1 numpy array containing the mass of the N particles
        """

        self.pos = np.array(np.atleast_2d(position), dtype=float)
        if self.pos.shape[1] != 3: print(f"Input position should contain a Nx3 array, current shape is {self.pos.shape}")

        self.vel = np.array(np.atleast_2d(velocity), dtype=float)
        if self.vel.shape[1] != 3: print(f"Input velocity should contain a Nx3 array, current shape is {self.pos.shape}")
        if len(self.vel) != len(self.pos): print(f"Position and velocity in input have not the same number of elements")

        self.mass = np.array(np.atleast_1d(mass), dtype=float)
        if len(self.mass) != len(self.pos): print(f"Position and mass in input have not the same number of elements")

        self.ID=np.arange(len(self.mass), dtype=int)

        self.acc=None

        self.flag=None

    def set_acc(self, acceleration: npt.NDArray[np.float64]):
        """
        Set the particle's acceleration

        :param acceleration: A Nx3 numpy array containing the acceleration 
        of the N particles
        """

        acc = np.atleast_2d(acceleration)
        if acceleration.shape[1] != 3: print(f"Input acceleration should contain \
                                            a Nx3 array, current shape is \
                                             {acc.shape}")

        self.acc=acc

    def vel_mod(self) -> npt.NDArray[np.float64]:
        """
        Estimate the module of the velocity of the particles

        :return: a Nx1 array containing the module of the particles's velocity
        """

        return np.sqrt(np.sum(self.vel*self.vel, axis=1))[:,np.newaxis]

    def com_pos(self) -> npt.NDArray[np.float64]:
        """
        Estimate the position of the centre of mass

        :return: a numpy  array with three elements corresponding to the centre 
        of mass position
        """

        return np.sum(self.mass*self.pos.T,axis=1)/np.sum(self.mass)

    def com_vel(self) -> npt.NDArray[np.float64]:
        """
        Estimate the velocity of the centre of mass

        :return: a numpy  array with three elements corresponding to centre 
        of mass velocity
        """

        return np.sum(self.mass*self.vel.T,axis=1)/np.sum(self.mass)

    def Ekin(self) -> float:
        """
        Estimate the total potential energy of the particles:
        Ekin=0.5 sum_i mi vi*vi

        :return: total kinetic energy
        """

        Ekin = 0.5 * np.sum(self.mass * (np.sum(self.vel**2, axis=1)))

        return Ekin

    def Epot(self, softening: float = 0.) -> float:
        """
        Estimate the total potential energy of the particles:
        Epot=-0.5 sumi sumj mi*mj / sqrt(rij^2 + eps^2)
        where eps is the softening parameter

        :param softening: Softening parameter
        :return: The total potential energy of the particles
        """

        # Calculate all pairwise distances between bodies
        rij = np.linalg.norm(self.pos[:, np.newaxis, :] - self.pos, axis=2)
        
        # Exclude self-distances (diagonal elements) to avoid division by zero
        np.fill_diagonal(rij, 1.0)
        
        # Calculate potential energy using vectorized operations
        Epot_mat = - np.outer(self.mass, self.mass) / \
            np.power((rij**2 + softening**2), 3/2)
        
        # Sum over all unique pairs
        Epot = np.sum(np.triu(Epot_mat, k=1))
        
        return Epot


    def Etot(self,softening: float = 0.) -> tuple[float,float,float]:
        """
        Estimate the total  energy of the particles: Etot=Ekintot + Epottot

        :param softening: Softening parameter
        :return: a tuple with

            - Total energy
            - Total kinetic energy
            - Total potential energy
        """

        Ekin = self.Ekin()
        Epot = self.Epot(softening=softening)
        Etot = Ekin + Epot

        return Etot, Ekin, Epot

    def copy(self):
        """
        Return a copy of this Particle class

        :return: a copy of the Particle class
        """

        par=Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))
        if self.acc is not None: par.acc=np.copy(self.acc)

        return Particles(np.copy(self.pos),np.copy(self.vel),np.copy(self.mass))

    def __len__(self) -> int:
        """
        Special method to be called when  this class is used as argument
        of the Python built-in function len()
        :return: Return the number of particles
        """

        return len(self.mass)

    # def __str__(self) -> str:
    #     """
    #     Special method to be called when  this class is used as argument
    #     of the Python built-in function print()
    #     :return: short info message
    #     """

    #     return f"Instance of the class Particles\nNumber of particles: {\
    #         self.__len__()}"

    # def __repr__(self) -> str:

    #     return self.__str__()


def ic_random_uniform(N: int, mass: list[float, float], pos: list[float, float], 
                      vel: list[float, float]) -> Particles:
    """
    Generate random initial condition drawing from a uniform distribution
    (between upper and lower boundary) for the mass, position and velocity.

    :param N: number of particles to generate
    :type N: int
    :param mass: list of lower and upper boundary for mass particle distribution
    :type mass: list of float
    :param pos: list of lower and upper boundary for position particle distribution
    :type pos: list of float
    :param vel: list of lower and upper boundary for velocity particle distribution
    :type vel: list of float
    :return: An instance of the class :class:`~fireworks.particles.Particles` 
    containing the generated particles, characterized by pos, vel and mass.
    """
    # Generate 1D array of N elements
    mass = np.random.uniform(low=mass[0], high=mass[1], size=N)
    # Generate 3xN 1D array and then reshape as a Nx3 array
    pos = np.random.uniform(low=pos[0], high=pos[1], size=3*N).reshape(N,3)
    # Generate 3xN 1D array and then reshape as a Nx3 array
    vel = np.random.uniform(low=vel[0], high=vel[1], size=3*N).reshape(N,3)

    return Particles(position=pos, velocity=vel, mass=mass)

'''def ic_random_normal(N: int, mass: float=1) -> Particles:
    """
    Generate random initial condition drawing from a normal distribution
    (centred in 0 and with dispersion 1) for the position and velocity.
    The mass is instead the same for all the particles.

    :param N: number of particles to generate
    :param mass: mass of the particles
    :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
    """
    # Generate 3xN 1D array and then reshape as a Nx3 array
    pos  = np.random.normal(size=3*N).reshape(N,3)
    # Generate 3xN 1D array and then reshape as a Nx3 array
    vel  = np.random.normal(size=3*N).reshape(N,3)
    mass = np.ones(N)*mass

    return Particles(position=pos, velocity=vel, mass=mass)'''

def ic_two_body(mass1: float, mass2: float, rp: float, e: float):
    """
    Create initial conditions for a two-body system.
    By default the two bodies will placed along the x-axis at the
    closest distance rp.
    Depending on the input eccentricity the two bodies can be in a
    circular (e<1), parabolic (e=1) or hyperbolic orbit (e>1).

    :param mass1:  mass of the first body [nbody units]
    :param mass2:  mass of the second body [nbody units]
    :param rp: closest orbital distance [nbody units]
    :param e: eccentricity
    :return: An instance of the class :class:`~fireworks.particles.Particles` containing the generated particles
    """

    Mtot=mass1+mass2

    if e==1.:
        vrel=np.sqrt(2*Mtot/rp)
    else:
        a=rp/(1-e)
        vrel=np.sqrt(Mtot*(2./rp-1./a))

    # To take the component velocities
    # V1 = Vcom - m2/M Vrel
    # V2 = Vcom + m1/M Vrel
    # we assume Vcom=0.
    v1 = -mass2/Mtot * vrel
    v2 = mass1/Mtot * vrel

    pos  = np.array([[0.,0.,0.],[rp,0.,0.]])
    vel  = np.array([[0.,v1,0.],[0.,v2,0.]])
    mass = np.array([mass1,mass2])

    return Particles(position=pos, velocity=vel, mass=mass)


def acceleration_direct(particles: Particles, softening: float =0.) \
        -> Tuple[npt.NDArray[np.float64]]:
    
  
    """
    Computes gravitational acceleration between particles using a direct method,
    considering optional softening.

    This function estimates the gravitational acceleration between particles 
    within a system. If 'softening' is provided as 0, a direct estimate with two
    nested for loop is used;
    otherwise, the specified 'softening' parameter is utilized.

    :param particles: An instance of the class Particles.
    :param softening: Softening parameter for gravitational calculations.
    :return: A tuple with 1 elements:

        - acc, Nx3 numpy array storing the acceleration for each particle.
    """
    # Using direct force estimate applcation2 - see slides Lecture 3 p.16
    def acc_2body(position_1, position_2, mass_2):
        
        """
        Implements definition of acceleration for two bodies i,j
        
        This is used in the following for loop
        """
        # Cartesian component of the i,j particles distance
        dx = position_1[0] - position_2[0]
        dy = position_1[1] - position_2[1]
        dz = position_1[2] - position_2[2]
        

        # Distance module
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Cartesian component of the i,j force
        acceleration = np.zeros(3)
        acceleration[0] = -mass_2 * dx / r**3
        acceleration[1] = -mass_2 * dy / r**3
        acceleration[2] = -mass_2 * dz / r**3

        return acceleration
        
    def acc_2body_softening(position_1, position_2, mass_2, softening):
        
        """
        Implements definition of acceleration for two bodies i,j 
        with Plummer softening 
        
        This is used in the following for loop
        """
        # Cartesian component of the i,j particles distance
        dx = position_1[0] - position_2[0]
        dy = position_1[1] - position_2[1]
        dz = position_1[2] - position_2[2]
        

        # Distance module
        r = np.sqrt(dx**2 + dy**2 + dz**2)

        # Cartesian component of the i,j force
        acceleration = np.zeros(3)
        acceleration[0] = -mass_2 * dx / (r**2 + softening**2)**(3/2)
        acceleration[1] = -mass_2 * dy / (r**2 + softening**2)**(3/2)
        acceleration[2] = -mass_2 * dz / (r**2 + softening**2)**(3/2)

        return acceleration
        

    pos  = particles.pos
    mass = particles.mass
    N    = len(particles) 

    # acc[i,:] ax,ay,az of particle i 
    acc  = np.zeros([N,3])

    # account for symmetry F_ij = -F_ji
    for i in range(N-1):
        for j in range(i+1,N):
            # Compute relative acceleration given
            # position of particle i and j
            mass_1 = mass[i]
            mass_2 = mass[j]

            if softening == 0: #if this condition is met, the others are not considered
                acc_ij = acc_2body(position_1=pos[i,:], position_2=pos[j,:], 
                                   mass_2=mass_2)
            else:
                acc_ij = acc_2body_softening(position_1=pos[i,:], 
                                              position_2=pos[j,:], 
                                              mass_2=mass_2,
                                              softening=softening)
        
            # Update array with accelerations
            acc[i,:] += acc_ij
            acc[j,:] -= mass_1 * acc_ij / mass_2 
            # because acc_2nbody already multiply by m[j]
        
    return acc


def integrator_leapfrog(particles: Particles,
                        tstep: float,
                        acceleration_estimator: Union[Callable,List],
                        softening: float = 0.):
    """
    Simple implementation of a symplectic Leapfrog integrator for N-body simulations.
    :param particles: Instance of the class :class:`~fireworks.particles.Particles`
    :param tstep: Times-step for current integration (notice some methods can 
    use smaller sub-time step to achieve the final result)
    :param acceleration_estimator: It needs to be a function from 
    `acceletation_direct`
    :param softening: softening parameter for the acceleration estimate, 
        can use 0 as default value
    
    :return: A tuple with 3 elements:

        - The updated particles instance
        - tstep, the effective timestep evolved in the simulation 
        (for some integrator this can be different wrt the input tstep)
        - acc, Nx3 numpy array storing the final acceleration for each particle,
            can be set to None
    """

    acc = acceleration_estimator(particles, softening)

    particles.pos = particles.pos + particles.vel*tstep + 0.5*acc*(tstep**2)

    acc2 = acceleration_estimator(
        Particles(particles.pos, particles.vel, particles.mass), 
        softening)

    particles.vel = particles.vel + 0.5*(acc + acc2)*tstep
    particles.set_acc(acc2)

    return (particles, tstep, acc2)

def integrator_rk4(particles: Particles,
                   tstep: float,
                   acceleration_estimator: Union[Callable,List],
                   softening: float = 0.):

    

    acc = acceleration_estimator(particles, softening)

    k1r = particles.vel*tstep
    k1v = acc*tstep

    acc2 = acceleration_estimator(Particles(particles.pos+(1/2)*k1r,
                                                         particles.vel + (1/2)*k1v,
                                                         particles.mass), 
                                                         softening)

    k2r = (particles.vel + 0.5*k1v)*tstep
    k2v = acc2*tstep

    acc3 = acceleration_estimator(Particles(particles.pos + (1/2)*k2r,
                                                         particles.vel + (1/2)*k2v,
                                                         particles.mass),
                                                         softening)

    k3r = (particles.vel + 0.5*k2v)*tstep
    k3v = acc3*tstep

    acc4 = acceleration_estimator(Particles(particles.pos + k3r, 
                                                         particles.vel + k3v, 
                                                         particles.mass), 
                                                         softening)

    k4r = (particles.vel + k3v)*tstep
    k4v = acc4*tstep

    particles.pos = particles.pos + (1/6)*(k1r + 2*k2r + 2*k3r + k4r)
    particles.vel = particles.vel + (1/6)*(k1v + 2*k2v + 2*k3v + k4v)
    particles.set_acc(acc4) 

    return (particles, tstep, acc4)


def adaptive_timestep_vel(particles: Particles, eta: float, 
                          acc = npt.NDArray[np.float64], 
                          tmin: Optional[float] = None, 
                          tmax: Optional[float] = None) -> float:

    # acc_mod = np.sqrt(np.sum(acc*acc, axis=1))[:,np.newaxis]

    ts = eta*np.nanmin(np.linalg.norm(particles.vel_mod(), axis=1)/\
                       np.linalg.norm(acc, axis=1))

    if tmin is not None: ts = max(ts, tmin)
    if tmax is not None: ts = min(ts, tmax)

    return ts
