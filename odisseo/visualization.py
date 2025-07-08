import matplotlib.pyplot as plt
plt.style.use('dark_background')
import math
from random import randint, uniform, random

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
from astropy import units as u
from astropy.coordinates import SkyCoord



def energy_angular_momentum_plot(snapshots, code_units, filename=None):
    """
    Plots the relative change in total energy and angular momentum of the system over time.

    Args:
        snapshots (odisseo.Snapshots): Snapshots object containing the simulation data.
        code_units (odisseo.CodeUnits): CodeUnits object containing the units of the simulation.
        filename (str, optional): The filename to save the plot to. If None, the plot will be displayed but not saved.

    Returns:
        None
    """
    fig = plt.figure(figsize=(17, 5), tight_layout=True)
    ax = fig.add_subplot(121)
    Delta_E = ((snapshots.total_energy - snapshots.total_energy[0])/snapshots.total_energy[0])
    ax.plot((snapshots.times * code_units.code_time).to(u.Gyr), 100*Delta_E,)
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel(r'$(E - E_0)/E_0 \% $')
    ax.set_ylim(-100, 100)
    ax.axhline(5, color='r', linestyle='--', label='5%')
    ax.axhline(-5, color='r', linestyle='--', )
    ax.grid(linestyle='dotted')
    ax.legend()

    ax = fig.add_subplot(122)
    Delta_AngMom = ((snapshots.angular_momentum - snapshots.angular_momentum[0])/snapshots.angular_momentum[0])
    ax.plot((snapshots.times * code_units.code_time).to(u.Gyr), 100*Delta_AngMom[:, 2], )
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel(r'$(L - L_0)/L_0 \% $')
    ax.set_ylim(-100, 100)
    ax.axhline(5, color='r', linestyle='--', label='5%')
    ax.axhline(-5, color='r', linestyle='--', )
    ax.grid(linestyle='dotted')
    ax.legend()

    if filename is not None:
        fig.savefig(filename)
    plt.show()

def plot_last_snapshot(snapshots, code_units, rp, plotting_units_length, filename=None):
    """
    Plots the last snapshot of the particles in 3D space.

    Args:
        snapshots (object): An object containing the states of the particles at different time steps.
        code_units (object): An object containing the code units for length conversion.
        plotting_units_length (object): The units to which the lengths should be converted for plotting.
        filename (str, optional): The filename to save the plot. If None, the plot is not saved (default is None).

    Returns:
        None
    """

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(f'X {plotting_units_length}')
    ax.set_ylabel(f'Y {plotting_units_length}')
    ax.set_zlabel(f'Z {plotting_units_length}')

    ax.scatter((snapshots.states[-1, :, 0, 0]* code_units.code_length).to(plotting_units_length).value,
                (snapshots.states[-1, :, 0, 1]* code_units.code_length).to(plotting_units_length).value,
                (snapshots.states[-1, :, 0, 2]* code_units.code_length).to(plotting_units_length).value)
    ax.scatter(0, 0, 0, s=100, marker='*', color='r')
    ax.set_xlim(-(rp* code_units.code_length).to(plotting_units_length).value, (rp* code_units.code_length).to(plotting_units_length).value)
    ax.set_ylim(-(rp* code_units.code_length).to(plotting_units_length).value, (rp* code_units.code_length).to(plotting_units_length).value)
    ax.set_zlim(-(rp* code_units.code_length).to(plotting_units_length).value, (rp* code_units.code_length).to(plotting_units_length).value)
    
    if filename is not None:
        fig.savefig(filename)
    
    plt.show()

def plot_orbit(snapshots, ax_lim, code_units, plotting_units_length, config, filename=None):
    """
    Plots the orbit of particles in 3D space.

    Args:
        snapshots (object): An object containing the states of the particles at different time steps.
        ax_lim (float): The limit for the axes in code units.
        code_units (object): An object containing the code units for length conversion.
        plotting_units_length (object): The units to which the lengths should be converted for plotting.
        config (object): Configuration object containing the number of particles (N_particles).
        filename (str, optional): The filename to save the plot. If None, the plot is not saved (default is None).
    Raises:
        AssertionError: If the number of particles in config.N_particles is 10 or more.
    Returns:
        None
    """
    
    assert config.N_particles < 10, "Too many particles! "

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(f'X {plotting_units_length}')
    ax.set_ylabel(f'Y {plotting_units_length}')
    ax.set_zlabel(f'Z {plotting_units_length}')
    ax.set_xlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
    ax.set_ylim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
    ax.set_zlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)

    colors = plt.get_cmap("tab10").colors
    for i in range(config.N_particles):
        ax.plot((snapshots.states[:, i, 0, 0]* code_units.code_length).to(plotting_units_length).value,
                (snapshots.states[:, i, 0, 1]* code_units.code_length).to(plotting_units_length).value,
                (snapshots.states[:, i, 0, 2]* code_units.code_length).to(plotting_units_length).value, label=f'Particle {i}', color=colors[i])
        ax.scatter((snapshots.states[-1, i, 0, 0]* code_units.code_length).to(plotting_units_length).value,
                     (snapshots.states[-1, i, 0, 1]* code_units.code_length).to(plotting_units_length).value,
                     (snapshots.states[-1, i, 0, 2]* code_units.code_length).to(plotting_units_length).value, s=25, marker='o', color=colors[i],)
    ax.legend()
    if filename is not None:
        fig.savefig(filename)
    plt.show()

def plot_sky_projection(snapshots, code_units, plotting_units_length, filename=None):
    """
    Plots the sky projection of the particles in the simulation.

    Args:
        snapshots (odisseo.Snapshots): Snapshots object containing the simulation data.
        code_units (odisseo.CodeUnits): CodeUnits object containing the units of the simulation.
        plotting_units_length (astropy.units.Quantity): The unit to which the lengths should be converted for plotting.
        filename (str, optional): The filename to save the plot to. If None, the plot will be displayed but not saved.

    Returns:
        None

    """

    # Example: 3D Cartesian coordinates in kpc
    x = (snapshots.states[-1, :, 0, 0]*code_units.code_length).to(plotting_units_length)
    y = (snapshots.states[-1, :, 0, 1]*code_units.code_length).to(plotting_units_length)
    z = (snapshots.states[-1, :, 0, 2]*code_units.code_length).to(plotting_units_length)

    # Observer's position at (-8, 0, 0) kpc
    x_obs, y_obs, z_obs = -8 * u.kpc, 0 * u.kpc, 0 * u.kpc

    # Shift to observer's frame
    x_rel = x - x_obs
    y_rel = y - y_obs
    z_rel = z - z_obs

    # Convert to Galactic longitude l and latitude b
    distance = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
    l = np.arctan2(y_rel, x_rel).to(u.deg)
    b = np.arcsin(z_rel / distance).to(u.deg)

    # Convert to Astropy SkyCoord object (if needed)
    galactic_coords = SkyCoord(l=l, b=b, distance=distance, frame="galactic")

    # Convert to Equatorial (RA, Dec) if needed
    equatorial_coords = galactic_coords.transform_to("icrs")

    # Get sky-plane projection
    ra = equatorial_coords.ra
    dec = equatorial_coords.dec


    # Convert longitude to range [-180, 180] for better visualization
    l_wrap = (l + 180 * u.deg) % (360 * u.deg) - 180 * u.deg

    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, projection= 'aitoff')
    ax.scatter(l_wrap, b, s=1, color='blue', alpha=0.5)
    ax.set_xlabel("Galactic Longitude l (deg)")
    ax.set_ylabel("Galactic Latitude b (deg)")
    ax.set_title("Sky Projection in Galactic Coordinates")
    ax.grid(True, linestyle="--", alpha=0.5)

    if filename is not None:
        plt.savefig(filename)

    plt.show()



def create_3d_gif(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, filename=None):
    """
    Create a 3D GIF animation from a series of snapshots.

    Args:
        snapshots (object): An object containing the states and times of the snapshots to be animated.
        ax_lim (float): The axis limit for the 3D plot.
        code_units (object): An object containing the code units for length and time.
        plotting_units_length (astropy.units.Unit): The units for plotting the length.
        plot_units_time (astropy.units.Unit): The units for plotting the time.
        filename (str, optional): The filename to save the GIF. If None, the GIF will not be saved.

    Returns:
        None

    """

    # Create a figure for plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()



    # Create lists of star positions for galaxy:
    leading_arm, trailing_arm = build_spiral_arms(b=-0.3, arms_info=arms_info)
    core_stars = build_core_stars(SCALE)
    inner_haze_stars = haze(SCALE, r_mult=2, z_mult=0.5, density=5)
    outer_haze_stars = haze(SCALE, r_mult=1, z_mult=0.3, density=5) 

    color_milky_way = 'w'

    ax.scatter(*zip(*leading_arm), c=color_milky_way, marker='.', s=1, alpha=0.5)
    ax.scatter(*zip(*trailing_arm), c=color_milky_way, marker='.', s=1, alpha=0.5)
    ax.scatter(*zip(*core_stars), c=color_milky_way, marker='.', s=1, alpha=0.5)
    ax.scatter(*zip(*inner_haze_stars), c=color_milky_way, marker='.', s=1, alpha=0.5)
    ax.scatter(*zip(*outer_haze_stars), c=color_milky_way, marker='.', s=1, alpha=0.5)

    # Initialize the scatter plot
    scatter1 = ax.scatter([], [], [], c='b')
    scatter2 = ax.scatter([], [], [], c='r', marker='*')

    def init():
        ax.set_xlabel(f'X {plotting_units_length}')
        ax.set_ylabel(f'Y {plotting_units_length}')
        ax.set_zlabel(f'Z {plotting_units_length}')
        return scatter1, scatter2

    def update(frame):
        ax.clear()
        ax.scatter(*zip(*leading_arm), c=color_milky_way, marker='.', s=1, alpha=0.5)
        ax.scatter(*zip(*trailing_arm), c=color_milky_way, marker='.', s=1, alpha=0.5)
        ax.scatter(*zip(*core_stars), c=color_milky_way, marker='.', s=1, alpha=0.5)
        ax.scatter(*zip(*inner_haze_stars), c=color_milky_way, marker='.', s=1, alpha=0.5)
        ax.scatter(*zip(*outer_haze_stars), c=color_milky_way, marker='.', s=1, alpha=0.5)
        ax.set_xlabel(f'X {plotting_units_length}')
        ax.set_ylabel(f'Y {plotting_units_length}')
        ax.set_zlabel(f'Z {plotting_units_length}')
        ax.set_xlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax.set_ylim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax.set_zlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax.set_title(f'Time: {(snapshots.times[frame] * code_units.code_time).to(plot_units_time):.2f} ')
        scatter1 = ax.scatter((snapshots.states[frame, :, 0, 0]* code_units.code_length).to(plotting_units_length).value, 
                              (snapshots.states[frame, :, 0, 1]* code_units.code_length).to(plotting_units_length).value, 
                              (snapshots.states[frame, :, 0, 2]* code_units.code_length).to(plotting_units_length).value, c='b', s=1)
        # scatter2 = ax.scatter(0, 0, 0, c='r', s=100, marker='*')
        return scatter1, scatter2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, len(snapshots.states), 1), init_func=init, blit=False)

    if filename is not None:
        # Save the animation as a GIF
        anim.save(filename, writer=PillowWriter(fps=10))

def create_3d_gif_velocitycoding(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, vmin=None, vmax=None, filename=None):
    
    # Create a figure for plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate velocity norms across all frames if vmin/vmax not provided
    if vmin is None or vmax is None:
        all_velocity_norms = jnp.linalg.norm(snapshots.states[:, :, 1], axis=2)
        vmin = vmin if vmin is not None else jnp.min(all_velocity_norms)
        vmax = vmax if vmax is not None else jnp.max(all_velocity_norms)
    
    # Store colorbar reference
    cbar = None
    

    # Initialize the scatter plot
    scatter1 = ax.scatter([], [], [], )
    scatter2 = ax.scatter([], [], [], c='r', marker='*')

    

    def init():
        ax.set_xlabel(f'X {plotting_units_length}')
        ax.set_ylabel(f'Y {plotting_units_length}')
        ax.set_zlabel(f'Z {plotting_units_length}')
        return scatter1, scatter2

    def update(frame):
        nonlocal cbar
        ax.clear()

        ax.set_xlabel(f'X {plotting_units_length}')
        ax.set_ylabel(f'Y {plotting_units_length}')
        ax.set_zlabel(f'Z {plotting_units_length}')
        ax.set_xlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax.set_ylim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax.set_zlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax.set_title(f'Time: {(snapshots.times[frame] * code_units.code_time).to(plot_units_time):.2f} ')
        
        velocity_norms = jnp.linalg.norm(snapshots.states[frame, :, 1], axis=1)

        scatter1 = ax.scatter((snapshots.states[frame, :, 0, 0]* code_units.code_length).to(plotting_units_length).value, 
                              (snapshots.states[frame, :, 0, 1]* code_units.code_length).to(plotting_units_length).value, 
                              (snapshots.states[frame, :, 0, 2]* code_units.code_length).to(plotting_units_length).value, 
                              c=velocity_norms,
                              s=1)
        scatter2 = ax.scatter(0, 0, 0, c='r', s=100, marker='*')

        if cbar is not None:
            cbar.remove()
        cbar = fig.colorbar(scatter1, ax=ax, shrink=0.8, pad=0.1)
        # vel_label = f"Velocity" if code_units.velocity_units is None else f"Velocity [{code_units.velocity_units}]"
        # cbar.set_label(vel_label)
        
        return scatter1, scatter2

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, len(snapshots.states), 1), init_func=init, blit=False)

    if filename is not None:
        # Save the animation as a GIF
        anim.save(filename, writer=PillowWriter(fps=10))

def create_projection_gif(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, filename=None):
    """
    Create a GIF animation of 3D projections from simulation snapshots.
    This function generates a GIF animation showing the evolution of 3D projections
    of simulation data over time. The projections are displayed in three subplots:
    X-Y, X-Z, and Y-Z planes.

    Args:
        snapshots (object): An object containing the simulation snapshots. It should have attributes
            `states` and `times`, where `states` is a 4D array with shape 
            (num_frames, num_particles, 1, 3) representing the positions of particles
            in each frame, and `times` is a 1D array representing the time of each frame.
        ax_lim (float): The axis limit for the plots, in code units.
        code_units (object): An object containing the code units for length and time. It should have
        attributes `code_length` and `code_time`.
        plotting_units_length (astropy.units.Unit): The units for plotting the lengths.
        plot_units_time (astropy.units.Unit): The units for plotting the time.
        filename (str, optional): The filename to save the GIF animation. If None, the animation is not saved.

    Returns:
        None

    """

    # Create a figure for plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Initialize the scatter plots
    scatter1 = ax1.scatter([], [], c='b')
    scatter2 = ax1.scatter([], [], c='r', marker='*')
    scatter3 = ax2.scatter([], [], c='b')
    scatter4 = ax2.scatter([], [], c='r', marker='*')
    scatter5 = ax3.scatter([], [], c='b')
    scatter6 = ax3.scatter([], [], c='r', marker='*')

    def init():
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
            ax.set_ylim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        ax1.set_xlabel(f'X {plotting_units_length}')
        ax1.set_ylabel(f'Y {plotting_units_length}')
        ax2.set_xlabel(f'X {plotting_units_length}')
        ax2.set_ylabel(f'Z {plotting_units_length}')
        ax3.set_xlabel(f'Y {plotting_units_length}')
        ax3.set_ylabel(f'Z {plotting_units_length}')
        return scatter1, scatter2, scatter3, scatter4, scatter5, scatter6

    def update(frame):
        fig.suptitle(f'Time: {(snapshots.times[frame]*code_units.code_time).to(plot_units_time):.2f}')

        for ax in [ax1, ax2, ax3]:
            ax.clear()
            ax.set_xlim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
            ax.set_ylim(-(ax_lim* code_units.code_length).to(plotting_units_length).value, (ax_lim* code_units.code_length).to(plotting_units_length).value)
        
        ax1.set_xlabel(f'X {plotting_units_length}')
        ax1.set_ylabel(f'Y {plotting_units_length}')
        ax1.grid(linestyle='dotted')
        scatter1 = ax1.scatter((snapshots.states[frame, :, 0, 0] * code_units.code_length).to(plotting_units_length).value, 
                               (snapshots.states[frame, :, 0, 1] * code_units.code_length).to(plotting_units_length).value, c='b', s=1)
        scatter2 = ax1.scatter(0, 0, c='r', s=100, marker='*')
        
        ax2.set_xlabel(f'X {plotting_units_length}')
        ax2.set_ylabel(f'Z {plotting_units_length}')
        ax2.grid(linestyle='dotted')
        scatter3 = ax2.scatter((snapshots.states[frame, :, 0, 0] * code_units.code_length).to(plotting_units_length).value, 
                               (snapshots.states[frame, :, 0, 2]* code_units.code_length).to(plotting_units_length).value, c='b', s=1)
        scatter4 = ax2.scatter(0, 0, c='r', s=100, marker='*')
        
        ax3.set_xlabel(f'Y {plotting_units_length}')
        ax3.set_ylabel(f'Z {plotting_units_length}')
        ax3.grid(linestyle='dotted')
        scatter5 = ax3.scatter((snapshots.states[frame, :, 0, 1] * code_units.code_length).to(plotting_units_length).value, 
                               (snapshots.states[frame, :, 0, 2] * code_units.code_length).to(plotting_units_length).value, c='b', s=1)
        scatter6 = ax3.scatter(0, 0, c='r', s=100, marker='*')
        
        return scatter1, scatter2, scatter3, scatter4, scatter5, scatter6

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, len(snapshots.states), 1), init_func=init, blit=False)

    if filename is not None:
        # Save the animation as a GIF
        anim.save(filename, writer=PillowWriter(fps=10))


# plt.style.use('dark_background')

# Set the radius of the galactic disc (scaling factor):
SCALE = 26  # Use range of 200 - 700.

def build_spiral_stars(b, r, rot_fac, fuz_fac):
    """Return list of (x,y,z) points for a logarithmic spiral.

    b = constant for spiral direction and "openness"
    r = scale factor (galactic disc radius)
    rot_fac = factor to rotate each spiral arm
    fuz_fac = randomly shift star position; applied to 'fuzz' variable
    """
    fuzz = int(0.030 * abs(r))  # Scalable initial amount to shift locations.
    num_stars = 1000
    spiral_stars = []
    for i in range(0, num_stars):
        theta = math.radians(i)
        x = r * math.exp(b*theta) * math.cos(theta - math.pi * rot_fac) - randint(-fuzz, fuzz) * fuz_fac
        y = r * math.exp(b*theta) * math.sin(theta - math.pi * rot_fac) - randint(-fuzz, fuzz) * fuz_fac
        z = uniform((-SCALE / (SCALE * 3)), (SCALE / (SCALE * 3)))
        spiral_stars.append((x, y, z))
    return spiral_stars

# Assign scale factor, rotation factor, and fuzz factor for spiral arms.
# Each arm is a pair: leading arm + trailing arm:
arms_info = [(SCALE, 1, 1.5), (SCALE, 0.91, 1.5), 
             (-SCALE, 1, 1.5), (-SCALE, -1.09, 1.5),
             (-SCALE, 0.5, 1.5), (-SCALE, 0.4, 1.5), 
             (-SCALE, -0.5, 1.5), (-SCALE, -0.6, 1.5)]

def build_spiral_arms(b, arms_info):
    """Return lists of point coordinates for galactic spiral arms.

    b = constant for spiral direction and "openness"
    arms_info = list of scale, rotation, and fuzz factors
    """
    leading_arms = []
    trailing_arms = []
    for i, arm_info in enumerate(arms_info):
        arm = build_spiral_stars(b=b, 
                                 r=arm_info[0], 
                                 rot_fac=arm_info[1], 
                                 fuz_fac=arm_info[2])
        if i % 2 != 0:
            leading_arms.extend(arm)
        else:
            trailing_arms.extend(arm)            
    return leading_arms, trailing_arms

def spherical_coords(num_pts, radius):
    """Return list of uniformly distributed points in a sphere."""
    position_list = []
    for _ in range(num_pts):
        coords = np.random.normal(0, 1, 3)
        coords *= radius
        coords[2] *= 0.02  # Reduce z range for matplotlib default z-scale.
        position_list.append(list(coords))
    return position_list

def build_core_stars(scale_factor):
    """Return lists of point coordinates for galactic core stars."""
    core_radius = scale_factor / 15
    num_rim_stars = 3000
    outer_stars = spherical_coords(num_rim_stars, core_radius)
    inner_stars = spherical_coords(int(num_rim_stars/4), core_radius/2.5)
    return (outer_stars + inner_stars)

def haze(scale_factor, r_mult, z_mult, density):
    """Generate uniform random (x,y,z) points within a disc for 2-D display.

    scale_factor = galactic disc radius
    r_mult = scalar for radius of disc
    z_mult = scalar for z values
    density = multiplier to vary the number of stars posted
    """
    haze_coords = []
    for _ in range(0, scale_factor * density):
        n = random()
        theta = uniform(0, 2 * math.pi)
        x = round(math.sqrt(n) * math.cos(theta) * scale_factor) / r_mult
        y = round(math.sqrt(n) * math.sin(theta) * scale_factor) / r_mult
        z = np.random.uniform(-1, 1) * z_mult
        haze_coords.append((x, y, z))
    return haze_coords