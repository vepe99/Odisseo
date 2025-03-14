import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
from astropy import units as u
from astropy.coordinates import SkyCoord



def energy_angular_momentum_plot(snapshots, code_units, filename=None):
    """
    Plots the relative change in total energy and angular momentum of the system over time.
    
    Parameters
    ----------
    snapshots : odisseo.Snapshots
        Snapshots object containing the simulation data.
    code_units : odisseo.CodeUnits
        CodeUnits object containing the units of the simulation.
    filename : str, optional
        The filename to save the plot to. If None, the plot
        will be displayed but not saved.
    
    Returns
    -------
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

def plot_orbit(snapshots, ax_lim, code_units, plotting_units_length, config, filename=None):
    
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

    Parameters
    ----------
    snapshots : odisseo.Snapshots
        Snapshots object containing the simulation data.
    code_units : odisseo.CodeUnits
        CodeUnits object containing the units of the simulation.
    filename : str, optional
        The filename to save the plot to. If None, the plot
        will be displayed but not saved.

    Returns
    -------
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
    # Create a figure for plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

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
        scatter2 = ax.scatter(0, 0, 0, c='r', s=100, marker='*')
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



