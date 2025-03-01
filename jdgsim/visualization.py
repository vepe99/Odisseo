import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


####PLOT MAKER####



######GIF MAKER####
def create_3d_gif(snapshots, rp, code_units, plotting_units_length, plot_units_time, filename=None):
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
        ax.set_xlim(-(rp* code_units.code_length).to(plotting_units_length).value, (rp* code_units.code_length).to(plotting_units_length).value)
        ax.set_ylim(-(rp* code_units.code_length).to(plotting_units_length).value, (rp* code_units.code_length).to(plotting_units_length).value)
        ax.set_zlim(-(rp* code_units.code_length).to(plotting_units_length).value, (rp* code_units.code_length).to(plotting_units_length).value)
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

def create_projection_gif(snapshots, rp, code_units, plotting_units_length, plot_units_time, filename=None):

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
            ax.set_xlim(-rp, rp)
            ax.set_ylim(-rp, rp)
        ax1.set_xlabel(f'X {plotting_units_length}')
        ax1.set_ylabel(f'Y {plotting_units_length}')
        ax2.set_xlabel(f'X {plotting_units_length}')
        ax2.set_ylabel(f'Z {plotting_units_length}')
        ax3.set_xlabel(f'Y {plotting_units_length}')
        ax3.set_ylabel(f'Z {plotting_units_length}')
        return scatter1, scatter2, scatter3, scatter4, scatter5, scatter6

    def update(frame):
        fig.suptitle(f'Time: {(snapshots.times[frame]*code_units.code_time).to(u.Gyr):.2f}')

        for ax in [ax1, ax2, ax3]:
            ax.clear()
            ax.set_xlim(-(rp.code_units.code_length).to(plotting_units_length).value, (rp.code_units.code_length).to(plotting_units_length).value)
            ax.set_ylim(-(rp.code_units.code_length).to(plotting_units_length).value, (rp.code_units.code_length).to(plotting_units_length).value)
        
        ax1.set_xlabel(f'X {plotting_units_length}')
        ax1.set_ylabel(f'Y {plotting_units_length}')
        scatter1 = ax1.scatter(snapshots.states[frame, :, 0, 0], snapshots.states[frame, :, 0, 1], c='b', s=1)
        scatter2 = ax1.scatter(0, 0, c='r', s=100, marker='*')
        
        ax2.set_xlabel(f'X {plotting_units_length}')
        ax2.set_ylabel(f'Z {plotting_units_length}')
        scatter3 = ax2.scatter(snapshots.states[frame, :, 0, 0], snapshots.states[frame, :, 0, 2], c='b', s=1)
        scatter4 = ax2.scatter(0, 0, c='r', s=100, marker='*')
        
        ax3.set_xlabel(f'Y {plotting_units_length}')
        ax3.set_ylabel(f'Z {plotting_units_length}')
        scatter5 = ax3.scatter(snapshots.states[frame, :, 0, 1], snapshots.states[frame, :, 0, 2], c='b', s=1)
        scatter6 = ax3.scatter(0, 0, c='r', s=100, marker='*')
        
        return scatter1, scatter2, scatter3, scatter4, scatter5, scatter6

    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(0, len(snapshots.states), 1), init_func=init, blit=False)

    # Save the animation as a GIF
    anim.save(filename, writer=PillowWriter(fps=10))



