:py:mod:`odisseo.visualization`
===============================

.. py:module:: odisseo.visualization

.. autodoc2-docstring:: odisseo.visualization
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`energy_angular_momentum_plot <odisseo.visualization.energy_angular_momentum_plot>`
     - .. autodoc2-docstring:: odisseo.visualization.energy_angular_momentum_plot
          :summary:
   * - :py:obj:`plot_last_snapshot <odisseo.visualization.plot_last_snapshot>`
     - .. autodoc2-docstring:: odisseo.visualization.plot_last_snapshot
          :summary:
   * - :py:obj:`plot_orbit <odisseo.visualization.plot_orbit>`
     - .. autodoc2-docstring:: odisseo.visualization.plot_orbit
          :summary:
   * - :py:obj:`plot_sky_projection <odisseo.visualization.plot_sky_projection>`
     - .. autodoc2-docstring:: odisseo.visualization.plot_sky_projection
          :summary:
   * - :py:obj:`create_3d_gif <odisseo.visualization.create_3d_gif>`
     - .. autodoc2-docstring:: odisseo.visualization.create_3d_gif
          :summary:
   * - :py:obj:`create_3d_gif_velocitycoding <odisseo.visualization.create_3d_gif_velocitycoding>`
     - .. autodoc2-docstring:: odisseo.visualization.create_3d_gif_velocitycoding
          :summary:
   * - :py:obj:`create_projection_gif <odisseo.visualization.create_projection_gif>`
     - .. autodoc2-docstring:: odisseo.visualization.create_projection_gif
          :summary:

API
~~~

.. py:function:: energy_angular_momentum_plot(snapshots, code_units, filename=None)
   :canonical: odisseo.visualization.energy_angular_momentum_plot

   .. autodoc2-docstring:: odisseo.visualization.energy_angular_momentum_plot

.. py:function:: plot_last_snapshot(snapshots, code_units, rp, plotting_units_length, filename=None)
   :canonical: odisseo.visualization.plot_last_snapshot

   .. autodoc2-docstring:: odisseo.visualization.plot_last_snapshot

.. py:function:: plot_orbit(snapshots, ax_lim, code_units, plotting_units_length, config, filename=None)
   :canonical: odisseo.visualization.plot_orbit

   .. autodoc2-docstring:: odisseo.visualization.plot_orbit

.. py:function:: plot_sky_projection(snapshots, code_units, plotting_units_length, filename=None)
   :canonical: odisseo.visualization.plot_sky_projection

   .. autodoc2-docstring:: odisseo.visualization.plot_sky_projection

.. py:function:: create_3d_gif(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, filename=None)
   :canonical: odisseo.visualization.create_3d_gif

   .. autodoc2-docstring:: odisseo.visualization.create_3d_gif

.. py:function:: create_3d_gif_velocitycoding(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, vmin=None, vmax=None, filename=None)
   :canonical: odisseo.visualization.create_3d_gif_velocitycoding

   .. autodoc2-docstring:: odisseo.visualization.create_3d_gif_velocitycoding

.. py:function:: create_projection_gif(snapshots, ax_lim, code_units, plotting_units_length, plot_units_time, filename=None)
   :canonical: odisseo.visualization.create_projection_gif

   .. autodoc2-docstring:: odisseo.visualization.create_projection_gif
