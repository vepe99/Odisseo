:py:mod:`odisseo.time_integration`
==================================

.. py:module:: odisseo.time_integration

.. autodoc2-docstring:: odisseo.time_integration
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SnapshotData <odisseo.time_integration.SnapshotData>`
     - .. autodoc2-docstring:: odisseo.time_integration.SnapshotData
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`time_integration <odisseo.time_integration.time_integration>`
     - .. autodoc2-docstring:: odisseo.time_integration.time_integration
          :summary:

API
~~~

.. py:class:: SnapshotData
   :canonical: odisseo.time_integration.SnapshotData

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: odisseo.time_integration.SnapshotData

   .. py:attribute:: times
      :canonical: odisseo.time_integration.SnapshotData.times
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.times

   .. py:attribute:: states
      :canonical: odisseo.time_integration.SnapshotData.states
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.states

   .. py:attribute:: total_energy
      :canonical: odisseo.time_integration.SnapshotData.total_energy
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.total_energy

   .. py:attribute:: angular_momentum
      :canonical: odisseo.time_integration.SnapshotData.angular_momentum
      :type: jax.numpy.ndarray
      :value: None

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.angular_momentum

   .. py:attribute:: runtime
      :canonical: odisseo.time_integration.SnapshotData.runtime
      :type: float
      :value: 0.0

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.runtime

   .. py:attribute:: num_iterations
      :canonical: odisseo.time_integration.SnapshotData.num_iterations
      :type: int
      :value: 0

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.num_iterations

   .. py:attribute:: current_checkpoint
      :canonical: odisseo.time_integration.SnapshotData.current_checkpoint
      :type: int
      :value: 0

      .. autodoc2-docstring:: odisseo.time_integration.SnapshotData.current_checkpoint

.. py:function:: time_integration(primitive_state, mass, config: odisseo.option_classes.SimulationConfig, params: odisseo.option_classes.SimulationParams)
   :canonical: odisseo.time_integration.time_integration

   .. autodoc2-docstring:: odisseo.time_integration.time_integration
