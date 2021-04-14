.. figure:: images\logo_4.svg
   :width: 60%
   :align: center

.. toctree::
   :maxdepth: 3
   :numbered:
   :hidden:
   
   functionalities
   connecting your own model
   installation
   tutorial
   auto_examples/index
   stochastic design space
   optimization
   uncertainty quantification
   methods
   energy system models
   API
   contribution
   bibliography

Introduction
============

The Robust optimization of renewable Hydrogen and dErIved energy cArrier systems (RHEIA) framework provides 
multi-objective optimization (deterministic and stochastic) and uncertainty quantification algorithms. 
These algorithms can be applied on hydrogen-based energy systems, which are included in RHEIA.
In addition, RHEIA allows to connect your own models to the algorithms as well.

A brief overview on the features of RHEIA is provided in :ref:`lab:functionalities`, 
followed by a detailed illustration on how to connect your own model (Python based, open source or closed source) in :ref:`lab:connectingyourownmodel`.
If these features comply with your need, the installation procedure and package dependencies are illustrated in :ref:`installationlabel`. 
As a first step, the :ref:`lab:tutorial` provides an initiation of using the framework on a hydrogen-based energy system. 
Additional examples are illustrated in :ref:`lab:examples`. 
The models are characterized by a design space, which defines the design variables, and a stochastic space when parameter uncertainty is considered.
Those spaces are defined in two files, which are elaborated in :ref:`lab:stochasticdesignspace`.
The guides to perform the deterministic optimization, robust optimization and uncertainty quantification are present in :ref:`lab:optimization` and 
:ref:`lab:uncertaintyquantification`, respectively. A first post-processing of the results can be performed following
the steps detailed in :ref:`lab:postprocessing`, such as plotting the fitness values after an optimization or reading the significant Sobol' indices after uncertainty quantification.
The applied methods and energy system models are described in :ref:`lab:methods` and :ref:`lab:energysystemmodels`, respectively. 
The documentation concludes with the :ref:`lab:APIref` and the details on how to contribute to the framework (:ref:`lab:contribution`).


Support
=======

Contact us
rheia.framework@gmail.com

License
=======

The project is licensed under MIT license. 


Indices and tables
==================

:ref:`genindex`
:ref:`modindex`
:ref:`search`
