.. _lab:stochasticdesignspace:

Stochastic design space
=======================

In this section, the characterization of the model parameters and design variables is described.
This characterization is performed through two files: :file:`design_space` and :file:`stochastic_space`.
In :file:`design_space`, the deterministic values for the model parameters and the range for the design variables are provided.
In :file:`stochastic_space`, uncertainty is allocated to the specific model parameters and design variables.
When a deterministic optimization is performed, only the :file:`design_space` file is required. 
In the other cases, i.e. uncertainty quantification and robust optimization, both files are required.

.. _lab:ssdesignspace:

design_space
------------

In this file, you define the model parameters which need a quantification in your model. 
When performing uncertainty quantification, the :file:`design_space` file consists only of :ref:`lab:ssmodelparameters`.
In the case of deterministic optimization or robust optimization, the :file:`design_space` file requires :ref:`lab:ssdesignvariables`. 
Additionally, if some model parameters require a quantification outside the model, the :file:`design_space` file includes both :ref:`lab:ssdesignvariables` and :ref:`lab:ssmodelparameters`.
The file uses whitespace as delimiter between the inputs related to the parameters and variables. 


.. _lab:ssdesignvariables:

design variables
^^^^^^^^^^^^^^^^
 
The design variables are parameters that are controllable by the designer within a certain range. 
This range should be provided, to shape the search space for the optimizer. 
To define a design variable, the set-up in the :file:`design_space` file is as follows::

	name parameter_type lb ub

where:

- name: name of the variable;
- parameter_type: a variable is indicated with `var`;
- lb: lower bound value for the design variable;
- ub: upper bound value for the design variable. 

An example for a configured design variable `n_pv` with a range between 1e-8 and 50 is::

	n_pv var 1e-8 50


.. _lab:ssmodelparameters:

model parameters
^^^^^^^^^^^^^^^^

A model parameter corresponds to a parameter that usually cannot be controlled by the designer, or the decision on this parameter value is fixed. 
Such a parameter can be considered deterministic or uncertain. In this file :file:`design_space`, the deterministic value (or mean value when the parameter is considered uncertain) is provided.
The configuration of a model parameter is similar to the configuration of a design variable::

	name parameter_type mean

where:

- name: name of the variable;
- parameter_type: a parameter is indicated with `par`;
- mean: mean value (or deterministic value if parameter is deterministic).

An example of a configured model parameter `opex_dcdc` with a mean value of 0.03 is::

	opex_dcdc par 0.03

.. _lab:ssexampleds:

example design_space
^^^^^^^^^^^^^^^^^^^^
Conclusively, an example of a configured :file:`design_space` file, which consists of 3 model parameters (par_1, par_2 and par_3) and 2 design variables (design_var_1 and design_var_2), is presented::

	par_1        par 4
	par_2        par 2.5
	par_3        par 175
	design_var_1 var 1 3
	design_var_2 var 1e-8 100

.. _lab:ssstochastic_space:

stochastic_space
----------------

This file is required when performing robust optimization and uncertainty quantification, where several parameters are subjected to uncertainty. 
This uncertainty can be allocated through the file :file:`stochastic_space`. 
For every design variable and model parameter defined in :file:`design_space`, an uncertainty can be defined.
Defining the uncertainty of a parameter can be done through the following syntax::

	name abs_rel distribution deviation

where:

	- name: name of the parameter or variable, equal to the name of the parameter or variable in :file:`design_space`;
	- abs_rel: absolute or relative uncertainty to the mean, defined with `absolute` or `relative`, respectively;
	- distribution: The distribution of the uncertainty;
	- deviation: uncertainty on the mean.

A more detailed description of these parameters is available in :ref:`lab:ssdistributions`

An example of a configured uncertain parameter `par_2`, characterized by a Uniform distribution with a :math:`\pm 1` deviation from the mean value::

	par_2 absolute Uniform 1

Note that it is not required to allocate an uncertainty to every design variable and model parameter defined in :file:`design_space`.
In other words, when a parameter (or variable) is defined in :file:`design_space`, but not in U, the parameter (or variable) is considered deterministic. 
Moreover, the order of appearance of parameters and variables in :file:`design_space` should not be kept in U.

.. _lab:ssdistributions:

uncertainty characterization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following distributions are available:

- Uniform
- Gaussian

The meaning of deviation at the end of the line depends on the distribution. When a Uniform distribution is considered,
the deviation refers to the absolute difference between the upper bound of the Uniform distribution and the mean: for :math:`\mathcal{U}(a,b)`, :math:`deviation = (b-a)/2`).
When a Gaussian distribution is considered, the value corresponds to the standard deviation: :math:`\mathcal{N}(mean,deviation)`.

example stochastic_space
^^^^^^^^^^^^^^^^^^^^^^^^

In summary, a :file:`stochastic_space` file corresponding to the illustrative :file:`design_space` example file in :ref:`lab:ssexampleds` might look like this::

	par_1        relative Gaussian 0.5
	par_2        absolute Uniform  1
	design_var_2 relative Uniform  0.1

Where the model parameter `par_3` and design variable `design_var_1` are considered deterministic, 
`par_1` is characterized by a Gaussian distribution with a 
relative standard deviation of 0.5 (i.e. :math:`\mathcal{N}(4,2)`),    
`par_2` is characterized by a Uniform distribution with an 
absolute deviation of 1 (i.e. :math:`\mathcal{U}(1.5,3.5)`) and    
`design_var_2` is characterized by a Uniform distribution with a 
relative deviation of 0.1. For `design_var_2`, the actual Uniform distribution depends on the mean value selected by the optimizer for each evaluated design.



