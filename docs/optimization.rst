.. _lab:optimization:

Run the optimization
====================

The deterministic optimization procedure optimizes multiple objectives by controlling the design variables. 
The robust optimization procedure optimizes the mean and minimizes the standard deviation of quantities of interest by controlling the design variables. 
The multi-objective optimization algorithm is Nondominating Sorting Genetic Algorithm (NSGA-II). More information on NSGA-II is available in :ref:`lab:ssnsga2`.
The design variables and model parameters are characterized in the :file:`design_space` folder.
For robust optimization, the uncertainty on the respective design variables and model parameters is characterized in the :file:`stochastic_space` folder.
More information on characterizing these files is available in :ref:`lab:ssdesignspace` and :ref:`lab:ssstochastic_space`, respectively. 
The system model evaluations are coupled with the optimization algorithm in :py:mod:`case_description.`.
More information on this Python wrapper is discussed in :ref:`lab:wrapper`. 
 

.. _lab:ssrundetopt:

run deterministic optimization
------------------------------

To run a deterministic optimization, first the optimization module should be imported::

    import rheia.OPT.optimization as rheia_opt

To characterize a deterministic optimization, the following dictionary should be completed::

    dict_opt = {'case':                case_name,
                'objectives':          {opt_type: weights}, 
                'population size':     n_pop,
                'stop':                ('BUDGET', comp_budget),
                'results dir':         directory,

                'x0':                  (pop_type, pop_method), #optional, default is ('AUTO', 'LHS')
                'cx prob':             c_prob,                 #optional, default is 0.9
                'mut prob':            mut_prob,               #optional, default is 0.1
                'eta':                 eta,                    #optional, default is 0.2
                'n jobs':              n_jobs,                 #optional, default is 1 
                'print results light': [light_bool, gen_step], #optional, default is [False]
                }

This dictionary is used as the argument for the :py:func:`run_opt()` function, 
located in the :py:mod:`optimization` module, which starts the optimization procedure::

    rheia_opt.run_opt(dict_opt)

The dictionary includes necessary items and optional items. The items are clarified in the following sections.

Necessary items
^^^^^^^^^^^^^^^

In the following subsections, the necessary items are described.
If one of these items is not provided, the code will return an error.

'case': case_name
~~~~~~~~~~~~~~~~~

The string `case_name` corresponds to the name of the case. 
This name should be equal to the name of the folder that comprises the case, which situates in the folder that contains the cases :file:`CASES`. 
To illustrate, if the optimization case is defined in :file:`CASES\\CASE_1`, 
the dictionary includes the following item::

		'case': 'CASE_1'

'objectives': {opt_type: (weights)} 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the item with `objectives` key, the optimization type and the weigths for the objectives are specified. 
Two optimization types are available: deterministic optimization ('DET') and robust optimization ('ROB').
The weights are defined in a tuple and determine if the objective is either maximized or minimized.
When minimization of an objective is desired, the weigth corresponds to -1. 
Instead, when maximization is desired, the weight corresponds to 1. 
For deterministic optimization ('DET'), the order of the weights corresponds to the order of the model outputs
returned by the method :py:meth:`evaluate()` (see :ref:`lab:wrapper`).  
For instance, for 2 objectives which should be minimized simultaneously in a deterministic optimization, the dictionary item reads::

	'objectives': {'DET': (-1, -1)}

Alternatively, maximizing the first objective and minimizing the second and third objective corresponds to::

	'objectives': {'DET': (1, -1, -1)}
	
In the robust optimization approach, the mean and standard deviation for each quantity of interest is optimized.
For each quantity of interest, the weight for the mean and standard deviation should be provided.
Hence, the weights with even index correspond to the mean, while the weigths with odd index correspond to the standard deviation.
To illustrate, when the mean should be maximized and the standard deviation minimized for two quantities of interest, the dictionary item reads::

	'objectives': {'ROB': (1, -1, 1, -1)}

Instead, when only one quantity of interest is desired, for which both the mean and standard deviation should be minimized, the item reads::

	'objectives': {'ROB': (-1, -1)}
	
Note that for robust optimization, the number of wheights should be equal to two times the number of quantities of interest (i.e. the mean and standard deviation for each
quantity of interest is an objective). Therefore, make sure that the number of quantities of interest defined (see :ref:`lab:secobjofint`) matches the number of weights defined.

'population size': n_pop
~~~~~~~~~~~~~~~~~~~~~~~~~~

The population size corresponds to the number of samples contained in a single population. 
After each evaluation of the entire population, the optimizer generates a new population with an equal number of samples.
This iterative process continues until the predefined computational budget is complied with. 
Hence, with a computational budget of 1440 model evaluations, 
a population size of 20 will lead to at least 72 generations for deterministic optimization::

	'population size': 20
	
Note that when the population number and computational budget do not result in an integer for the number of generations, 
the number of generations is rounded up to the nearest integer.  
Additional details on defining the value for the population size is illustrated in :ref:`lab:choosepop`. 

'stop': ('BUDGET', comp_budget)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stopping criterium for the optimization is defined by the computational budget, i.e. the number of model evaluations. 
This is a common engineering stopping criterium, which is defined based on the time available
to perform the optimization. To illustrate, when the system model takes 10 seconds to evaluate and 4 cores are available for parallel processing, 
the computational budget for a deterministic optimization procedure of 1 hour is equal to 1440.
The allocation of this computational budget through the integer `comp_budget` is illustrated below::

	'stop': ('BUDGET', 1440)

'results dir': directory
~~~~~~~~~~~~~~~~~~~~~~~~

The result directory corresponds to the folder where the results are stored. 
For an illustrative case `CASE_1`, the results are stored in the folder :file:`RESULTS\\CASE_1\\DET\\results_1` by initiating the following key-value pair in the dictionary::

'results dir': 'results_1'

If previous results exist in this directory, the optimization procedure continues from the last, previously generated, population. 
Then, the characterization of the initial population in :ref:`lab:ssx0` is ignored, but the computational budget is renewed. 

.. _lab:optitemsdet:

Optional items
^^^^^^^^^^^^^^

In addition to the necessary items, optional items can be added to the dictionary. 
If one of these items is not provided in the dictionary, a typical value will be assigned to the key. 
The default configuration for these optional items is::

                'x0':                  ('AUTO', 'LHS'), 
                'cx prob':             0.9,
                'mut prob':            0.1,
                'eta':                 0.2,
                'n jobs':              1, 
                'print results light': [False],

.. _lab:ssx0:

'x0': (pop_type, pop_method) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Information can be provided to characterize the starting population. If no information is available on the starting population, 
the population can be generated automatically by defining the string `pop_type` with 'AUTO'. 
When 'AUTO' is selected, there are two ways of generating the population automatically: 
randomly (`pop_method` = 'RANDOM') or through Latin Hypercube Sampling (`pop_method` = 'LHS'). 
The default configuration for this item is thegeneration of the first population through LHS::

	'x0': ('AUTO', 'LHS')

Alternatively, when information on the starting population is available, the `pop_type` should be defined by 'CUSTOM'. 
In that case, the starting population should be provided in a separate file,
located in the case folder. The name of the file corresponds to the string that defines `pop_method`. 
To illustrate for 'CASE_1', with a starting population saved in :file:`CASES\\CASE_1\\x0_start`, the item is defined as::

	'x0': ('CUSTOM', 'x0_start')

This extensionless file should contain a number of samples equal to the population size. 
Each sample is characterized by a number of values equal to the number of design variables, delimited by a white space.
Each value should situate between the lower bound and upper bound of the corresponding design variable, 
in the order of appearance of the design variables in the :file:`design_space` file.

Example: 

The following design variables are defined in :file:`design_space`::

	var_1 var 1 3
	var_2 var 0.4 0.9
	var_3 var 12 15

Then, for a population size of 5, a suitable characterization of the starting population file is::

	1.43 0.78 13.9
	2.97 0.44 12.1
	1.12 0.64 14.2
	2.31 0.51 14.5
	2.05 0.88 13.6

'cx prob': c_prob
~~~~~~~~~~~~~~~~~

The probability of crossover at the mating of two parent samples.
The default crossover probability is equal to 0.9::

	'cx prob': 0.9
	
More information on setting the crossover probability is illustrated in :ref:`lab:choosepop`. 

'mut prob': mut_prob
~~~~~~~~~~~~~~~~~~~~

The probability of mutation, i.e. the probability of values in the design samples being flipped.
The default value on the mutation probability corresponds to::

	'mut prob': 0.1

More information on setting the mutation probability is illustrated in :ref:`lab:choosepop`. 

'eta': eta
~~~~~~~~~~

The crowding degree of the crossover, which determines the resemblance of the children to their parents. 
The default crowding degree is::

    'eta': 0.2

'n jobs': n_jobs
~~~~~~~~~~~~~~~~

The number of parallel processes can be defined by the number of available cores on the Central Processing Unit. 
The default value corresponds to linear processing::

	'n jobs': 1
	
Alternatively, the number of parallel processes can be retreived through the `cpu_count` function from the multiprocessing package.
After importing multiprocessing, the item can be defined by::

    'n jobs': int(multiprocessing.cpu_count()/2)

.. _lab:detprintreslight:

'print results light': [light_bool, gen_step]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For every optimization case, three different files are generated and continuously appended during the optimization:
A :file:`STATUS` file which saves the generation number and the computational budget spent after each generation;
a file with the population for each generation; a file with the fitness values for each population.
When the computational budget is large and a significant number of design variables are present in the optimization problem,
these three result files can become large, i.e. several MB. Therefore, the framework provides the option to avoid saving each generation.
By setting the light_bool to True and providing the step size `gen_step` for each saved generation, the files size can be significantly reduced.
To illustrate, to save only generation 4, 8, 12, 16 and 20 in a case for which 20 generations are evaluated, 
the following item can be provided in the dictionary::

'print results light': [True, 4]

The default configuration stores each generation::

'print results light': [False]

deterministic optimization dictionary examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When combining the examples in the previous section, a configurated optimization dictionary with the necessary items looks as follows::

    In [1]: import rheia.OPT.optimization as rheia_opt

    In [3]: dict_opt = {'case':                'CASE_1',
       ...:             'objectives':          {'DET': (-1,-1)}, 
       ...:             'population size':     20,
       ...:             'stop':                ('BUDGET', 1440),
       ...:             'results dir':         'results_1',
       ...:             }

    In [4]: rheia_opt.run_opt(dict_opt)

In the example below, parallel processing is considered, the optimization starts from a predefined population, defined in `x0_start`, 
and the crossover probability is decreased to 0.85::

    In [1]: import rheia.OPT.optimization as rheia_opt
    In [2]: import multiprocessing as mp

    In [3]: dict_opt = {'case':                'CASE_1',
       ...:             'objectives':          {'DET': (-1,-1)}, 
       ...:             'population size':     20,
       ...:             'stop':                ('BUDGET', 1440),
       ...:             'results dir':         'results_1',
       ...:             'x0':                  ('CUSTOM', 'x0_start'), 
       ...:             'cx prob':             0.85,
       ...:             'n jobs':              int(mp.cpu_count()/2),
       ...:             }

    In [4]: rheia_opt.run_opt(dict_opt)


Run robust optimization
-----------------------

Like for deterministic optimization, first the optimization module should be imported::

    import rheia.OPT.optimization as rheia_opt

To characterize the robust optimization, the following dictionary with parameters related to the case, optimization 
and uncertainty quantification should be completed::

    dict_opt = {'case':                  case_name,
                'objectives':            {opt_type: weights}, 
                'population size':       n_pop,
                'stop':                  ('BUDGET', comp_budget),
                'results dir':           directory,
                'pol order':             pol_order,
                'objective names':       obj_names,
                'objective of interest': obj_of_interest,

                'x0':                  (pop_type, pop_method), #optional, default is ('AUTO', 'LHS')
                'cx prob':             c_prob,                 #optional, default is 0.9
                'mut prob':            mut_prob,               #optional, default is 0.1
                'eta':                 eta,                    #optional, default is 0.2
                'n jobs':              n_jobs,                 #optional, default is 1 
                'print results light': [light_bool, gen_step], #optional, default is [False]
                'sampling method':       sampling_method       #optional, default is 'SOBOL'
                }

This dictionary is used as the argument for the `run_opt()` function, which starts the optimization procedure::

    rheia_opt.run_opt(dict_opt)

The necessary and optional keys related to the optimization are described in :ref:`lab:ssrundetopt`.
The additional necessary and optional items related to the uncertainty quantification are described in the following subsections. 

Necessary items
^^^^^^^^^^^^^^^

In the following subsections, the necessary items are described.
If one of these items is not provided, the code will return an error.


'pol order': pol_order
~~~~~~~~~~~~~~~~~~~~~~

The polynomial order corresponds to the maximum polynomial degree in the PCE trunctation scheme.
The polynomial order is characterized by an integer, e.g. for a polynomial order of 2::

	'pol order': 2
	
Determining the appropriate polynomial order is case-specific. A method to determine the order is presented in the next section :ref:`lab:detpolorder`.

'objective names': [obj_names]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model might return several outputs (i.e. for multi-objective optimization).
The names of the different model outputs can be provided in the list `objective_names`. 
These names are chosen freely by the user, formatted in a string.
If the model returns 3 outputs, the list can be constructed as::

	'objective names': ['output_1', 'output_2', 'output_3']
 

.. _lab:secobjofint:
'objective of interest': obj_of_interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite that several outputs can be returned for each model evaluation, not all outputs might be of interest for the robust optimization.
The quantities of interest should be provided in the list `obj_of_interest`. These names should be present in the list of all the objective names.
To illustrate, for a robust optimization with the mean and standard deviation of 'output_2' and 'output_3' as objectives, 
the item in the dictionary is configurated as::

	'objective of interest': ['output_2','output_3']

Instead, if a robust optimization is desired with 'output_3' as quantity of interest::

	'objective of interest': ['output_3']

Optional items
^^^^^^^^^^^^^^

When running robust optimization, only one additional optional item exists, in addition to the 
optional items presented in the deterministic optimization section (:ref:`lab:optitemsdet`).
The item is described below.

'sampling method': sampling_method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the construction of a PCE, a number of model evaluation are required. These samples can be generated
in two different ways: randomly, or through a Sobol' sequence. 
The random generation is called through the string 'RANDOM', while the Sobol' sequence is initiated through 'SOBOL'.
The default configuration for generating the samples for PCE is through a Sobol' sequence::

	'sampling method': 'SOBOL'

Robust optimization dictionary example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When combining the examples in the previous section, a configurated optimization dictionary with only necessary items for robust optimization looks as follows::

    In [1]: import rheia.OPT.optimization as rheia_opt

    In [3]: dict_opt = {'case':                  'CASE_1',
       ...:             'objectives':            {'ROB': (-1,-1,-1,-1)}, 
       ...:             'population size':       20,
       ...:             'stop':                  ('BUDGET', 1440),
       ...:             'results dir':           'results_1',
       ...:             'pol order':             2,
       ...:             'objective names':       ['output_1', 'output_2', 'output_3'],
       ...:             'objective of interest': ['output_2','output_3'],
       ...:             }

    In [4]: rheia_opt.run_opt(dict_opt)

An additional example, where parallel processing is considered, the mutation probability is decreased to 0.05 and the sampling method is random::

    In [1]: import rheia.OPT.optimization as rheia_opt
    In [2]: import multiprocessing as mp

    In [3]: dict_opt = {'case':                  'CASE_1',
       ...:             'objectives':            {'ROB': (-1,-1,-1,-1)}, 
       ...:             'population size':       20,
       ...:             'stop':                  ('BUDGET', 1440),
       ...:             'results dir':           'results_1',
       ...:             'pol order':             2,
       ...:             'objective names':       ['output_1', 'output_2', 'output_3'],
       ...:             'objective of interest': ['output_2','output_3'],
       ...:             'mut prob':              0.05,
       ...:             'sampling method':       'RANDOM',
       ...:             'n jobs':                int(mp.cpu_count()/2), 
       ...:             }

    In [4]: rheia_opt.run_opt(dict_opt)


The post-processing of the results is described in :ref:`lab:optimizationresults`.

