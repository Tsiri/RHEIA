.. _lab:uncertaintyquantification:

Run the uncertainty quantification
==================================

The uncertainty quantification procedure provides the mean and standard deviation of the quantity of interest and the Sobol' indices 
by constructing a Polynomial Chaos Expansion (PCE). More information on PCE is available in :ref:`lab:PCE`.
The mean value and the corresponding uncertainty of the model parameters are characterized in :file:`design_space` and :file:`stochastic_space`, respectively.
More information on characterizing these files is available in :ref:`lab:ssdesignspace` and :ref:`lab:ssstochastic_space`, respectively.  
The model returns the values for the quantity of interest through the function :py:meth:`evaluate()` defined in :py:mod:`case_description`.
More information on this Python wrapper is discussed in :ref:`lab:wrapper`. 


run uncertainty quantification
------------------------------
To run the uncertainty quantification, first the uncertainty quantification module should be imported::

    import rheia.UQ.uncertainty_quantification as rheia_uq

To characterize the uncertainty quantification, the following dictionary with parameters related to the case and uncertainty quantification should be completed::

    dict_uq = {'case':                  case_name,
               'pol order':             pol_order,
               'objective names':       obj_names,
               'objective of interest': obj_of_interest,
               'results dir':           directory      

               'sampling method':       sampling_method,        #optional, default is 'SOBOL'
               'create only samples':   only_samples_bool,      #optional, default is False
               'draw pdf cdf':          [draw_bool, n_samples], #optional, default is [False]
               'n jobs':                n_jobs,                 #optional, default is 1
              }  

The items of the uncertainty quantification dictionary are described in the following subsections. This dictionary is used as the argument for the :py:func`run_uq()` function, 
which initiates the uncertainty quantification procedure::

    rheia_uq.run_uq(dict_uq)

Necessary items
^^^^^^^^^^^^^^^

In the following subsections, the necessary items are described.
If one of these items is not provided, the code will return an error.

'case': case_name
~~~~~~~~~~~~~~~~~

The string `case_name` corresponds to the name of the case. 
This name should be equal to the name of the folder that comprises the case, which situates in the folder that contains the cases `CASES`. 
To illustrate, if the case is defined in :file:`CASES\\CASE_1`, 
the dictionary includes the following item::

		'case': 'CASE_1'


'pol order': pol_order
~~~~~~~~~~~~~~~~~~~~~~

The polynomial order corresponds to the maximum polynomial degree in the PCE.
The polynomial order is characterized by an integer, e.g. for a polynomial order of 2::

	'pol order': 2

'objective names': obj_names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PCE is constructed for only 1 quantity of interest. However, the statistical moments for several model outputs can be of interest.
To avoid that for each model output, a new set of model evaluations need to be performed, different model outputs are stored for model evaluation.
The names of the different model outputs can be provided in the list `objective_names`. These names are chosen freely by the user, but should be formatted in a string.
If the model returns 3 outputs, the list can be constructed as::

	'objective names': ['output_1', 'output_2', 'output_3']
 
'objective of interest': obj_of_interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite that several outputs can be returned for each model evaluation, only one output can be selected as a quantity of interest for the PCE.
The name of this quantity of interest `obj_of_interest` should be provided. This name should be present in the list of all the objective names.
To illustrate, if the quantity of interest is 'output_2', out of the list ['output_1', 'output_2', 'output_3'], then the item in the dictionary is configurated as::

	'objective of interest': 'output_2'

'results dir': directory
~~~~~~~~~~~~~~~~~~~~~~~~

The results directory corresponds to the folder where the results are stored. 
For an illustrative case `CASE_1`, the UQ results are saved in the folder :file:`RESULTS\\CASE_1\\UQ\\results_1` 
by initiating the following key-value pair in the dictionary::

'results dir': results_1

Optional items
^^^^^^^^^^^^^^

The following items are optional items. If one of these items is not provided in the dictionary, 
a default value will be assigned to the key. If none of these are provided, the optional dictionary
items are defined as follows::

               'sampling method':       'SOBOL',
               'create only samples':   False,
               'draw pdf cdf':          [False],
               'n jobs':                1,

'sampling method': sampling_method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the construction of a PCE, a number of model evaluation are required. The samples for model evaluation can be generated
in two different ways: randomly, or through a Sobol' sequence. 
The random generation is called through the string 'RANDOM', while the Sobol' sequence is initiated through 'SOBOL'.
The default configuration for generating the samples for PCE is through a Sobol' sequence::

	'sampling method': 'SOBOL'
 
'create only samples': only_samples_bool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, the coupling of the system model with the framework is complex. To avoid this coupling, the samples required to determine the statistical moments
can be generated and then evaluated manually in the system model. Hence, in this first step, the framework should only generate the samples. To do so,
the Bool `only_samples_bool` can be set to True::

	'create only samples': True

However, the default configuration sets the value of 'create only samples' to False::

	'create only samples': False

Additional information on how to create just the samples is present in :ref:`lab:sscreateonlysamples`.

'draw pdf cdf': [draw_bool, n_samples]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the statistical moments, the data for generation the probability density function (pdf) and cumulative distribution function (cdf) can be generated.
This information can be generated by setting the `draw_bool` to True and providing the number of samples evaluated on the PCE `n_samples`.
To illustrate, to generate pdf and cdf datapoints based on a PCE Monte Carlo evaluation with 100,000 samples::

    'draw pdf cdf': [True, 1000000]

In the default configuration, the pdf and cdf are not generated::

    'draw pdf cdf': [False]

'n jobs': n_jobs
~~~~~~~~~~~~~~~~

The number of parallel processes can be defined by the number of available cores on the CPU. 
The default value corresponds to linear processing::

	'n jobs': 1
	
Alternatively, the number of parallel processes can be retreived through the `cpu_count` function from the multiprocessing package.
After importing multiprocessing, the item can be defined by::

    'n jobs': int(multiprocessing.cpu_count()/2)

uncertainty quantification dictionary example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When combining the examples in the previous section, a configurated uncertainty quantification dictionary with the necessary items looks as follows::

    In [1]: import rheia.UQ.uncertainty_quantification as rheia_uq

    In [3]: dict_uq = {'case': 'CASE_1',
       ...:            'pol order': 2,
       ...:            'objective names': ['output_1', 'output_2', 'output_3'],
       ...:            'objective of interest': 'output_2',
       ...:            'results dir': 'results_1'      
       ...:            }  

    In [4]: rheia_uq.run_uq(dict_uq)

Alternatively, a uncertainty quantification dictionary which considers random sampling and generates 100,000 PDF and CDF samples on the PCE surrogate::
 
    In [1]: import rheia.UQ.uncertainty_quantification as rheia_uq

    In [3]: dict_uq = {'case': 'CASE_1',
       ...:            'pol order': 2,
       ...:            'objective names': ['output_1', 'output_2', 'output_3'],
       ...:            'objective of interest': 'output_2',
       ...:            'results dir': 'results_1'      
       ...:            'sampling method': 'RANDOM',
       ...:            'draw pdf cdf': [True, 1000000],                
       ...:            }  

    In [4]: rheia_uq.run_uq(dict_uq)

The post-processing of the results is described in :ref:`lab:uqresults`.
	
.. _lab:sscreateonlysamples:

Create samples for unconnected model
------------------------------------

When it is burdensome to connect the system model to the framework, the framework provides the option to just generate the random samples for uncertainty quantification,
based on the stochastic space defined in :file:`design_space` and :file:`stochastic_space`. These samples can then be evaluated in the model externally.
To generate the samples, use (or make a copy of) the :file:`NO_MODEL` folder in :file:`CASES`.
In this folder, a :py:mod:`case_description` module is present, as well as :file:`design_space` and :file:`stochastic_space`.
The :py:mod:`case_description` module simply contains the instantiation of the class, as no model evaluations are required.
In :file:`design_space` and :file:`stochastic_space`, the stochastic design space is defined. The samples can be generated, required to perform PCE::

    dict_uq = {'case': 'NO_MODEL',
               'pol order': 2,
               'objective names': ['output_1', 'output_2', 'output_3'],
               'objective of interest': 'output_2',
               'results dir': 'results_1',      
               'create only samples': True,                
              }  

For this example, the samples are written in :file:`RESULTS\\NO_MODEL\\UQ\\results_1\\samples`. Once these samples are evaluated in the model on an external location,
the results can be added to the :file:`RESULTS\\NO_MODEL\\UQ\\results_1\\samples` file. When the results are added for 'output_1', 'output_2', 'output_3', 
the PCE can be constructed for the three quantities of interest. In that case, the value for 'create only samples' is set back to False (i.e. the default value).
To illustrate, for a PCE on 'output_2'::

    dict_uq = {'case': 'NO_MODEL',
               'pol order': 2,
               'objective names': ['output_1', 'output_2', 'output_3'],
               'objective of interest': 'output_2',
               'results dir': 'results_1',      
              }  

Make sure that the result directory is equal to the result directory where the updated :file:`samples` file is saved.

.. _lab:detpolorder:

Determine the polynomial order
------------------------------

The maximum polynomial degree for the multivariate polynomials needs to be determined up front and its value should ensure accurate
statistical moments on the quantity of interest in the considered stochastic space. An indication on the accuracy of the PCE is
the Leave-One-Out (LOO) error. If the error is below a certain threshold, the PCE achieves an acceptable accuracy. This threshold is a user-defined constant. 
To ensure accurate statistical moments during the robust optimization procedure, the polynomial order should be sufficient 
over the entire design space. In other words, for each design sample, the polynomial order should be sufficient to construct an accuracte PCE.
Latin Hypercube Sampling is used to construct a set of design samples, which provides a representation of the design space. If the worst-case LOO 
for the corresponding PCEs is still below a certain threshold, the corresponding polynomial order can be considered sufficient to be used during
the robust optimization procedure.

After providing the name of the case, a dictionary with the design variable names, lower bounds and upper bounds can be defined
via the :py:func:`get_design_variables` function::

    In [1]: import rheia.UQ.uncertainty_quantification as rheia_uq
    In [2]: import multiprocessing as mp

    In [1]: case = 'case_name'    
    In [3]: var_dict = rheia_uq.get_design_variables(case)
    
From this dictionary, the design samples can be constructed through LHS via :py:func:`set_design_samples`. 
The number of design samples and the dictionary with information on the design variables are provided as arguments::

    In [1]: n_samples = 5
    In [1]: X = set_design_samples(var_dict, n_samples)

Then, for each design sample in the array `X`, a :file:`design_space` file is constructed through the function :py:func:`write_design_space()`. 
For each :file:`design_space` file, the PCE can be constructed through the characterization of the uncertainty quantification dictionary. 
For more information on the characterization of this dictionary, we refer to :ref:`lab:uncertaintyquantification`.
The uncertainty quantification dictionary and the specific :file:`design_space` file is then provided to the :py:func:`run_uq` function.
In a for loop with iterations equal to the number of design samples, the PCEs are constructed::


	In [5]: for iteration,x in enumerate(X):
	  ....:     rheia_uq.write_design_space(case, iteration, var_dict, x)
	  ....:     dict_uq = {'case': case,
	  ....:                'pol order': 1,
	  ....:                'objective names': ['obj_1','obj_2'],
	  ....:                'objective of interest': 'obj_1',
	  ....:                'results dir': 'res_%i' %iteration      
	  ....:               }   
	  ....:     rheia_uq.run_uq(dict_uq, design_space = 'design_space_%i' %iteration)
		
This results in a PCE for each design sample, with a corresponding LOO error. That LOO error is stored in the :file:`RESULTS` folder.
Considering the specific dictionary determined above, the results for the different design samples are stored in :file:`\\RESULTS\\case\\UQ`::

    RESULTS 
      case
        UQ
          res_0
          res_1
          res_2
          res_3
          res_4
	
Where in each folder, the LOO error is stored in `full_PCE_order_2_obj_1`.

The worst-case LOO error (i.e. the highest LOO error over the diffferent design samples) can be determined through 
the post-processing module :py:mod:`lib_post_process`.
Instantiating an object from the :py:class:`post_process` class is by passing the case name as an argument::

	In [4]: import rheia.POST_PROCESS.lib_post_process as rheia_pp

	In [6]: my_post_process = rheia_pp.post_process(case)

This object is used to instantiate an object from the class :py:class:`post_process_uq`, 
by passing the arguments related to the polynomial order:: 

    In [7]: pol_order = 2

    In [13]: my_post_process_uq = rheia_pp.post_process_uq(my_post_process,pol_order)

Then, the :py:meth:`get_LOO()` method returns the LOO error for every sample::

    In [8]: result_dirs = ['run_%i' %i for i in range(5)]

    In [9]: objective = 'obj_1'

    In [9]: loo = [0]*5

    In [11]: for index,result_dir in enumerate(result_dirs):
       ....:     loo[index] = my_post_process_uq.get_LOO(result_dir,objective))
       ....: print(max(loo))

Based on the worst-case LOO error, the maximum polynomial degree of the PCE for the robust design optimization can be evaluated.


