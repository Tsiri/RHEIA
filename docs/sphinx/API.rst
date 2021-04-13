

.. _lab:APIref:

API reference
=============



.. currentmodule:: rheia.OPT
   
Optimization
------------

The main function that initiates the optimization procedure.

.. autosummary::
   :toctree: generated/
   
   optimization.run_opt

In this function, the starting samples are created.

.. autosummary::
   :toctree: generated/
   
   optimization.check_existing_results
   optimization.scale_samples_to_design_space
   optimization.write_starting_samples
   optimization.create_starting_samples
      
In addition, the name of the optimization class is loaded.

.. autosummary::
   :toctree: generated/
   
   optimization.parse_available_opt
   optimization.load_optimizer
   genetic_algorithms.return_opt_methods
   genetic_algorithms.return_opt_obj   

Finally, an object from the optimization class is instantiated
and the :py:meth:`run_optimizer` is called.
The NSGA-II optimization class includes methods to perform NSGA-II.

.. autosummary::
   :toctree: generated/
   
   genetic_algorithms.NSGA2
   genetic_algorithms.NSGA2.nsga2_1iter
   genetic_algorithms.NSGA2.run_optimizer

The methods to create and evaluate the samples.

.. autosummary::
   :toctree: generated/

   genetic_algorithms.NSGA2.define_samples_to_eval
   .genetic_algorithms.NSGA2.evaluate_samples
   genetic_algorithms.NSGA2.assign_fitness_to_population
   genetic_algorithms.NSGA2.read_doe
   genetic_algorithms.NSGA2.eval_doe
 
The methods to create and update the result files.

.. autosummary::
   :toctree: generated/

   genetic_algorithms.NSGA2.init_opt
   genetic_algorithms.NSGA2.parse_status
   genetic_algorithms.NSGA2.write_status
   genetic_algorithms.NSGA2.append_points_to_file

.. currentmodule:: rheia.UQ

Uncertainty Quantification
--------------------------

The main function that initiates the uncertainty quantification procedure.

.. autosummary::
   :toctree: generated/

   uncertainty_quantification.run_uq

This function instantiates an object from :py:class:`Data`. This class includes
methods to acquire the characteristics of the stochastic parameters and to create
the file where the samples are stored

.. autosummary::
   :toctree: generated/

   pce.Data
   pce.Data.create_samples_file
   pce.Data.read_stoch_parameters

An object from the class :py:class:`RandomExperiment` is instantiated. This class 
includes a method to determine the number of samples required to construct the PCE.

.. autosummary::
   :toctree: generated/

   pce.RandomExperiment
   pce.RandomExperiment.n_terms

In addition, methods to create the distributions, generate the samples and evaluate 
the samples are present. 

.. autosummary::
   :toctree: generated/

   pce.RandomExperiment.read_previous_samples
   pce.RandomExperiment.create_distributions
   pce.RandomExperiment.create_samples
   pce.RandomExperiment.create_only_samples
   pce.RandomExperiment.evaluate

The PCE class enables to construct a PCE.

.. autosummary::
   :toctree: generated/

   pce.PCE
   pce.PCE.n_to_sum
   pce.PCE.multindices
   pce.PCE.ols
   pce.PCE.calc_a
   pce.PCE.run
	
The statistics, Sobol' indices and Leave-One-Out error 
are extracted out of the PCE in the methods below.

.. autosummary::
   :toctree: generated/

   pce.PCE.get_statistics
   pce.PCE.get_psi_sq
   pce.PCE.calc_sobol
   pce.PCE.calc_loo
	
Finally, the results are printed and stored in corresponding
result files.

.. autosummary::
   :toctree: generated/

   pce.PCE.print_res
   pce.PCE.draw

.. currentmodule:: rheia.POST_PROCESS

Post-processing
---------------

The post-processing of the optimization and uncertainty quantification 
is performed by instantiating a :py:class:`PostProcess` object.

.. autosummary::
   :toctree: generated/
   lib_post_process.PostProcess

The optimization results are extracted with the methods in :py:class:`PostProcessOpt`.

.. autosummary::
   :toctree: generated/
   lib_post_process.PostProcessOpt
   lib_post_process.PostProcessOpt.determine_pop_gen
   lib_post_process.PostProcessOpt.get_fitness_values
   lib_post_process.PostProcessOpt.get_population_values
   lib_post_process.PostProcessOpt.sorted_result_file
   lib_post_process.PostProcessOpt.get_fitness_population

The uncertainty quantification results are extracted with the methods in :py:class:`PostProcessUQ`.

.. autosummary::
   :toctree: generated/
   lib_post_process.PostProcessUQ
   lib_post_process.PostProcessUQ.read_distr_file
   lib_post_process.PostProcessUQ.get_sobol
   lib_post_process.PostProcessUQ.get_pdf
   lib_post_process.PostProcessUQ.get_cdf
   lib_post_process.PostProcessUQ.get_loo
   lib_post_process.PostProcessUQ.get_max_sobol