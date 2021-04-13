.. currentmodule:: rheia


.. _lab:APIref:

API reference
=============



   
Optimization
------------

The main function that initiates the optimization procedure.

.. autosummary::
   :toctree: generated/
   
   rheia.OPT.optimization.run_opt

In this function, the starting samples are created.

.. autosummary::
   :toctree: generated/
   
   rheia.OPT.optimization.check_existing_results
   rheia.OPT.optimization.scale_samples_to_design_space
   rheia.OPT.optimization.write_starting_samples
   rheia.OPT.optimization.create_starting_samples
      
In addition, the name of the optimization class is loaded.

.. autosummary::
   :toctree: generated/
   
   rheia.OPT.optimization.parse_available_opt
   rheia.OPT.optimization.load_optimizer
   rheia.OPT.genetic_algorithms.return_opt_methods
   rheia.OPT.genetic_algorithms.return_opt_obj   

Finally, an object from the optimization class is instantiated
and the :py:meth:`run_optimizer` is called.
The NSGA-II optimization class includes methods to perform the NSGA-II.

.. autosummary::
   :toctree: generated/
   
   rheia.OPT.genetic_algorithms.NSGA2
   rheia.OPT.genetic_algorithms.NSGA2.nsga2_1iter
   rheia.OPT.genetic_algorithms.NSGA2.run_optimizer

The methods to create and evaluate the samples.

.. autosummary::
   :toctree: generated/

   rheia.OPT.genetic_algorithms.NSGA2.define_samples_to_eval
   rheia.OPT.genetic_algorithms.NSGA2.evaluate_samples
   rheia.OPT.genetic_algorithms.NSGA2.assign_fitness_to_population
   rheia.OPT.genetic_algorithms.NSGA2.read_doe
   rheia.OPT.genetic_algorithms.NSGA2.eval_doe
 
The methods to create and update the result files.

.. autosummary::
   :toctree: generated/

   rheia.OPT.genetic_algorithms.NSGA2.init_opt
   rheia.OPT.genetic_algorithms.NSGA2.parse_status
   rheia.OPT.genetic_algorithms.NSGA2.write_status
   rheia.OPT.genetic_algorithms.NSGA2.append_points_to_file

Uncertainty Quantification
--------------------------

The main function that initiates the uncertainty quantification procedure.

.. autosummary::
   :toctree: generated/

   rheia.UQ.uncertainty_quantification.run_uq

This function instantiates an object from :py:class:`Data`. This class includes
methods to acquire the characteristics of the stochastic parameters and to create
the file where the samples are stored

.. autosummary::
   :toctree: generated/

	rheia.UQ.pce.Data
	rheia.UQ.pce.Data.create_samples_file
	rheia.UQ.pce.Data.read_stoch_parameters

An object from the class :py:class:`RandomExperiment` is instantiated. This class 
includes a method to determine the number of samples required to construct the PCE.

.. autosummary::
   :toctree: generated/

	rheia.UQ.pce.RandomExperiment
	rheia.UQ.pce.RandomExperiment.n_terms

In addition, methods to create the distributions, generate the samples and evaluate 
the samples are present. 

.. autosummary::
   :toctree: generated/

	rheia.UQ.pce.RandomExperiment.read_previous_samples
	rheia.UQ.pce.RandomExperiment.create_distributions
	rheia.UQ.pce.RandomExperiment.create_samples
	rheia.UQ.pce.RandomExperiment.create_only_samples
	rheia.UQ.pce.RandomExperiment.evaluate

The PCE class enables to construct a PCE.

.. autosummary::
   :toctree: generated/

	rheia.UQ.pce.PCE
	rheia.UQ.pce.PCE.n_to_sum
	rheia.UQ.pce.PCE.multindices
	rheia.UQ.pce.PCE.ols
	rheia.UQ.pce.PCE.calc_a
	rheia.UQ.pce.PCE.run
	
The statistics, Sobol' indices and Leave-One-Out error 
are extracted out of the PCE in the methods below.

.. autosummary::
   :toctree: generated/

	rheia.UQ.pce.PCE.get_statistics
	rheia.UQ.pce.PCE.get_psi_sq
	rheia.UQ.pce.PCE.calc_sobol
	rheia.UQ.pce.PCE.calc_loo
	
Finally, the results are printed and stored in corresponding
result files.

.. autosummary::
   :toctree: generated/

	rheia.UQ.pce.PCE.print_res
	rheia.UQ.pce.PCE.draw

Post-processing
---------------

The optimization results are extracted with the methods in :py:class:`PostProcessOpt`.

.. autosummary::
   :toctree: generated/
	rheia.POST_PROCESS.lib_post_process.PostProcessOpt
	rheia.POST_PROCESS.lib_post_process.PostProcessOpt.determine_pop_gen
	rheia.POST_PROCESS.lib_post_process.PostProcessOpt.get_fitness_values
	rheia.POST_PROCESS.lib_post_process.PostProcessOpt.get_population_values
	rheia.POST_PROCESS.lib_post_process.PostProcessOpt.sorted_result_file
	rheia.POST_PROCESS.lib_post_process.PostProcessOpt.get_fitness_population

The uncertainty quantification results are extracted with the methods in :py:class:`PostProcessUQ`.

.. autosummary::
   :toctree: generated/
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ.read_distr_file
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ.get_sobol
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ.get_pdf
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ.get_cdf
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ.get_loo
	rheia.POST_PROCESS.lib_post_process.PostProcessUQ.get_max_sobol