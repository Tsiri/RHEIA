.. _lab:postprocessing:

post-processing
===============

The results from deterministic optimization, robust optimization and uncertainty quantification are stored in the :file:`RESULTS` folder.
The post-processing of the results from the specific case can be initiaded by first instantiating an object from the :py:class:`post-process` class::

    In [1]: import rheia.POST_PROCESS.lib_post_process as rheia_pp
	
    In [2]: case = 'CASE_1'

    In [3]: my_post_process = rheia_pp.post_process(case)
	
Two different subclasses are available for either optimization or uncertainty quantification. Both approaches are described in the following sections.

.. _lab:optimizationresults:

optimization results
--------------------

An illustrative path directs towards the result files from optimization, 
for which the path depends on the case name (e.g. `CASE_1`), the analysis type (DET, ROB)
and the results directory (e.g. `results_1`): :file:`\\RESULTS\\CASE_1\\DET\\results_1`.
In this folder, 3 folder are present: :file:`STATUS`, :file:`fitness` and :file:`population` (or :file:`fitness_light` and :file:`population_light`, see :ref:`lab:detprintreslight`).
The :file:`STATUS` file consists of two columns: ITER and EVALS. In ITER, the finished generation number is saved, while the corresponding number in EVALS
provides the actual computational budget spent after completing that generation.
The :file:`population` and :file:`fitness` file contain the design samples and results, respectively. 
This information is stored for every design sample in every generation 
(or in every :math:`i^\mathrm{th}` generation in case of light printing). The design sample on line :math:`j` in :file:`population` corresponds to the fitness 
on line :math:`j` in :file:`fitness`.
Plotting the results can be performed by first instantiating an object from :py:class:`post_process_opt` with the arguments related to 
the analysis type (DET or ROB) and a boolean that determines if light printing was considered::

    In [4]: eval_type = 'DET'

    In [6]: LIGHT = False

    In [7]: my_opt_plot = rheia_pp.post_process_opt(my_post_process,LIGHT,eval_type)

Then, the method :py:meth:`get_fitness_population()` can be called::

    In [8]: result_dir = 'result_1'

    In [8]: y,x = my_opt_plot.get_fitness_population(result_dir)
 
The function returns, for the last available generation, the objectives and the population.
Additionally, the design samples and fitness values are sorted based on the first objective and saved in :file:`population_final_sorted` 
and :file:`fitness_final_sorted`, respectively, in the results directory.
For instance, the first two objectives can be plotted with respect to eachother as follows::

    In [7]: import matplotlib.pyplot as plt

    In [8]: plt.plot(y[0],y[1])

    In [8]: plt.show()

Another example: the third input design variable can be plotted in function of the first objective::

    In [8]: plt.plot(y[0],x[2])

    In [8]: plt.show()

Alternatively, a number of generations can be plotted on the same graph. 
This enables to evaluate the convergence of the result. To illustrate, plotting 
generation 5, 15 and 25 is done as follows::

	
    In [8]: for i in [5,15,25]:
       ...:     y,x = my_opt_plot.get_fitness_population(result_dir, gen = i)
       ...:     plt.plot(y[0],y[1])
       ...: plt.show()

.. _lab:uqresults:

uncertainty quantification results
----------------------------------

The results path depends on the case name (e.g. `CASE_1`), the analysis type (UQ)
and the results directory (e.g. `results_1`), i.e. :file:`\\RESULTS\\CASE_1\\UQ\\results_1`.
In this folder, at least 1 folder is present: the :file:`samples`  file. This file includes the samples 
and the corresponding deterministic model response, when a system model is connected to the framework (i.e. 'create only samples' set to False).
The second file and third file are named based on the selected maximum polynomial degree and the quantity of interest 
(e.g. :file:`full_pce_order_2_output_2` and :file:`full_pce_order_2_output_2_Sobol_indices`).
These files respectively include the PCE information (LOO error, mean and standard deviation) and the Sobol indices (first order and total order).

To post-process the UQ results, first the object from the :py:class:`post_process_uq` class is instantiated. 
The object is characterized by the arguments on the maximum polynomial degree considered::

    In [10]: pol_order = 1

    In [13]: my_post_process_uq = rheia_pp.post_process_uq(my_post_process, pol_order)

Once the object is instantiated, the Sobol' indices can be retreived through the :py:meth:`get_sobol` method,
for which the result directory and name of the quantity of interest are passed as arguments::

    In [11]: result_dir = 'results_1'

    In [12]: QoI = 'output_2'

    In [14]: names, sobol = my_post_process_uq.get_sobol(result_dir,QoI)

To illustrate, the Sobol' indices can then be plotted in a bar chart::

    In [14]: plt.bar(names, sobol)

    In [14]: plt.show()

Alternatively, the LOO-error can be extracted::

    In [14]: loo = my_post_process_uq.get_LOO(result_dir,QoI)
	
If the data for the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) was generated, both functions can be plotted as follows::

    In [15]: x,y = my_post_process_uq.get_pdf(result_dir,QoI)

    In [16]: x,y = my_post_process_uq.get_pdf(result_dir,QoI)
 
When UQ was performed on multiple design samples (e.g. 30) to determine the polynomial order (:ref:`lab:detpolorder`), 
the worst-case LOO error and the significant Sobol' indices can be presented through::

    In [8]: result_dirs = ['sample_%i' %i for i in range(30)]

    In [9]: loo = [0]*30

    In [11]: for index,result_dir in enumerate(result_dirs):
       ....:     loo[index] = my_post_process_uq.get_LOO(result_dir,QoI)
       ....: print(max(loo))

    In [12]: my_post_process_uq.get_max_sobol(results_dir,objective,threshold=1./15.)	
	
The threshold argument in get_max_sobol() provides the threshold for which Sobol' indices are considered significant.
