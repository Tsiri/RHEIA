.. _lab:functionalities:

A short description
===================

RHEIA is a computational framework able to perform multi-objective, deterministic and/or probabilistic design optimization, as well as uncertainty quantification. Following its providing functionalities two main structural parts, method-wise, are entailed: design optimization and uncertainty propagation methods. 
The established Nondominated Sorting Genetic Algorithm, NSGA-II, is used as the deterministic design optimization algorithm. It considers a set of deterministic, model parameters (i.e. parameters which are not controlled by the designer), and provides optimized values based on the minimization (or maximization) of selected model ouputs. And that can work quite well in a fully deterministic setting. However, in most cases deviations from determinism are observerd from a smalle to a larger extent, thus fostering the need to optimize the underlying models considering a probabilistic domain. In this case, the model parameters are considered uncertain (e.g.the uncertainty on the future evolution of the grid electricity price), while the selected model outputs turn to be uncertain as well (e.g. the levelized cost of electricity). Thus, the objectives of such an optimization process are now the minimization (or maximization) of the mean and the minimization of the standard deviation associated to the selected model ouputs.
While optimizing the mean corresponds to optimizing the stochastic performance, minimizing the standard deviation corresponds to minimizing the sensitivity of the quantity of interest to the random environment.
Hence, in this approach, the most robust design can be captured (i.e. the design which achieves the lowest standard deviation on the quantity of interest).
The advantages of this robust design can then be compared with the optimized deterministic designs and designs with an optimized mean on the selected model outputs. 
To materialize this process and propagate the uncertainties of model parameter to outputs, the selected optimizer (i.e. NSGA-II) is coupled to an uncertainty quantification process, in our case Polynomial Chaos Expansion (PCE). In addition to the mean and standard deviation of the selected outputs, the PCE provides the probability density function,
cumulative distribution function and the Sobol' indices on the model outputs as well.  

While the optimization and uncertainty quantification can be performed on any system model, RHEIA focuses hydrogen-based energy system models.
Despite its potential as an energy carrier, and its key role in the future decarbonized economy, hydrogen is scarsely integrated in design optimization studies of hybrid renewable energy systems.
Moreover, when hydrogen is considered, mainly deterministic model parameters (i.e. perfectly known and free from inherent variation) are assumed, despite the uncertainty
that affects the performance of such systems (e.g. market values, climate conditions).
Therefore, RHEIA combines robust design optimization with hydrogen-based energy systems, to unlock robust designs which are least-sensitive to the random environment.
Moreover, the uncertainty quantification enables to rank the effect of model parameters' uncertainties to the outputs for an optimized design, using Sobol's indices.
Thus, it can illustrate the uncertainties that dominate the variance of the outputs of interest.
The hydrogen-based energy system models considered here, encollapse the main hydrogen valorization pathways: power-to-power, power-to-mobility and power-to-fuel.
For each model, a specific set of parameters are defined with uncertainty, which are considered during robust design optimization and uncertainty quantification. 
The techno-economic and environmental uncertainties are adopted from scientific literature.
The models depend on the climate data and/or energy demand data. The framework provides a method to gather climate data and energy demand data for the location of interest. This enables the user to use the framework as a decision support tool, 
e.g. determining the optimized mean and standard deviation of the levelized cost of hydrogen for a photovoltaic-electrolyzer system in Brussels, Belgium. 
Evidently, the climate and demand database can be extended by your own climate and energy demand data.

Despite our focus to the hydrogen-based system models, RHEIA is not limited to only this domain. Based on the structure of the framework and a tailored Python wrapper RHEIA enables to connect any Python-based, open-source and closed-source system model.
