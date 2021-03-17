.. _lab:functionalities:

A short description
===================

RHEIA provides a tool to perform multi-objective deterministic optimization, robust optimization and uncertainty quantification on hydrogen-based energy systems.
Despite its potential as an energy carrier, hydrogen is scarsely integrated in design optimization studies of hybrid renewable energy systems.
Moreover, when hydrogen is considered, mainly deterministic model parameters (i.e. perfectly known and free from inherent variation) are assumed, despite the uncertainty
that affects the performance of such systems (e.g. market values, climate conditions).
Therefore, RHEIA combines surrogate-assisted robust design optimization with hydrogen-based energy systems, to unlock robust design which are least-sensitive to the random environment,
based on techno-economic uncertainty adopted from scientific literature. Moreover, the uncertainty quantification enables to define the Sobol' indices of an optimized design,
which illustrate the uncertainties that dominate the variance of the quantity of interest.

The deterministic optimization can be performed for multiple objectives, while the robust optimization is performed on the mean and standard deviation of the objective.
To determine the mean and standard deviation for each design sample evaluated during the robust optimization procedure, an uncertainty quantification is performed.
Uncertainty quantification can be performed separately as well, which provides, in addition to the statistical moments, the Sobol' indices, probability density function and
cumulative distribution function. 

The hydrogen models encollapse the 5 main topics were hydrogen proves interesting: power-to-power, power-to-mobility, power-to-gas, power-to-fuel and power-to-industry.
For each model, a specific set of parameters are defined with uncertainty, which are considered during robust design optimization and uncertainty quantification. 
Several models depend on the climate data and/or energy demand data. The framework provides climate data and energy demand data for numerous locations, 
which enables the user to use the framework as a decision support tool, 
e.g. determining the optimized mean and standard deviation of the levelized cost of hydrogen for a photovoltaic-electrolyzer system in Brussels, Belgium. 
Evidently, it is possible to add your own climate and energy demand data.

In addition to the hydrogen-based system models, the framework allows to connect your own model. 
The documentation illustrates how to connect a Python model and a closed-source model.


