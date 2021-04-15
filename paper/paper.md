---
title: 'RHEIA: Robust optimization of renewable Hydrogen and dErIved energy cArrier systems'
tags:
  - Python
  - hydrogen-based systems
  - robust optimization
  - uncertainty quantification
authors:
  - name: Diederik Coppitters
    orcid: 0000-0001-9480-2781
    affiliation: "1, 2, 3"
  - name: Panagiotis Tsirikoglou
    orcid: xxx
    affiliation: 4
  - name: Ward De Paepe
    orcid: 0000-0001-5008-2946
    affiliation: 1
  - name: Francesco Contino
    orcid: 0000-0002-8341-4350
    affiliation: 5
affiliations:
 - name: Thermal Engineering and Combustion Unit, University of Mons
   index: 1
 - name: Fluid and Thermal Dynamics, Vrije Universiteit Brussel
   index: 2
 - name: Combustion and Robust Optimization Group, Vrije Universiteit Brussel and Universit\'e Libre de Bruxelles
   index: 3
 - name: Limmat Scientific AG
   index: 4
 - name: Institute of Mechanics, Materials and Civil Engineering, Universit\'e catholique de Louvain
   index: 5
date: 1 May 2021
bibliography: paper.bib
---

# Summary

A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

When creating a hydrogen-based HRES, an optimal design is found through simulations, assuming
fixed model parameters. However, during real-life operation, such system parameters are subject
to uncertainties (e.g. unexpected expenditures). Therefore, the resulting physical plant might
produce drastically different results, compared to the predicted outcome. To address this issue,
uncertainty on the system parameters should be considered. An optimal design under uncertainties
is found through Robust Design Optimization (RDO). The provided robust design is least sensitive
to input variations, and therefore ensuring maximum reliability during its lifetime (Fig. 1).
When applying RDO to hydrogen-based HRES, the current state of the art has some limitations. The
HRES models in RDO applications are assumed linear, optimized for a single economic objective,
with many system parameters still assumed exact [7,8]. Therefore, the fidelity towards reality is
limited and the application to real-life situations is restricted. The main challenge of this thesis is to
tackle all these limitations simultaneously. We will develop a framework for RDO based on
accurate hydrogen-based HRES models, which provides cross-field robust designs, enhanced by
experimental data.


In design optimization studies of HRES, the optimal integration of battery systems and hydrogen-based energy systems in grid-connected 
applications received limited attention. Moreover, the model parameters are often assumed fixed and free from inherent variations, 
while the rare consideration of uncertainty is limited to linear models and only a handful of uncertain parameters <5, 
characterized by generic ranges based on assumptions. Akbari et al. evaluated a distributed energy system, 
subject to a general variability of 20% for a handful of financial parameters and demand parameters. 
Parisio~et~al. considered the converter efficiencies related to electricity and heat demand to be uncertain between a general range of 10% 
on a linear model of an energy hub. These linear models are subject to large inherent uncertainty, while the variation of other 
highly-uncertain parameters (e.g. investment cost, lifetime) during real-world design, planning and operation is ignored. Moreover, generic 
variability ranges assume equal weights for every uncertainty, which leads to biased results. Combined, these assumptions bring forward designs 
that are highly sensitive to real-world uncertainties and result in a drastic mismatch between simulated and actual performances. 
To fill the research gap on design optimization under uncertainty of grid-connected, HRES including hydrogen storage and battery storage, 
we provide the following main contributions: the significant techno-economic uncertain parameters are characterized by their uncertainty as 
described in literature; to handle this large stochastic dimension, the advantages of the sparse PCE algorithm developed by Abraham~et~al. 
are exploited for the first time in a surrogate-assisted RDO algorithm; the Cumulative Density Functions (CDF) of the 
optimized designs are used to compare the respective stochastic performances, which provides new insights into the probability of attaining 
an affordable levelized cost of electricity when combining battery storage and hydrogen storage with a PV array.


Balancing of intermittent energy such as solar energy can be achieved by batteries and hydrogen-based storage. 
However, combining these systems received limited attention in a grid-connected framework and its design optimization is often 
performed assuming fixed parameters. Hence, such optimization induces designs highly sensitive to real-world uncertainties, 
resulting in a drastic mismatch between simulated and actual performances. To fill the research gap on design optimization of grid-connected, 
hydrogen-based renewable energy systems, we performed a computationally efficient robust design optimization under different scenarios and 
compared the stochastic performance based on the corresponding cumulative density functions. This paper provides the optimized stochastic designs 
and the advantage of each design based on the financial flexibility of the system owner. The results illustrate that the economically preferred 
solution is a photovoltaic array when the self-sufficiency ratio is irrelevant 30%. When a higher self-sufficiency ratio threshold 
is of interest, i.e. up to 59%, photovoltaic-battery designs and photovoltaic-battery-hydrogen designs provide the cost-competitive 
alternatives which are least-sensitive to real-world uncertainty. Conclusively, including storage systems improves the probability of attaining 
an affordable levelized cost of electricity over the system lifetime. Future work will focus on the integration of the heat demand.


# Statement of need

A Statement of Need section that clearly illustrates the research purpose of the software.

A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.

The robust design optimization framework has been applied on hydrogen-based 
energy systems, developed in Python: A directly-coupled 
photovoltaic-electrolyzer system [@Coppitters2019a] and a power-to-power system 
(photovoltaic-battery-hydrogen system [@coppitters2020robust], where the 
hydrogen system consists of a Proton Exchange Membrane electrolyzer, hydrogen 
storage tank and PEM fuel cell). In addition, an Aspen Plus model of a 
power-to-ammonia model has been connected to the framework [@verleysen2020can].
Other Aspen Plus models have been optimized under uncertainty as well through 
RHEIA: a micro gas turbine with carbon capture plant [@giorgetti2019] and a 
micro gas turbine [@Paepe2019a]. Finally, uncertainty quantification has been
performed on an EnergyScope model [@limpensa2020impact].


# Future work

sparce PCE
additional optimizers
extra H2 models

# Acknowledgements

The first author acknowledges the support of Fonds de la Recherche Scientifique - FNRS [35484777 FRIA-B2].

# References