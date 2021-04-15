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
 - name: Combustion and Robust Optimization Group, Vrije Universiteit Brussel and Université Libre de Bruxelles
   index: 3
 - name: Limmat Scientific AG
   index: 4
 - name: Institute of Mechanics, Materials and Civil Engineering, Université catholique de Louvain
   index: 5
date: 1 May 2021
bibliography: paper.bib
---

# Summary

To limit global CO2 emissions, hydrogen enables to integrate the massively deployed 
intermittent renewable power generation (solar and wind) in the power, heating
and mobility sector. The techno-economic and environmental design of 
hydrogen-based systems are subject to uncertainties, such as the costs related
to the production of renewable hydrogen and the energy consumption of hydrogen-fueled
buses. Therefore, RHEIA provides a robust design optimization algorithm, which 
yields the designs least-sensitive to its random environment (i.e. the robust design). 
These robust designs ensure with the highest probability that the designs will operate
near its expected performance in real-life conditions. 
Moreover, the main drivers of the uncertainty on the
performance of the optimized designs are characterized. This enables to provide
effective guidelines to further reduce the uncertainty of the optimized designs. 

# Statement of need

A Statement of Need section that clearly illustrates the research purpose of the software.

In design optimization of renewable energy systems, incorporating hydrogen systems
is still an anomaly [@Eriksson2017] and often only deterministic model 
parameters are assumed. When uncertainties are considered, the applications are 
often limited to linear models with a handful of uncertainties (<5), 
characterized by generic ranges. Combined, these assumptions bring forward designs 
that are highly sensitive to real-world uncertainties and result in a drastic mismatch between simulated and actual performances.
Handling a large set of uncertainties leads to computational issues.
To fill these research gaps, RHEIA provides a multi-objective robust design optimization 
algorithm, for which the computational burden of uncertainty quantification is tackled
by the Polynomial Chaos Expansion algorithm. In addition, RHEIA includes Python-based
system models for the main valorization pathways of hydrogen: power-to-fuel, power-to-power
and power-to-mobility. The significant techno-economic and environmental uncertainties
for these models are characterized based on scientific literature.     


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