.. _lab:methods:

Details on the methods
======================

The math behind the different methods, with appropriate references.

Uncertainty Quantification
--------------------------

.. _lab:pce:

Polynomial Chaos Expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^

The PCE representation of the system model consists of a series of orthogonal polynomials :math:`\Psi_i` with corresponding coefficients :math:`u_i`:

:math:`\mathcal{M}^{\mathrm{PCE}}(\pmb{X}) = \sum_{\pmb{\alpha} \in \mathcal{A}} u_{\pmb{\alpha}} \Psi_{\pmb{\alpha}} (\pmb{X}) \approx \mathcal{M}(\pmb{X})` 

where :math:`\pmb{\alpha}` are the multi-indices and :math:`\mathcal{A}` is the considered set of multi-indices, for which the size is defined by a truncation scheme. 
A typical truncation scheme is limiting the polynomial order up to a certain degree, which constrains the number of multi-indices in the set to \cite{Sudret2014}:

:math:`|\mathcal{A}^{M,p}| = \dfrac{(p + M)!}{p!M!}`,

where :math:`p` corresponds to the polynomial order and :math:`M = |\pmb{X}|` is the stochastic dimension, i.e. number of random variables.
The polynomial family that is orthogonal with respect to the assigned probability distributions are known for classic distributions~\cite{xiu2002wiener}. 
As an example, uniformly distributed stochastic input parameters associate with the Legendre polynomials.
Consequently, :math:`P+1` coefficients are present in a full PCE. To quantify these coefficients, Least-Square Minimization is applied, based on actual system model evaluations~\cite{Sudret2014}. 
To ensure a well-posed Least-Square Minimization, :math:`2(P+1)` system model evaluations are usually required~\cite{Sudret2014}. 
When the coefficients are quantified, the mean :math:`\mu` and standard deviation :math:`\sigma` of the objective follow analytically:

:math:`\mu = u_0`,

:math:`\sigma^2 = \sum_{i=1}^P u_i^2`.

Next to these statistical moments, the contribution of each stochastic parameter to the variance of the objective provides valuable information on the system behavior under uncertainty. 
Generally, this contribution is quantified through Sobol' indices. P
CE provides an analytical solution to quantify these Sobol' indices through post-processing of the coefficients (i.e.\ no additional model evaluations required). The total-order Sobol' indices (:math:`S_i^{T,\mathrm{PC}}`) quantify the total impact of a stochastic input parameter, including all interactions: 

:math:`S_i^{T,\mathrm{PC}} = \sum_{\alpha \in A_i^T}^{} u_\alpha^2/\sum_{i=1}^P u_i^2 ~~~~~~ A_i^T = \{\alpha \in A | \alpha_i > 0\}`.

For every coefficient that is characterized by considering input parameter :math:`i`, among others, at an order :math:`> 0` is added to the total Sobol' index :math:`S_i`.

optimization
------------

.. _lab:choosepop:

Choosing the population size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The computational budget and population size are fixed by the user. Both parameters provide an indication of the number of generations performed,
i.e. number of generations >= computational budget / population size. The population is spread over the design space. The higher the population size,
the higher the number of explored areas in the design space. On the one hand, when the population size is small and the the number of mutations are limited, the population
might converge to a local optimum. On the other hand, when the population size is large, a significant computational budget is spent at each generation,
which limits the exploitation and thus results in suboptimal designs. There is no strict rule on the population size, as it highly depends on the number of design variables,
the non-linearity of the relation between input-output of the system model and the number of local optima in this relation.
Nevertheless, based on the experience of the authors in engineering optimization, the population size is suggested between 20 and 50. Below 20, the possibility of ending in a
local optimum is significant, while a population size larger than 50 does not add significant improvement in design space exploration as opposed to the increase in cost per generation.
In other words, under a limited computational budget and large population, the number of generations might result in unsatisfactory exploitation.  
To ensure sufficient exploitation, we suggest to reach a number of generations of at least 75 generations. Above 250 generations, the gain in exploitation becomes limited. 

The probability of crossover and mutation are user-defined constants which support the exploitation and exploration, respectively.
Typically, the crossover probability is at least 0.85, while the probability of mutation usually remains below 0.1.




.. _lab:ssnsga2:

NSGA-II
^^^^^^^

NSGA-II is a multi-objective genetic algorithm, suitable for optimization of complex, non-linear models~\cite{Deb2002a}. 
First, this algorithm creates a set of design samples (i.e.\ population), based on Latin Hypercube Sampling~\cite{Stein1987}. 
Thereafter, a second population (i.e.\ children) is generated with characteristics based on the previous population (i.e.\ parents), 
following crossover and mutation rules. Each design sample out of both populations is evaluated through the uncertainty quantification algorithm 
and sorted based on their dominance in the objectives (i.e.\ mean and standard deviation of the LCOE). The top half of the sorted samples remain and represent the next population. 
The algorithm continues until either convergence is reached or the maximum number of iterations is realized. When the process is finalized, either the solution converged to a single design sample, 
achieving minimum mean and standard deviation, or a set of optimized design samples where each sample dominates every other design sample in at least one objective.


Latin Hypercube Sampling
^^^^^^^^^^^^^^^^^^^^^^^^

