import numpy as np
import scipy as sp
import itertools
from scipy.stats import *
import os
from sobol_mod import *
import multiprocessing as mp


class Data:

    def __init__(self, inputs, tc):
        """
        Parameters
        ----------
        inputs : dict
            input dictionary with information on
            the uncertainty quantification.
        tc : object
            object that contains information on the test case.

        """
        self.inputs = inputs
        self.path = os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0]
        self.tc = tc
        self.path_res = None

        self.stoch_data = {}

    def create_samples_file(self):
        """
        Creating the file that saved the input samples and model outputs.
        """
        self.path_res = os.path.join(self.path,
                                     'RESULTS',
                                     self.tc.case,
                                     'UQ',
                                     self.inputs['results dir'],
                                     )

        if not os.path.exists(self.path_res):

            os.makedirs(self.path_res)

        self.filename_samples = os.path.join(self.path_res, 'samples')

        if not os.path.isfile(self.filename_samples):
            with open(self.filename_samples, "w") as f:
                for name in self.stoch_data['names'] + \
                        self.inputs['objective names']:
                    f.write('%25s ' % name)
                f.write('\n')

    def read_stoch_parameters(self, var_values=[]):
        """
        Read in the stochastic design space
        and save the information in a dictionary

        Parameters
        ----------
        var_values : list, optional
            The design variable values in case of robust optimization.
            The default is [].

        """

        if len(var_values) != len(self.tc.var_dict):
            raise ValueError(
                """When performing UQ, make sure that no design variableS
                   are present in design_space. Each variable should be
                   converted into a model parameter.""")

        tmp = {}
        for i, key in enumerate(self.tc.var_dict):
            tmp[key] = var_values[i]

        det_dict = {**self.tc.par_dict, **tmp}

        minimum, maximum, mean, deviation = [], [], [], []
        for key in self.tc.upar_dict:
            if self.tc.upar_dict[key][0] == 'absolute':
                minimum.append(det_dict[key] - self.tc.upar_dict[key][2])
                maximum.append(det_dict[key] + self.tc.upar_dict[key][2])
                deviation.append(self.tc.upar_dict[key][2])
            elif self.tc.upar_dict[key][0] == 'relative':
                minimum.append(det_dict[key] *
                               (1. - self.tc.upar_dict[key][2]))
                maximum.append(det_dict[key] *
                               (1. + self.tc.upar_dict[key][2]))
                deviation.append(self.tc.upar_dict[key][2] * det_dict[key])
            else:
                raise ValueError(""" The relation of the deviation to the mean
                                     should be 'relative' or 'absolute'.""")

            mean.append(det_dict[key])

        self.stoch_data = {
            'names': list(
                self.tc.upar_dict.keys()),
            'types': [
                x[1] for x in list(
                    self.tc.upar_dict.values())],
            'minimum': minimum,
            'maximum': maximum,
            'mean': mean,
            'deviation': deviation}


class RandomExperiment(Data):

    def __init__(self, my_data, objective_position):
        self.my_data = my_data
        self.objective_position = objective_position

        self.dimension = len(my_data.stoch_data['types'])
        self.dists = [None] * len(my_data.stoch_data['types'])
        self.polydists = [None] * len(my_data.stoch_data['types'])
        self.polytypes = [None] * len(my_data.stoch_data['types'])
        self.size = None
        self.seed = None
        self.X_u = None      # UNSCALED DoE (uncorrelated)
        self.U = None       # SCALED DoE (uncorrelated)

        self.Y = None
        self.DoE = None

    def n_terms(self):
        '''
        This method sets the number of samples to 2*(p+n)!/p!n!,
        i.e the number of terms in the full PC
        expansion of order p in n random variables.

        '''
        n = len(self.my_data.stoch_data['mean'])
        p = self.my_data.inputs['pol order']
        result = 1.
        mmin = min(n, p)
        for i in range(mmin):
            result *= p + n - i
        result_terms = int(result / np.math.factorial(mmin))
        self.n_samples = 2 * result_terms

    def read_previous_samples(self, create_only_samples):
        """
        Read the previously evaluated samples
        and store them for future PCE construction.

        Parameters
        ----------
        create_only_samples : bool
            the boolean that indicates if samples
            should only be created.

        """

        with open(self.my_data.filename_samples, 'r') as f:
            lines = f.readlines()
            x_int = np.zeros(
                (len(lines) - 1, len(self.my_data.stoch_data['mean'])))
            y_int = np.zeros((len(lines) - 1, 1))

            if len(lines) > 1:
                if (len(lines[-1].split()) != len(
                        self.my_data.stoch_data['mean']) +
                        len(self.my_data.inputs['objective names'])):
                    raise SyntaxError(
                        """The samples file is not properly formatted
                           or it already contains samples
                           without model output.""")
                elif create_only_samples:
                    raise ValueError(
                        """The samples file already contains samples
                           with model output.
                           Consider changing the result directory
                           or switching "create only samples" to False.""")
                for i, line in enumerate(lines[1:]):
                    li = line.split()
                    x_int[i] = [float(el) for el in li[:len(
                        self.my_data.stoch_data['mean'])]]
                    y_int[i] = float(
                        li[len(self.my_data.stoch_data['mean']) +
                            self.objective_position])

        self.y_prev = np.array(y_int)
        self.x_prev = np.array(x_int)

    def create_distributions(self):
        """
        Create the distributions, polynomial distributions and polynomial types
        based on the stochastic design space. The available distributions
        are uniform and Gaussian distributions.

        """

        for i, j in enumerate(self.my_data.stoch_data['types']):
            if j == 'uniform':
                self.dists[i] = sp.stats.uniform(
                    self.my_data.stoch_data['mean'][i] -
                    self.my_data.stoch_data['deviation'][i],
                    2. * self.my_data.stoch_data['deviation'][i])
                self.polydists[i] = sp.stats.uniform(-1., 2.)
                self.polytypes[i] = 'Legendre'

            elif j == 'Gaussian':
                self.dists[i] = sp.stats.norm(
                    self.my_data.stoch_data['mean'][i],
                    self.my_data.stoch_data['deviation'][i])
                self.polydists[i] = sp.stats.norm(0., 1.)
                self.polytypes[i] = 'Hermite'

    def create_samples(self, size=0):
        """
        Generate the samples for model evaluations. The sampling
        methods available are random sampling and Sobol sampling.
        The number of samples generated is equal to the integer
        `size`. If `size` is equal or lower than zero, no new samples
        are generated. Instead, the required samples are extracted
        from the existing samples file.

        Parameters
        ----------
        size : int, optional
            The number of newly created samples. The default is 0.

        """
        min = self.my_data.stoch_data['minimum']
        max = self.my_data.stoch_data['maximum']
        method = self.my_data.inputs['sampling method']

        if size > 0:
            self.X_u = np.zeros((size, self.dimension))
            if method == 'RANDOM':
                for i in range(self.dimension):
                    self.X_u[:, i] = self.dists[i].rvs(size)

            elif method == 'SOBOL':
                skip = 123456
                XX = np.transpose(i4_sobol_generate(
                    self.dimension, size + len(self.x_prev), skip))
                for i in range(self.dimension):
                    self.X_u[:, i] = self.dists[i].ppf(
                        XX[len(self.x_prev):, i])

            if len(self.x_prev) > 0:
                self.X_u = np.concatenate((self.x_prev, self.X_u))

            self.size = (len(self.X_u))
            self.U = np.zeros((self.size, self.dimension))

        elif size == 0:
            self.X_u = self.x_prev
            self.size = len(self.X_u)
            self.U = np.zeros((self.size, self.dimension))

        else:
            self.X_u = np.split(
                self.x_prev, [len(self.x_prev) + size, len(self.x_prev)])[0]
            self.size = len(self.X_u)
            self.U = np.zeros((self.size, self.dimension))

        for i in range(self.dimension):
            for j in range(self.size):
                self.U[j, i] = (self.X_u[j, i] - min[i]) / \
                    (max[i] - min[i]) * 2. - 1.

    def create_only_samples(self, create_only_samples):
        """

        Add the generated samples to the samples file.
        These samples can be used for model evaluation,
        when the model is not connected to the UQ algorithm.

        Parameters
        ----------
        create_only_samples : bool
            the boolean that indicates
            if samples should only be created

        """

        if create_only_samples:
            with open(self.my_data.filename_samples, 'a+') as f:
                for i, x in enumerate(self.X_u):
                    li = list(x)
                    for j in li:
                        f.write('%25f ' % j)

                    f.write('\n')

    def evaluate(self):
        """
        Evaluate the samples in the model
        and store the samples and outputs in the samples file.

        """
        size = self.n_samples - len(self.x_prev)
        # create uniform/gaussian distributions and corresponding orthogonal
        # polynomials
        self.create_samples(size=size)

        X = self.X_u[-(self.n_samples - len(self.x_prev)):]
        temp_unc_samples = np.tile(
            list(self.my_data.tc.par_dict.values()), (size, 1))

        par_var_list = list(self.my_data.tc.par_dict.keys()) + \
            list(self.my_data.tc.var_dict.keys())
        for j, elem in enumerate(self.my_data.tc.upar_dict.keys()):

            unc_idx = par_var_list.index(elem)

            temp_unc_samples[:, unc_idx] = X[:, j]

        pool = mp.Pool(processes=self.my_data.inputs['n jobs'])

        res = pool.map(self.my_data.tc.evaluate, enumerate(temp_unc_samples))
        pool.close()
        if self.objective_position > len(res[0]) - 1:
            raise IndexError(""" The objective "%s" falls out of
                                 the range of predefined quantities
                                 of interest. Only %i outputs are
                                 returned from the model""" %
                             (self.my_data.inputs['objective names'][
                                 int(self.objective_position)], len(res[0])))
        y_res = [row[self.objective_position] for row in res]

        if self.y_prev.size:
            self.Y = np.vstack((self.y_prev, np.array(y_res).reshape((-1, 1))))
        else:
            self.Y = np.array(y_res).reshape((-1, 1))

        with open(self.my_data.filename_samples, 'a+') as f:
            for i, x in enumerate(X):
                li = list(np.concatenate((x, res[i])))
                for j in li:
                    f.write('%25f ' % j)

                f.write('\n')


class PCE(RandomExperiment):

    def __init__(self, Experiment):
        '''
        Class which creates a Polynomial Chaos Expansion (PCE) object. A PCE is
        characterized by the following attributes:

            - Basis : PC basis functions
            - Coefficients : PC coefficients
            - Moments : Statistical moments
            - sensitivity : Sobol' indices

        '''
        self.my_experiment = Experiment
        self.order = self.my_experiment.my_data.inputs['pol order']
        self.A = 0.0
        self.LOO = 0.0
        self.basis = dict()
        self.coefficients = dict()
        self.sensitivity = dict()
        self.moments = dict()
        self.y_hat = 0.0
        self.psi_sq = 0.0
        self.Info = dict()

    def n_to_sum(self, n, s):
        '''
        This function creates a list
        of all possible vectors of length 'n' that sum to 's'

        Parameters
        ----------
        n : int
            vector's length (dimension)
        s : int
            sum (total order)

        Return
        ------
        The output is a generator object to be called as follows
            - list(n_to_sum(dimension,i))
        '''

        if n == 1:
            yield (s,)
        else:
            for i in range(s + 1):
                for j in self.n_to_sum(n - 1, s - i):
                    yield j + (i,)

    def uq_multindices(self, idx):
        """

        This method returns a set of multi-indices

        Parameters
        ----------
        idx : list
            the range for the number of terms in the PCE

        Returns
        -------
        list
            the list with multi-indices

        """
        dimension = self.my_experiment.dimension

        multindices = list()

        for i in range(self.order + 1):
            multindices.extend(list(self.n_to_sum(dimension, i)))

        return [multindices[i] for i in idx]

    def uq_OLS(self, A, b):
        """


        Parameters
        ----------
        A : array
            the matrix with the response of the basis functions to the samples
        b : array
            the model output for the input samples

        Returns
        -------
        2_D array
            result of the ordinary least-square regression

        """

        m, n = A.shape
        dof = m - n

        sol = np.linalg.lstsq(A, b, rcond=None)
        s2 = np.column_stack(1.0 / dof * sol[1])

        var = np.dot(
            np.row_stack(
                np.diag(
                    np.linalg.inv(
                        np.dot(
                            np.transpose(A),
                            A)))),
            s2)
        return np.column_stack((sol[0], var))

    def uq_calc_A(self, multindices):
        """

        This method builds the matrix containing the basis functions evaluated
        at sample locations, i.e. the matrix A in Au = b

        Parameters
        ----------
        multindices : list
            the list with the multi-indices

        Returns
        -------
        A : array
            the matrix with the evaluated basis functions for the samples

        """

        dimension = self.my_experiment.dimension  # number of random dimension
        n = len(multindices)             # number of basis functions
        m = self.my_experiment.size   # number of samples
        U = self.my_experiment.U
        A = np.ones([m, n])
        for i, multiindex in enumerate(multindices):
            for j in range(dimension):
                deg = multiindex[j]
                if self.my_experiment.polytypes[j] == 'Legendre':
                    '''
                    Note: For the uniform distribution,
                    Legendre polynomials are used
                    '''
                    A[:, i] *= sp.special.eval_legendre(deg, U[:, j])

                elif self.my_experiment.polytypes[j] == 'Hermite':
                    '''
                    We use the PROBABILISTS' version
                    of Hermite polynomials,
                    which are orthogonal with respect to the STANDARD
                    normal variable N(0,1).
                    '''
                    A[:, i] *= sp.special.eval_hermitenorm(deg, U[:, j])

                elif self.my_experiment.polytypes[j] == 'Jacobi':
                    '''
                    For the beta distribution, we use Jacobi polynomials
                    '''
                    alpha, beta = self.my_experiment.polydists[j].args[
                        0], self.my_experiment.polydists[j].args[1]
                    A[:,
                      i] *= sp.special.eval_jacobi(deg,
                                                   alpha - 1,
                                                   beta - 1,
                                                   U[:,
                                                     j])

        return A

    ###
    # Start module uncertainty quantification
    ###
    def uq_run(self):
        """

        Solve Ordinary Least Squares problem
        Full PC expansion is assumed containing n_terms(dimension,order)


        """
        self.residu = self.my_experiment.Y

        n_terms = int(self.my_experiment.n_samples / 2)

        self.basis['model'] = range(n_terms)
        # Full PC basis
        self.basis['multi-indices'] = self.uq_multindices(self.basis['model'])
        self.basis['polytypes'] = self.my_experiment.polytypes

        self.A = self.uq_calc_A(self.basis['multi-indices'])
        self.coefficients = self.uq_OLS(self.A, self.my_experiment.Y)

        self.get_statistics(
            mean=True,
            variance=True,
            skewness=False,
            kurtosis=False)

    ###
    # End module
    ###

    ###
    # Start module for extracting statistical information from PC expansion
    ###
    def get_statistics(
            self,
            mean=True,
            variance=True,
            skewness=False,
            kurtosis=False):
        '''
        This function calculates high order moments (up to order 4) by
        taking advantage of the fact that any permutation of indices
        will lead to the same value for the summand. See report.

        Returns Statistics: mean, std. deviation, skewness, kurtosis

        '''
        alpha = 0.05
        dof = self.my_experiment.size - len(self.basis['multi-indices'])
        t_star = t.ppf(1 - alpha / 2, dof)

        u = self.coefficients

        self.moments['mean'] = (0.0, 0.0)
        self.moments['variance'] = (0.0, 0.0)
        self.moments['skewness'] = (0.0, 0.0)
        self.moments['kurtosis'] = (0.0, 0.0)

        if len(self.basis['model']) == 0:
            print('WARNING :: Empty model!')

        else:

            if mean:
                if self.basis['model'][0] == 0:
                    self.moments['mean'] = (u[0, 0], t_star * np.sqrt(u[0, 1]))

                elif self.basis['model'][0] != 0:
                    print('There is no mean term')

            n_terms = len(self.basis['model'])

            if variance:
                self.psi_sq = self.get_psi_sq()

                var = 0.0

                if self.basis['model'][0] == 0:
                    for i in range(1, n_terms):
                        var += self.psi_sq[i] * u[i, 0]**2

                elif self.basis['model'][0] != 0:
                    for i in range(n_terms):
                        var += self.psi_sq[i] * u[i, 0]**2

                self.moments['variance'] = (var, 0.0)

            if skewness:
                P = n_terms - 1
                psi_cub = self.get_psi_cub()

                skew = 0.0

                permutations = self.get_permutations(range(P), 'skewness')
                for count, elem in enumerate(permutations):
                    i = elem[0] + 1
                    j = elem[1] + 1
                    k = elem[2] + 1

                    summand = u[i, 0] * u[j, 0] * u[k, 0] * psi_cub[count]

                    if count < P:
                        skew += 1.0 * summand

                    elif count >= P and count < P + 2 * sp.special.binom(P, 2):
                        skew += 3.0 * summand

                    elif count >= P + 2 * sp.special.binom(P, 2):
                        skew += 6.0 * summand

                skew /= var**(3.0 / 2.0)

                self.moments['skewness'] = (skew, 0.0)

            if kurtosis:
                P = n_terms - 1
                psi_four = self.get_psi_four()
                kurt = 0.0
                permutations = self.get_permutations(
                    range(n_terms - 1), 'kurtosis')

                for count, elem in enumerate(permutations):
                    i = elem[0] + 1
                    j = elem[1] + 1
                    k = elem[2] + 1
                    m = elem[3] + 1

                    summand = u[i, 0] * u[j, 0] * \
                        u[k, 0] * u[m, 0] * psi_four[count]

                    if count < P:
                        kurt += 1.0 * summand
                    elif count >= P and count < P + P * (P - 1):
                        kurt += 4.0 * summand
                    elif (count >= P + P * (P - 1) and
                          count < P + P * (P - 1) + sp.special.binom(P, 2)):
                        kurt += 6.0 * summand
                    elif (count >= P + P * (P - 1) + sp.special.binom(P, 2) and
                          count < P + P * (P - 1) + sp.special.binom(P, 2) +
                          sp.special.binom(P, 3) * 3):
                        kurt += 12.0 * summand
                    elif count >= (P + P * (P - 1) + sp.special.binom(P, 2) +
                                   sp.special.binom(P, 3) * 3):
                        kurt += 24.0 * summand

                kurt = kurt / var**(4.0 / 2.0) - 3.0

                self.moments['kurtosis'] = (kurt, 0.0)

    def get_permutations(self, lis, string):
        '''
        This function returns a set of indices {ijk} (resp. {ijkl}) that are
        actually necessary for calculating the skewness (resp. kurtosis).
        It takes the repetition of the indices into account.

        Inputs:
        -------
        lis : list with the indices for the range of number of terms

        s : string
            string that can take only two different values,
            i.e. 'skewness' or 'kurtosis'

        Returns:
        --------
        B : list
            set of indices for skewness and curtosis calculation
        '''
        A = []
        B = []

        if string == 'skewness':
            k = 3

            for i in xrange(1, k + 1):
                A.append(list(itertools.combinations(lis, i)))

            # --) factor 1.0 (i,i,i), binom(P,1)
            [B.append(i * k) for i in A[0]]

            for i in range(len(A[1])):
                for j in range(k - 1):
                    # --) factor 3.0 (i,j,i), binom(P,2) * 2
                    B.append(A[1][i] + (A[1][i][j],))

            # --) factor 6.0 (i,j,k), binom(P,3)
            [B.append(i) for i in A[2]]

        elif string == 'kurtosis':
            k = 4

            for i in xrange(1, k + 1):
                A.append(list(itertools.combinations(lis, i)))

            # --) factor 1.0 (i,i,i,i), binom(P,1)
            [B.append(elem * k) for elem in A[0]]

            for elemi in A[0]:
                for elemj in A[0]:
                    if elemi != elemj:
                        # --) factor 4.0 (i,i,i,j), P * (P-1)
                        B.append(elemi * 3 + (elemj[0],))

            # --) factor 6.0 (i,j,i,j), binom(P,2)
            [B.append(elem * 2) for elem in A[1]]

            for i in range(len(A[2])):
                for j in range(k - 1):
                    # --) factor 12.0 (i,j,k,i), binom(P,3) * 3
                    B.append(A[2][i] + (A[2][i][j],))

            # --) factor 24.0 (i,j,k,l), binom(P,4)
            [B.append(elem) for elem in A[3]]

        return B

    def get_psi_sq(self):
        """

        Calculate the term <psii,psij>

        Returns
        -------
        psi_sq : array
            the term <psii,psij>

        """
        d = self.my_experiment.dimension

        multindices = self.basis['multi-indices']

        n_terms = len(multindices)

        psi_sq = np.ones(n_terms,)

        for i in range(n_terms):
            for j in range(d):
                deg = multindices[i][j]

                if self.my_experiment.polytypes[j] == 'Legendre':

                    xi, wi = sp.special.p_roots(deg + 1)

                    '''
                    We need to integrate exactly the SQUARE
                    of the Legendre polynomial. For example,
                    if the Legendre polynomial is of order (deg),
                    the numerical integration must be exact
                    till order (deg**2). Thus, we need at least
                    (deg+1) abscissas' and weights.
                    '''
                    poly = sp.special.legendre(deg)**2
                    psi_sq[i] *= 1.0 / 2 * sum(wi * poly(xi))

                elif self.my_experiment.polytypes[j] == 'Hermite':

                    xi, wi = sp.special.he_roots(deg + 1)

                    '''
                    sp.special.he_roots(deg) and
                    np.polynomial.hermite_e.hermegauss(deg)
                    returns the same abscissas'
                    but different weights (!). There is a factor 2
                    between the two. Given the fact that the integral of
                    the standard Gaussian must be 1,
                    np.polynomial.hermite_e.hermegauss(deg)
                    provides the right weights.
                    '''
                    poly = sp.special.hermitenorm(deg)**2

                    # 2*wi*poly(xi)
                    psi_sq[i] *= 1.0 / np.sqrt(2 * np.pi) * sum(wi * poly(xi))

                elif self.my_experiment.polytypes[j] == 'Jacobi':
                    '''
                    Note : Defined over the interval [-1,1] !!!
                    '''
                    alpha, beta = self.my_experiment.polydists[j].args[
                        0], self.my_experiment.polydists[j].args[1]
                    xi, wi = sp.special.roots_jacobi(
                        deg + 1, alpha - 1, beta - 1)
                    poly = sp.special.jacobi(deg, alpha - 1, beta - 1)**2

                    psi_sq[i] *= (1.0 / (2**(alpha + beta - 1) *
                                         sum(wi * poly(xi)) *
                                         sp.special.beta(alpha, beta)))

        return psi_sq

    def get_psi_cub(self):
        """


        Returns
        -------
        psi_cub : array


        """
        dimension = self.my_experiment.dimension

        copy = self.basis['model'][:]
        copy.remove(0)

        multindices = self.uq_multindices(copy)

        n_terms = len(copy)

        permutations = self.get_permutations(range(n_terms), 'skewness')

        psi_cub = np.ones(len(permutations),)
        for count, elem in enumerate(permutations):
            i = elem[0]
            j = elem[1]
            k = elem[2]

            for l in range(dimension):
                degi = multindices[i][l]
                degj = multindices[j][l]
                degk = multindices[k][l]

                if self.my_experiment.polytypes[l] == 'Legendre':

                    poly = sp.special.legendre(degi) * \
                        sp.special.legendre(degj) * \
                        sp.special.legendre(degk)

                    n = int(ceil((degi + degj + degk + 1.0) / 2))
                    xi, wi = sp.special.p_roots(n)

                    psi_cub[count] *= 1.0 / 2 * sum(wi * poly(xi))

                elif self.my_experiment.polytypes[l] == 'Hermite':

                    poly = sp.special.hermitenorm(degi) * \
                        sp.special.hermitenorm(degj) * \
                        sp.special.hermitenorm(degk)

                    n = int(ceil((degi + degj + degk + 1.0) / 2))
                    xi, wi = sp.special.he_roots(n)

                    psi_cub[count] *= 1.0 / \
                        np.sqrt(2 * np.pi) * sum(2 * wi * poly(xi))

                    '''
                    The analytical solution is given by 2**((i+j+k)/2)
                    i! j! k! np.sqrt(np.pi) / (((i+j-k)/2)! ((j+k-i)/2)!
                    ((k+i-j)/2)!) in book: "Special functions"
                    by Andrews-Askey-Roy (Equation 6.8.3)
                    '''

                elif self.my_experiment.polytypes[l] == 'Jacobi':
                    alpha, beta = self.my_experiment.polydists[l].args[
                        0], self.my_experiment.polydists[l].args[1]

                    poly = sp.special.jacobi(degi, alpha - 1, beta - 1) * \
                        sp.special.jacobi(degj, alpha - 1, beta - 1) * \
                        sp.special.jacobi(degk, alpha - 1, beta - 1)

                    n = int(ceil((degi + degj + degk + 1.0) / 2))
                    xi, wi = sp.special.j_roots(n, alpha - 1, beta - 1)

                    psi_cub[count] *= 1. / (2**(alpha + beta - 1) *
                                            sum(wi * poly(xi)) *
                                            sp.special.beta(alpha, beta))

                elif self.my_experiment.polytypes[l] == 'Laguerre':
                    a, b = params[l]

                    n = int(ceil((degi + degj + degk + 1.0) / 2))
                    xi, wi = sp.special.la_roots(n, a - 1)
                    poly = sp.special.genlaguerre(degi, a) * \
                        sp.special.genlaguerre(degj, a) * \
                        sp.special.genlaguerre(degk, a)

                    psi_cub[count] *= 1.0 / \
                        (sp.special.gamma(a)) * sum(wi * poly(xi / b))

        return psi_cub

    def get_psi_four(self):
        '''

        '''

        dimension = self.my_experiment.dimension

        copy = self.basis['model'][:]
        copy.remove(0)

        n_terms = len(copy)

        multindices = self.uq_multindices(copy)

        permutations = self.get_permutations(range(n_terms), 'kurtosis')

        psi_four = np.ones(len(permutations),)

        for count, elem in enumerate(permutations):
            i = elem[0]
            j = elem[1]
            k = elem[2]
            l = elem[3]

            for m in range(dimension):
                degi = multindices[i][m]
                degj = multindices[j][m]
                degk = multindices[k][m]
                degl = multindices[l][m]

                if self.my_experiment.polytypes[m] == 'Legendre':

                    poly = sp.special.legendre(degi) * \
                        sp.special.legendre(degj) * \
                        sp.special.legendre(degk) * \
                        sp.special.legendre(degl)

                    n = int(ceil((degi + degj + degk + degl + 1.0) / 2))
                    xi, wi = sp.special.p_roots(n)

                    psi_four[count] *= 1.0 / 2 * sum(wi * poly(xi))

                elif self.my_experiment.polytypes[m] == 'Hermite':

                    poly = sp.special.hermitenorm(degi) * \
                        sp.special.hermitenorm(degj) * \
                        sp.special.hermitenorm(degk) * \
                        sp.special.hermitenorm(degl)

                    n = int(ceil((degi + degj + degk + degl + 1.0) / 2))
                    xi, wi = sp.special.he_roots(n)

                    psi_four[count] *= 1.0 / \
                        np.sqrt(2 * np.pi) * sum(2 * wi * poly(xi))

                elif self.my_experiment.polytypes[m] == 'Jacobi':
                    alpha, beta = self.my_experiment.polydists[m].args[
                        0], self.my_experiment.polydists[m].args[1]

                    poly = sp.special.jacobi(degi, alpha - 1, beta - 1) * \
                        sp.special.jacobi(degj, alpha - 1, beta - 1) * \
                        sp.special.jacobi(degk, alpha - 1, beta - 1) * \
                        sp.special.jacobi(degl, alpha - 1, beta - 1)

                    n = int(ceil((degi + degj + degk + degl + 1.0) / 2))
                    xi, wi = sp.special.j_roots(n, alpha - 1, beta - 1)

                    psi_four[count] *= 1. / (2**(alpha + beta - 1) *
                                             sum(wi * poly(xi)) *
                                             sp.special.beta(alpha, beta))

        return psi_four
    ###
    # End module
    ###

    ###
    # Start module sensitivity
    ###
    def uq_calc_Sobol(self):
        '''
        This method calculates the Sobol' indices Si of PCE.
        They represent the fraction of the total variance
        that can be attributed to each input variable i (Si)
        or combinations of variables (i1,i2,...,is).
        The total Sobol indices quantify the total effect
        of variable i, i.e. including the first order effect
        and the interactions with the other variables.

        '''

        dimension = self.my_experiment.dimension

        var = self.moments['variance'][0]

        coeff = self.coefficients

        psi_sq = self.get_psi_sq()  # vector of size (P+1)

        # Create list of all of terms
        terms = list()
        for i in range(1, self.order + 1):
            terms.extend(list(itertools.combinations(range(dimension), i)))

        copy = self.basis['multi-indices'][:]
        del copy[0]

        # Compute the Sobol' indices
        Si = np.zeros(len(terms),)
        for i, multindex in enumerate(copy):
            for j, term in enumerate(terms):
                if (len(np.nonzero(multindex)[0]) == len(term)) and (
                        np.nonzero(multindex)[0] == term).all():
                    Si[j] += 1.0 / var * coeff[i + 1][0]**2 * psi_sq[i + 1]

                    break

        # Compute the total Sobol' indices
        STi = np.zeros(dimension,)
        for i in range(dimension):
            idx = np.array([item for item in range(len(terms))
                            if (terms[item] == np.array([i])).any()])
            for j in range(len(idx)):
                STi[i] += Si[idx[j]]

        self.sensitivity['Si'] = Si
        self.sensitivity['STi'] = STi
        self.sensitivity['terms'] = terms

    ###
    # End module
    ###

    def uq_calc_LOO(self):
        '''
        This function returns the best model among a set of models.
        The best model is chosen using modified cross validation technique
        and especially using a Leave-One-Out (LOO) approach.

        Di = (y[i] - y_hat[i]) / (1 - h[i])

        '''
        n = self.my_experiment.size
        Y = self.my_experiment.Y
        y_hat = np.row_stack((np.dot(self.A, self.coefficients[:, 0])))

        B = np.linalg.inv(np.dot(np.transpose(self.A), self.A))
        C = np.dot(B, np.transpose(self.A))
        h = np.dot(self.A, C)

        D = np.diag(h)
        self.LOO = 0.
        for i in range(n):
            deltai = (Y[i, 0] - y_hat[i, 0]) / (1 - D[i])

            self.LOO += 1.0 / float(n) * deltai**2. / (np.var(Y))

        if self.LOO > 1.:
            raise ValueError("""The LOO error is higher than 1.
                                Evaluate the UQ case.""")

    def uq_print(self):
        """
        This module prints an overview of the inputs and results of the PCE.
        Additionally, it writes the PCE information and Sobol' indices
        in the result files.'
        """

        m = self.moments['mean']
        var = self.moments['variance']
        skew = self.moments['skewness']
        kurt = self.moments['kurtosis']

        print('%s Polynomial chaos output %s \n' % ('-' * 20, '-' * 20))
        print(' Number of input variables ::'.ljust(30) + '%d' %
              len(self.my_experiment.dists))
        print(' Maximal degree ::'.ljust(30) + '%d' %
              max([sum(i) for i in self.basis['multi-indices']]))
        print(
            ' Size of full basis ::'.ljust(30) +
            '%d' %
            self.my_experiment.n_samples)
        print(' Size of sparse basis ::'.ljust(30) + '%d' %
              len(self.basis['multi-indices']) + '\n')

        print(
            ' Full model evaluation ::'.ljust(30) +
            '%d' %
            self.my_experiment.size)
        print(' Leave-one-out error ::'.ljust(30) + '%s' % self.LOO + '\n')

        print(' Mean value ::'.ljust(30) + '%s' % m[0])
        print(' Std. deviation ::'.ljust(30) + '%s' % np.sqrt(var[0]))
        print(' skewness ::'.ljust(30) + '%s' % skew[0])
        print(' kurtosis ::'.ljust(30) + '%s' % kurt[0])

        print(' First-order Sobol indices ::'.ljust(30) + '%s' %
              np.around(self.sensitivity['Si'], decimals=4) +
              '\n')
        print(' Total Sobol indices ::'.ljust(30) + '%s' %
              np.around(self.sensitivity['STi'], decimals=4) +
              '\n')

        print('-' * 65)

        filename_res = (
            "full_pce_order_%d_%s" %
            (self.order,
             self.my_experiment.my_data.inputs['objective of interest']))

        with open(os.path.join(self.my_experiment.my_data.path_res,
                               filename_res), "w") as f:
            f.write('%25s %25f \n' % ('LOO', self.LOO))
            f.write(
                '%25s %25f \n' %
                ('sparse basis', len(
                    self.basis['multi-indices'])))
            f.write('%25s %25f \n' % ('mean', self.moments['mean'][0]))
            f.write(
                '%25s %25f \n' %
                ('std. dev.', np.sqrt(
                    self.moments['variance'][0])))

        with open(os.path.join(self.my_experiment.my_data.path_res,
                               filename_res + '_Sobol_indices'), "w") as f:
            f.write(
                '%30s %30s %30s  \n' %
                ('name',
                 'First-order Sobol indices',
                 'Total-order Sobol indices'))
            indices = np.argsort(self.sensitivity['STi'])[::-1]
            for i in indices:
                f.write(
                    '%30s %30f %30f  \n' %
                    (self.my_experiment.my_data.stoch_data['names'][i],
                     self.sensitivity['Si'][i],
                     self.sensitivity['STi'][i]))

    def uq_draw(self, size):
        """
        This module creates the probability density function
        and cumulative distribution function and writes
        the corresponding values to construct these values
        in the result files.

        Parameters
        ----------
        size : int
            number of samples to be created

        """

        self.my_experiment.create_samples(size=size)

        A = self.uq_calc_A(self.basis['multi-indices'])

        u = np.row_stack((self.coefficients[:, 0]))

        data = np.dot(A, u)

        self.ModelSampled = data

        density, bins = np.histogram(data, bins=100, density=1)
        centers = [(a + b) / 2 for a, b in zip(bins[::1], bins[1::1])]
        with open(os.path.join(self.my_experiment.my_data.path_res,
                               ("data_pdf_%s"
                                % (self.my_experiment.my_data.inputs[
                                    'objective of interest']))), "w") as f:

            f.write(
                '%25s %25s \n' %
                (self.my_experiment.my_data.inputs['objective of interest'],
                 'probability density'))
            for i, j in enumerate(centers):
                f.write('%25f %25f' % (j, density[i]))
                f.write('\n')

        # evaluate the cumulative
        cdf = np.cumsum(density * np.diff(bins))
        with open(os.path.join(self.my_experiment.my_data.path_res,
                               ("data_cdf_%s"
                                % (self.my_experiment.my_data.inputs[
                                    'objective of interest']))), "w") as f:

            f.write(
                '%25s %25s \n' %
                (self.my_experiment.my_data.inputs['objective of interest'],
                 'cumulative probability'))
            for i, j in enumerate(centers):
                f.write('%25f %25f' % (j, cdf[i]))
                f.write('\n')
