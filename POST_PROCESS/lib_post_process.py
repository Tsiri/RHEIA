import os
import numpy as np


class post_process:

    def __init__(self, case):
        self.case = case
        path = os.path.dirname(os.path.abspath(__file__))
        self.path_start = os.path.abspath(os.path.join(path, os.pardir))
        self.path = os.path.join(self.path_start,
                                 'RESULTS',
                                 case
                                 )


class post_process_opt(post_process):

    def __init__(self, my_post_process, LIGHT, eval_type):
        self.my_post_process = my_post_process
        self.LIGHT = LIGHT
        self.n_pop = 0
        self.x_lines = []
        self.y_lines = []
        self.result_path = os.path.join(self.my_post_process.path,
                                        eval_type,
                                        )
        self.fitness_file = ''
        self.population_file = ''

    def determine_pop_gen(self):
        """

        Determines the number of design samples in the population
        and the number of generations performed.

        """

        self.n_gen = 0
        with open(self.fitness_file, 'r') as f1:
            self.y_lines = f1.readlines()
            for string in self.y_lines:
                if '-' in string:
                    self.n_gen += 1
            self.n_pop = int(len(self.y_lines) / self.n_gen - 1)

        print('n_gen: %i' % self.n_gen)

    def get_fitness_values(self, gen):
        """

        Returns the fitness values for the population
        generated in the specified generation.

        Parameters
        ----------
        gen : int
            The generation of interest.

        Returns
        -------
        Y : ndarray
            The fitness values for the population of interest.

        """

        Y = np.zeros((len(self.y_lines[0].split()), self.n_pop))
        for index, l in enumerate(
                self.y_lines[(gen - 1) * self.n_pop - 1 + gen:
                             gen * self.n_pop - 1 + gen]):
            Y[:, index] = [float(i) for i in l.split()]

        for r in range(len(Y)):
            v = r*2.
        return Y

    def get_population_values(self, gen):
        """
        Returns the fitness values for the population
        generated in the specified generation.

        Parameters
        ----------
        gen : int
            The generation of interest.

        Returns
        -------
        X : ndarray
            The population of interest.

        """

        with open(self.population_file, 'r') as f2:
            self.x_lines = f2.readlines()

        X = np.zeros((len(self.x_lines[0].split()), self.n_pop))
        for index, l in enumerate(
                self.x_lines[(gen - 1) * self.n_pop - 1 + gen:
                             gen * self.n_pop - 1 + gen]):
            X[:, index] = [float(i) for i in l.split()]

        return X

    def sorted_result_file(self, y, x):
        """

        Generates the files that include the sorted population
        and fitness files. The population and fitness are sorted
        based on the first objective.

        Parameters
        ----------
        y : ndarray
            The fitness values.
        x : ndarray
            The set of design samples.

        """

        with open(self.fitness_file + "_final_sorted", 'w') as f:
            for sample in y:
                for value in sample:
                    f.write('%f, ' % value)
                f.write('\n')

        with open(self.population_file + "_final_sorted", 'w') as f:
            for sample in x:
                for value in sample:
                    f.write('%f, ' % value)
                f.write('\n')

    def get_fitness_population(self, result_dir, gen=0):
        """

        Returns the population and corresponding fitness values
        for the generation of interest.

        Parameters
        ----------
        result_dir : str
            The directory were the results are stored.
        gen : int, optional
            The generation of interest. The default is 0,
            i.e. the final generation.

        Returns
        -------
        y : ndarray
            The fitness values.
        x : ndarray
            The set of design samples.

        """

        self.fitness_file = os.path.join(self.result_path,
                                         result_dir,
                                         'fitness',
                                         )

        self.population_file = os.path.join(self.result_path,
                                            result_dir,
                                            'population',
                                            )

        if self.LIGHT:
            self.fitness_file += '_light'
            self.population_file += '_light'

        self.determine_pop_gen()

        Y = self.get_fitness_values(gen)

        X = self.get_population_values(gen)

        a, b = X.shape
        c, d = Y.shape
        indices = np.argsort(Y[0])

        x = np.zeros((a, b))
        y = np.zeros((c, d))
        for j, k in enumerate(indices):
            for L, y_in in enumerate(y):
                y[L][j] = Y[L][k]
            for m, x_in in enumerate(x):
                x[m][j] = X[m][k]

        self.sorted_result_file(y.transpose(), x.transpose())

        return y, x


class post_process_uq(post_process):

    def __init__(self, my_post_process, pol_order):
        self.my_post_process = my_post_process
        self.pol_order = pol_order

    def read_distr_file(self, distr_file):
        """

        Reads the file with information on the
        cumulative density function or probability
        density function.

        Parameters
        ----------
        distr_file : str
            The name of the distribution file.

        Returns
        -------
        x : ndarray
            The values from the PDF or CDF on the
            quantity of interest.
        y : ndarray
            The probability density (for the PDF)
            or cumulative probability (for the CDF).

        """

        with open(distr_file, 'r') as f:
            lines = f.readlines()
            x = np.ones(len(lines) - 1)
            y = np.ones(len(lines) - 1)
            for index, line in enumerate(lines[1:]):
                tmp = line.split()
                x[index] = float(tmp[0])
                y[index] = float(tmp[1])

        return x, y

    def get_sobol(self, result_dir, objective):
        """

        Retrieves the information on the Sobol' indices from
        the corresponding file in the result directory.

        Parameters
        ----------
        result_dir : str
            The result directory.
        objective : str
            The name of the quantity of interest.

        Returns
        -------
        names : list
            The names of the stochastic parameters.
        sobol : list
            The total' order Sobol' indices.

        """

        sobol_file = os.path.join(self.my_post_process.path,
                                  'UQ',
                                  '%s' % result_dir,
                                  'full_pce_order_%i_%s_Sobol_indices' % (
                                      self.pol_order, objective)
                                  )
        res_tmp = []
        with open(sobol_file, 'r') as f:
            for line in f.readlines()[1:]:
                res_tmp.append([i for i in line.split()])
            names = [row[0] for row in res_tmp]
            sobol = [float(row[2]) for row in res_tmp]

        return names, sobol

    def get_pdf(self, result_dir, objective):
        """

        Retrieves the points that define the probability density function.

        Parameters
        ----------
        result_dir : str
            The result directory.
        objective : str
            The name of the quantity of interest.

        Returns
        -------
        x : ndarray
            The values from the PDF on the
            quantity of interest.
        y : ndarray
            The probability density.

        """

        pdf_file = os.path.join(self.my_post_process.path,
                                'UQ',
                                '%s' % result_dir,
                                'data_pdf_%s' % objective
                                )

        x, y = self.read_distr_file(pdf_file)

        return x, y

    def get_cdf(self, result_dir, objective):
        """

        Retrieves the points that define the cumulative density function.

        Parameters
        ----------
        result_dir : str
            The result directory.
        objective : str
            The name of the quantity of interest.

        Returns
        -------
        x : ndarray
            The values from the CDF on the
            quantity of interest.
        y : ndarray
            The cumulative probability.

        """

        cdf_file = os.path.join(self.my_post_process.path,
                                'UQ',
                                '%s' % result_dir,
                                'data_cdf_%s' % objective
                                )

        x, y = self.read_distr_file(cdf_file)

        return x, y

    def get_LOO(self, result_dir, objective):
        """

        Reads the Leave-One-Out error from the corresponding
        file in the result directory.

        Parameters
        ----------
        result_dir : str
            The result directory.
        objective : str
            The name of the quantity of interest.

        Returns
        -------
        loo : float
            The Leave-One-Out error.

        """

        loo_file = os.path.join(self.my_post_process.path,
                                'UQ',
                                '%s' % (result_dir),
                                'full_pce_order_%i_%s' % (
                                    self.pol_order, objective)
                                )
        with open(loo_file, 'r') as f:
            line = f.readlines()[0]
            loo = float(line.split()[1])

        return loo

    def get_max_sobol(self, result_dirs, objective, threshold=0.05):
        """

        This method gathers the Sobol' indices for each stochastic parameter
        for each sample. The highest Sobol' index for each stochastic
        parameter is compared with the threshold value. If the highest
        Sobol' index is higher than the threshold, the name of the
        stochastic parameter is printed under 'significant Sobol indices'.
        If not, it is printed under 'negligible Sobol indices'.

        Parameters
        ----------
        result_dir : list
            The result directories.
        objective : str
            The name of the quantity of interest.
        threshold : float, optional
            The threshold that determines if a Sobol' index
            is considered significant. The default is 0.05.

        """

        n_samples = len(result_dirs)
        res_dict = [{}] * n_samples
        for index, result_dir in enumerate(result_dirs):
            names, sobol = self.get_sobol(result_dir, objective)
            res_dict[index] = dict(zip(names, sobol))

        max_dict = dict()
        for name in names:
            sobol_res = np.zeros(n_samples)
            for j, dic in enumerate(res_dict):
                sobol_res[j] = dic[name]
            max_dict[name] = max(sobol_res)

        print('significant Sobol indices:')
        for k in names:
            if max_dict[k] >= threshold:
                print('%s: %4f' % (k, max_dict[k]))

        print('\nnegligible Sobol indices:')
        for k in names:
            if max_dict[k] < threshold:
                print('%s: %4f' % (k, max_dict[k]))
