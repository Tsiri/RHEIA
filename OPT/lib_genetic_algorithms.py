# -*- coding: utf-8 -*-
"""
Created on Mon May 06 13:30:57 2019

@author::   tsiri
@subject::  library script for the Genetic Algorithms optimizers

"""

import multiprocessing as mp
import os
import sys
import numpy as np
import random
from copy import deepcopy
from deap import creator, base, tools


class NSGA2:

    def __init__(
            self,
            run_dict,
            case,
            config_obj,
            start_from_last_gen,
            file_add):
        self.run_dict = run_dict
        self.tc = case

        # Define configuration dictionary
        self.config = config_obj.config_opt_dict

        self.start_from_last_gen = start_from_last_gen
        self.file_add = file_add
        self.opt_res_dir = os.path.join(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..')),
            'RESULTS',
            self.tc.case,
            list(
                run_dict['objectives'].keys())[0],
            run_dict['results dir'],
        )

    def parse_status(self):
        """

        This function extracts the current generation number
        and the number of model evaluations performed.

        Returns
        -------
        ite : int
            The number of generations performed.
        evals : int
            The number of model evaluations performed.

        """

        stat_dir = os.path.join(self.opt_res_dir, 'STATUS')

        with open(stat_dir, 'rb') as f:

            # Read header
            f.readline()

            # Read mode and current number of evaluations
            line = f.readlines()[-1]

            ite, evals = int(line.split()[0]), int(line.split()[1])

        return ite, evals

    def write_status(self, msg):
        """

        A message is appended to the STATUS file.
        This message consists of the generation number
        and computational budget spent.

        Parameters
        ----------
        msg : str
            The message, to be appended to the STATUS file.

        """

        stat_dir = os.path.join(self.opt_res_dir, 'STATUS')

        f = open(stat_dir, 'a')
        f.write(msg + '\n')
        f.close()

    def append_points_to_file(self, nests, filename):
        """

        This function is used to append the population
        to the population file.

        Parameters
        ----------
        nests : list
            The samples of the population.
        filename : str
            The filename where the population
            should be appended.

        """

        file_dir = os.path.join(self.opt_res_dir, filename)

        with open(file_dir, 'a') as f:

            for n in nests:

                for item in n:

                    f.write('%.10f ' % item)

                f.write('\n')

            f.write('- \n')

    def append_fitness_to_file(self, fitness, filename):
        """

        This function is used to append the population
        fitness to the fitness file.

        Parameters
        ----------
        fitness : list
            The population fitness.
        filename : str
            The filename where the population fitness
            should be appended.

        """

        file_dir = os.path.join(self.opt_res_dir, filename)

        with open(file_dir, 'a') as f:

            for item in fitness:

                temp = ''
                for fit in item:

                    temp += '%.10f ' % fit

                f.write(temp + '\n')

            f.write('- \n')

    def make_individual(self, solution):
        """

        Generates the instance Individual.

        Parameters
        ----------
        solution : list
            The list [x1, x2,...,xn].

        Returns
        -------
        ind : instance
            The Individual instance.

        """

        ind = creator.Individual(solution)

        return ind

    def find_index_of_par_var(self, par_var_name, par_var_list):
        """

        Finds the index of the parameter or variable in the list.


        Parameters
        ----------
        par_var_name : str
            The name of the parameter or variable.
        par_var_list : list
            The design variables of the model.

        Returns
        -------
        par_var_idx : int
            The index of the parameter of variable in the list.

        """
        par_var_idx = par_var_list.index(par_var_name)

        return par_var_idx

    def define_samples_to_eval(self, pop):
        """

        Defines the set of samples considered for evaluation.
        Adds the list of parameters to the initial samples
        drawn form the desing space.

        Parameters
        ----------
        pop : list
            The set of samples (population form, only variables)
            considered for evaluation

        Returns
        -------
        samples_to_eval : list
            The set of samples considered for evaluation
            samples_to_eval = [[p11, p12, ..., p1N, x11, x12, ..., x1M],
                              [p21, p22, ..., p2N, x21, x22, ..., x2M]
                              ...
                              [pk1, pk2, ..., pkN, xk1, xk2, ..., xkM]
                              ]

           where N is the number of parameters
                 M is the number of variables
                 k is the number of samples

        unc_samples_to_eval : list
            The set of samples considered for uncertainty quantification

        """

        # Attach parameters list to the given set of samples

        temp = np.tile(list(self.tc.par_dict.values()), (len(pop), 1))
        for pop_sample in pop:
            if len(pop_sample) != len(self.tc.var_dict):
                raise NameError(
                    """A sample in the list of starting design samples
                       does not match the number of design variables.""")

        temp_samples = np.hstack((temp, np.array(pop))).tolist()
        if 'DET' in self.tc.opt_type:

            return temp_samples, []

        elif 'ROB' in self.tc.opt_type:

            uq_path = os.path.join(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        '..')),
                'UQ')

            sys.path.insert(0, uq_path)
            import lib_PCE as uq

            samples_to_eval, unc_samples_to_eval = [], []
            for sample in temp_samples:
                self.my_data = uq.Data(self.run_dict, self.tc)

                # acquire information on stochastic parameters
                self.my_data.read_stoch_parameters(
                    var_values=sample[-len(self.tc.var_dict):])

                # create experiment object
                self.objective_position = np.zeros(
                    len(self.run_dict['objective names']))
                for index, obj in enumerate(
                        self.run_dict['objective of interest']):
                    self.objective_position[index] = self.run_dict[
                        'objective names'].index(obj)
                self.my_experiment = uq.RandomExperiment(
                    self.my_data, self.objective_position)

                # create uniform/gaussian distributions and corresponding
                # orthogonal polynomials
                self.my_experiment.create_distributions()

                self.my_experiment.n_terms()

                n_samples_unc = self.my_experiment.n_samples

                self.my_experiment.x_prev = np.array([])
                self.my_experiment.y_prev = np.array([])
                # create a design of experiment for the remaining samples
                self.my_experiment.create_samples(
                    size=self.my_experiment.n_samples)

                unc_samples = self.my_experiment.X_u

                # Produce final set to evaluate
                temp_unc_samples = np.tile(sample, (n_samples_unc, 1))
                for j, elem in enumerate(self.tc.upar_dict.keys()):

                    unc_idx = self.find_index_of_par_var(
                        elem,
                        list(
                            self.tc.par_dict.keys()) +
                        list(
                            self.tc.var_dict.keys()))
                    temp_unc_samples[:, unc_idx] = self.my_experiment.X_u[:, j]

                # Convert ndarray to list
                temp_unc_samples = list(temp_unc_samples)

                # Check if solutions are outside the design space
                # for i, elem in enumerate(temp_unc_samples):

                #    temp_unc_samples[i] = moveInsideBoundaries(elem, tc)

                # Add samples to the overall database
                samples_to_eval += temp_unc_samples
                unc_samples_to_eval += list(unc_samples)
            return samples_to_eval, unc_samples_to_eval

    def evaluate_samples(self, samples):
        """

        Evaluation of the set of samples. If the number of jobs
        is larger than 1, parallel processing of the samples
        is considered.

        Parameters
        ----------
        samples : list
            The set of samples considered for evaluation.

        Returns
        -------
        fitness : list
            The calculated fitness for the set of samples.

        """
        if self.config['n jobs'] == 1:
            fitness = []
            for sample in samples:
                fitness.append(self.config['evaluate'](sample))

        else:
            pool = mp.Pool(processes=self.config['n jobs'])
            fitness = pool.map(self.config['evaluate'], enumerate(samples))
            pool.close()

        return fitness

    def assign_fitness_to_population(
            self,
            pop,
            fitness,
            unc_samples):
        """

        Assigns the calulated fitness to the corresponding samples. In case
        of robust optimization, a PCE is constructed first, based on the
        random samples and the corresponding model output. Thereafter, the
        mean and standard deviation are extracted for the quantities of
        interest

        Parameters
        ----------
        pop : list
            Set of samples in population format, considered for evaluation.
        fitness : list
            The calculated fitness.
        unc_samples : list
            The set of samples for uncertainty quantification.

        Returns
        -------
        pop : list
            The population, appended with the fitness values.

        """

        if 'DET' in self.tc.opt_type:

            for i, sample in enumerate(pop):
                pop[i].fitness.values = fitness[i]

        elif 'ROB' in self.tc.opt_type:

            uq_path = os.path.join(
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        '..')),
                'UQ')

            sys.path.insert(0, uq_path)
            import lib_PCE as uq

            # Attach parameters list to the sample
            temp = np.tile(list(self.tc.par_dict.values()), (len(pop), 1))
            temp_samples = np.hstack((temp, np.array(pop))).tolist()
            for i in range(len(temp)):

                tempFitness = fitness[:self.my_experiment.n_samples]

                del fitness[:self.my_experiment.n_samples]
                self.my_experiment.x_prev = np.array(
                    unc_samples[i *
                                self.my_experiment.n_samples:(i + 1) *
                                self.my_experiment.n_samples])

                # create a design of experiment for the remaining samples
                self.my_experiment.create_samples()

                temp_fitness = []

                for obj_position in self.objective_position:
                    if obj_position > len(tempFitness[0]) - 1:
                        raise TypeError(""" The objective "%s" falls out
                                            of the range of predefined
                                            quantities of interest.
                                            Only %i outputs are returned
                                            from the model""" %
                                        (self.run_dict[
                                            'objective names'][
                                            int(obj_position)],
                                            len(tempFitness[0])))
                    self.my_experiment.Y = np.array(tempFitness)[
                        :, int(obj_position)].reshape(-1, 1)

                    # create PCE object
                    my_pce = uq.PCE(self.my_experiment)

                    # evaluate the PCE
                    my_pce.uq_run()

                    # Assign statistics
                    if my_pce.moments['mean'][0] > 1e7:
                        temp_fitness.append(1e8)
                        temp_fitness.append(1e8)
                    else:
                        temp_fitness.append(my_pce.moments['mean'][0])
                        temp_fitness.append(
                            np.sqrt(my_pce.moments['variance'][0]))

                pop[i].fitness.values = temp_fitness

        return pop

    def read_doe(self, doe_dir):
        """


        Parameters
        ----------
        doe_dir : str
            The directory of the file with the population.

        Returns
        -------
        DOE : list
            The Design Of Experiments, i.e. the samples
            to be evaluated in the system model.

        """

        DOE = []
        d = open(doe_dir, 'rb')

        # Read header
        # d.readline()

        # Read DOE points
        for line in d:
            DOE.append([float(i) for i in line.split()])

            violate = False
            for index, elem in enumerate(DOE[-1]):
                if elem < self.tc.lb[index] or elem > self.tc.ub[index]:
                    violate = True

            if violate:
                raise TypeError("""Design sample %s violates the
                                   design variable bounds. """ % str(DOE[-1]))

        if len(DOE) != self.run_dict['population size']:
            raise TypeError(
                """The number of design samples in the starting population
                   file does not match with the population number provided
                   in the dictionary item "population number". """)
        return DOE

    def init_opt(self):
        '''

        Initialization of the results directory and writing of the
        column names in the STATUS file.

        '''

        if not os.path.exists(self.opt_res_dir):
            os.makedirs(self.opt_res_dir)

        # Initialize the config file
        msg = '%8s%8s' % ('ITER', 'EVALS') + '\n' + '%8s%8s' % (0, 0)
        self.write_status(msg)

    def eval_doe(self):
        """

        Evaluation of the Design Of Experiments (DoE). First, the DoE
        is read from the corresponding population file and stored in
        a list. Then, the design samples are evaluated in the model.
        When a population and fitness values are available from a previous
        optimization run, the method stores the final population and fitness
        values from this previous run. Finally, the population and
        corresponding fitness values are stored in the population and
        fitness file in the results directory, respectively.

        Returns
        -------
        current_pop : list
            The current population.

        """

        # DoE FOLDER PATH
        path_doe = os.path.join(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    '..')),
            'OPT',
            'INPUTS',
            self.tc.case,
            '%iD' %
            self.tc.n_dim)

        file_doe = 'DOE_n%i' % self.run_dict['population size']

        # READ DoE
        doe = self.read_doe(os.path.join(path_doe, file_doe))

        # EVALUATE DoE
        n_eval = 0
        current_pop = []
        for i, sol in enumerate(doe):

            current_pop.append(self.make_individual(sol))

        if not self.start_from_last_gen:

            # Create samples to evaluate
            individuals_to_eval, unc_samples = self.define_samples_to_eval(
                current_pop)

            # Evaluate samples
            fitnesses = self.evaluate_samples(individuals_to_eval)
            n_eval += len(individuals_to_eval)

            # Assign fitness to the initial population
            current_pop = self.assign_fitness_to_population(
                current_pop,
                fitnesses,
                unc_samples)

            # --------------------------- SAVE PARETO -------------------------

            if not self.run_dict['print results light'][0]:
                # UPDATE THE BEST PARTICLE FILES
                self.append_points_to_file(current_pop,
                                           'population')

                self.append_fitness_to_file(
                    [x.fitness.values for x in current_pop], 'fitness')
            # --------------------------------------------------------------------------

            # ----------------------- UPDATE THE STATUS FILE ------------------
            msg = '%8i%8i' % (1, n_eval)

            self.write_status(msg)
            # --------------------------------------------------------------------------

        else:

            with open(os.path.join(
                    self.opt_res_dir, 'fitness%s' % self.file_add)) as f:

                # Read DOE points
                output = []
                for line in f.readlines()[-len(doe) - 1:-1]:
                    output.append([float(i) for i in line.split()])
            for i, x in enumerate(current_pop):
                x.fitness.values = output[i]

        return current_pop

    def nsga2_1iter(self, current_pop):
        """

        Run one iteration of the NSGA-II optimizer.
        Based on the crossover and mutation probabilities, the
        offsprings are created and evaluated. Based on the fitness
        of these offsprings and of the current population,
        a new population is created.

        Parameters
        ----------
        current_pop : list
            The initial population.

        Returns
        -------
        new_pop : list
            The updated population.

        """

        # ----------------------- PRODUCE NEXT POPULATION ---------------------

        # CREATE OFFSPRING [CLONE CURRENT POPULATION]
        offspring = [deepcopy(ind) for ind in current_pop]

        # PERFORM CROSSOVER
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.config['cx prob']:

                # APPLY CROSSOVER OPERATOR - IN PLACE EDITING OF INDIVIDUALS
                tools.cxSimulatedBinaryBounded(
                    ind1, ind2, self.config['eta'], self.tc.lb, self.tc.ub)

                # SET FITNESS TO EMPTY TUPLE
                del ind1.fitness.values
                del ind2.fitness.values

        # PERFORM MUTATION
        for mutant in offspring:

            if random.random() < self.config['mut prob']:

                # APPLY MUTATION OPERATOR - IN PLACE EDITING OF INDIVIDUALS
                tools.mutPolynomialBounded(
                    mutant,
                    self.config['eta'],
                    self.tc.lb,
                    self.tc.ub,
                    self.config['mut prob'])

                # SET FITNESS TO EMPTY TUPLE
                del mutant.fitness.values

        # EVALUATE MODIFIED INDIVIDUALS
        n_eval = 0
        invalid_indices = []
        invalid_fit_individuals = []
        for i, ind in enumerate(offspring):

            if not ind.fitness.valid:

                invalid_indices.append(i)
                invalid_fit_individuals.append(ind)

        # Create samples to evaluate
        individuals_to_eval, unc_samples = self.define_samples_to_eval(
            invalid_fit_individuals)

        # Evaluate samples
        fitnesses = self.evaluate_samples(individuals_to_eval)
        n_eval += len(individuals_to_eval)

        # Assign fitness to the orginal samples list
        individuals_to_assign = self.assign_fitness_to_population(
            invalid_fit_individuals,
            fitnesses,
            unc_samples)

        # Construct offspring list
        for i, ind in zip(invalid_indices, individuals_to_assign):

            offspring[i] = deepcopy(ind)

        # SELECT NEXT POPULATION USING NSGA-II OPERATOR
        new_pop = tools.selNSGA2(current_pop + offspring, len(current_pop))

        # -------------------------------------------------------------------------

        # ------------------------- UPDATE STATUS FILE ------------------------
        ite, evals = self.parse_status()

        msg = '%8i%8i' % (ite + 1, evals + n_eval)
        self.write_status(msg)
        # -------------------------------------------------------------------------

        # --------------------------- SAVE PARETO -----------------------------

        # UPDATE THE BEST PARTICLE FILES
        if self.run_dict['print results light'][0]:
            if (ite + 1) % self.run_dict['print results light'][1] == 0:

                self.append_points_to_file(new_pop,
                                           'population_light')

                fitness_values = [x.fitness.values for x in new_pop]
                self.append_fitness_to_file(fitness_values,
                                            'fitness_light')

        else:
            self.append_points_to_file(new_pop,
                                       'population')

            fitness_values = [x.fitness.values for x in new_pop]
            self.append_fitness_to_file(fitness_values,
                                        'fitness')

        # --------------------------------------------------------------------------

        return new_pop

    def run_optimizer(self):
        """

        Run an optimization using the NSGA-II algorithm.

        """

        weigth = list(self.tc.obj.values())[0]
        creator.create('Fitness', base.Fitness, weights=weigth)
        creator.create('Individual', list, fitness=creator.Fitness)

        if not self.start_from_last_gen:
            self.init_opt()

        ite, init_evals = self.parse_status()

        temp_output_pop = self.eval_doe()

        # RUN GENERATIONS

        evals = 0
        while evals < self.config['stop crit']['threshold']:

            # PERFORM ONE ITERATION OF NSGA-II
            temp_output_pop = self.nsga2_1iter(
                temp_output_pop,
            )

            # UPDATE EVALUATIONS COUNTER
            ite, current_evals = self.parse_status()
            evals = current_evals - init_evals


def return_opt_methods():
    """

    Returns the names of the available optimizers.

    Returns
    -------
    name_list : list
        A list with the names of the available optimizers.

    """

    name_list = ['NSGA2']

    return name_list


def return_opt_obj(name):
    """

    Returns the optimizer fuction object

    Parameters
    ----------
    name : str
        Name of the optimizer.

    Returns
    -------
    object
        The optimizer function object.

    """

    switcher = {'NSGA2': NSGA2,
                }

    return switcher.get(name)


# CONFIGURATION FUCTIONS


def createConfig4_NSGA2(
        func,
        max_eval,
        n_pop=20,
        cx_prob=0.9,
        mut_prob=0.1,
        eta=0.2):
    """

    Creates the NSGA-II configuration dictionary.

    Parameters
    ----------
    func : TYPE
        The model evaluation method.
    max_eval : int
        The available computational budget.
    n_pop : int, optional
        The population size. The default is 20.
    cx_prob : float, optional
        The crossover probability. The default is 0.9.
    mut_prob : float, optional
        The mutation probability. The default is 0.1.
    eta : float, optional
        The eta parameter. The default is 0.2.

    Returns
    -------
    config : dict
        The NSGA-II configuration dictionary.

    """

    config = {'evaluate': func,

              'cx prob': cx_prob,
              'mut prob': mut_prob,
              'eta': eta,

              'stop crit': {'criterion': 'budget', 'threshold': max_eval},

              'population': n_pop
              }

    return config
