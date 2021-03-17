# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 12:55:25 2019

@author::   tsiri
@subject::  Run ERGO framework

"""

import collections
import os
import sys
import imp
import lib_config
import numpy as np
from pyDOE import lhs
from shutil import copyfile
#from CASES.determine_stoch_des_space import load_case, check_dictionary

# OPTIMIZERS


def parse_available_opt():
    """

    Parse all available optimizers. In this version,
    only NSGA-II is available.

    Returns
    -------
    methods : list of tuples
        The available optimizers with the corresponding classes.

    """

    opt_dir = os.path.join(
        os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0],
        'OPT')
    fs = [f for f in os.listdir(opt_dir) if f.endswith('algorithms.py')]
    methods = []
    for f in fs:
        obj = imp.load_source(f.split('.')[0], os.path.join(opt_dir, f))
        tmp = obj.return_opt_methods()
        for method in tmp:
            methods.append((method, obj))
    return methods


def load_optimizer(optimizer):
    """

    Load selected optimizer.

    Parameters
    ----------
    optimizer : string
        The name of the optimizer (currently equals to 'NSGA2').

    Returns
    -------
    opt_obj : function object
        The optimization function.

    """

    opt_list = parse_available_opt()
    optimizers = [elem[0] for elem in opt_list]

    # Check if optimizer exists and load optimization module
    if optimizer not in optimizers:
        raise KeyError('Optimizer is not available!')
    else:
        for opt in opt_list:
            if optimizer == opt[0]:
                opt_obj = opt[1].return_opt_obj(optimizer)

    return opt_obj

# CONFIGURATION FUNCTIONS


def load_configuration(run_dict, tc, optimizer):
    """

    Load default configuration dictionary.
    This dictionary includes information on the
    characterization of the optimization, including
    the evaluation function name, the crossover and
    mutation probability, the eta parameter,
    the stopping criterion, threshold and the number
    of parallel processes.

    Parameters
    ----------
    run_dict : dict
        The dictionary with user-defined values for
        the characterization of the optimization.
    tc : object
        The class object of the case.
    optimizer : string
        The name of the optimizer (currently equals to 'NSGA2').

    Returns
    -------
    conf_obj : dict
        The configuration dictionary.

    """

    opt_config_dir = os.path.join(
        os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0],
        'OPT')
    sys.path.insert(0, opt_config_dir)
    conf_obj_name = lib_config.get_config_obj(optimizer)
    conf_obj = conf_obj_name(tc, run_dict)

    return conf_obj

# STARTING SAMPLES


def scale_samples_to_design_space(nondim_doe, tc):
    """
    Scales the starting sample to the given design space.

    Parameters
    ----------
    nondim_doe : ndarray
        The non-dimensionalized design of experiment.
    tc : object
        The class object of the case.

    Returns
    -------
    dim_doe : ndarray
        The design of experiment scaled up to the design space.

    """

    dim_doe = np.zeros(nondim_doe.shape)
    for j in range(tc.n_dim):
        dim_doe[:, j] = (tc.ub[j] - tc.lb[j]) * nondim_doe[:, j] + tc.lb[j]
    return dim_doe


def write_starting_samples(doe, filename):
    """

    Writes the starting samples to file.

    Parameters
    ----------
    doe : list
        The set of starting samples.
    filename : string
        The path to the filename for the starting samples.

    """

    with open(filename, 'w') as f:
        for x in doe:
            if not isinstance(x, list):
                x = x.tolist()
            for item in x:
                f.write('%.8f ' % item)
            f.write('\n')


def create_starting_samples(
        run_dict,
        tc,
        conf_obj,
        start_from_last_gen,
        file_add):
    """

    Load the starting samples of the optimization run.

    Parameters
    ----------
    run_dict : dict
        The dictionary with user-defined values for
        the characterization of the optimization.
    tc : object
        The class object of the case.
    conf_obj : object
        The optimizer configuration object.
    start_from_last_gen : bool
        Boolean that determines if the starting samples
        start from a previously generated population.
    file_add : string
        Addition to the filename in case of light printing.

    """

    # Define folder to put starting samples file
    doe_path = os.path.join(
        os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0],
        'OPT',
        'INPUTS',
        tc.case.upper(),
        '%iD' %
        tc.n_dim)
    # Check if folder exists in INPUTS
    # If not create one
    if not os.path.exists(doe_path):
        os.makedirs(doe_path)

    # Create the doe set of samples
    doe_filename = os.path.join(
        doe_path, 'DOE_n%i' % run_dict['population size'])

    if not start_from_last_gen:
        if 'AUTO' in run_dict['x0'][0]:
            if 'RANDOM' in run_dict['x0'][1]:
                ddoe = np.random.random(
                    (run_dict['population size'], tc.n_dim))
            if 'LHS' in run_dict['x0'][1]:
                ddoe = lhs(tc.n_dim, samples=run_dict['population size'])

            # Scale starting samples
            doe = scale_samples_to_design_space(ddoe, tc)

            # Write starting samples to file
            write_starting_samples(doe, doe_filename)

        else:
            doe_custom_file = os.path.join(
                os.path.split(
                    os.path.dirname(
                        os.path.abspath(__file__)))[0],
                'CASES',
                tc.case,
                run_dict['x0'][1])

            if not os.path.isfile(doe_custom_file):
                raise NameError(
                    """The initial population file %s
                    is not found in the case folder.""" %
                    os.path.basename(doe_custom_file))

            copyfile(doe_custom_file, doe_filename)

    else:

        doe_custom_file = os.path.join(
            os.path.split(
                os.path.dirname(
                    os.path.abspath(__file__)))[0],
            'RESULTS',
            tc.case.upper(),
            list(
                run_dict['objectives'].keys())[0],
            run_dict['results dir'],
            'population%s' %
            file_add)

        d = open(doe_custom_file, 'r')

        # Read DOE points
        doe = []
        for line in d.readlines()[-run_dict['population size'] - 1:-1]:
            doe.append([float(i) for i in line.split()])

        write_starting_samples(doe, doe_filename)
        d.close()


def run_opt(run_dict, design_space='design_space'):
    """
    This function runs the optimization pipeline.
    First, the case, optimizer and configuration
    are loaded. Thereafter, the starting samples
    are created, the specific optimization class
    instantiated and the :py:meth:`run_optimizer`
    method is called.

    Parameters
    ----------
    run_dict : dict
        The dictionary with user-defined values for
        the characterization of the optimization.
    design_space : string, optional
        The design_space filename. The default is 'design_space'.

    """

    check_dictionary(run_dict)

    # Load test case
    tc = load_case(run_dict, design_space)

    if run_dict['print results light'][0]:
        file_add = '_light'
    else:
        file_add = ''

    pop_file = os.path.join(
        os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0],
        'RESULTS',
        tc.case.upper(),
        list(
            run_dict['objectives'].keys())[0],
        run_dict['results dir'],
        'population%s' %
        file_add)

    fitness_file = os.path.join(
        os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0],
        'RESULTS',
        tc.case.upper(),
        list(
            run_dict['objectives'].keys())[0],
        run_dict['results dir'],
        'fitness%s' %
        file_add)

    start_from_last_gen = False
    if os.path.isfile(fitness_file) and os.path.isfile(pop_file):

        start_from_last_gen = True

    # Load optimizer
    optfunc = load_optimizer('NSGA2')

    # Load optimizer configuration
    conf_obj = load_configuration(run_dict, tc, 'NSGA2')

    # Define starting samples
    create_starting_samples(
        run_dict,
        tc,
        conf_obj,
        start_from_last_gen,
        file_add)

    # Run the optimizer
    res = optfunc(run_dict,
                  tc,
                  conf_obj,
                  start_from_last_gen,
                  file_add
                  )

    res.run_optimizer()
