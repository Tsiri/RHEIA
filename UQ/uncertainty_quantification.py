# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:33:58 2016
@author: Diederik Coppitters
"""
import os, sys
import lib_PCE as uq
from pyDOE import lhs


path = os.path.split(
    os.path.dirname(
        os.path.abspath(__file__)))[0]
sys.path.insert(0, os.path.join(path,'CASES'))
from determine_stoch_des_space import load_case, check_dictionary

def get_design_variables(case):
    """
    This function loads the design variable names and bounds
    out of the :file:`design_space` file.

    Parameters
    ----------
    case : string
        The name of the case.


    Returns
    -------
    var_dict : dict
        A dictionary which includes the design variables and their bounds.

    """
    var_dict = {}

    try:
        path = os.path.dirname(os.path.abspath(__file__))
        path_to_read = os.path.join(
            os.path.abspath(
                os.path.join(
                    path,
                    os.pardir)),
            'CASES',
            case,
            'design_space')

        with open(path_to_read, 'r') as f:
            for line in f:
                tmp = line.split()
                if tmp[1] == 'var':
                    var_dict[tmp[0]] = [float(tmp[2]), float(tmp[3])]

    except BaseException:
        raise ValueError(
            'Missing file: "design_space". Design space cannot be created!')

    return var_dict


def set_design_samples(var_dict, n_samples):
    """
    Based on the design variable characteristics,
    a set of design samples is created through
    Latin Hypercube Sampling.

    Parameters
    ----------
    var_dict : dict
        A dictionary which includes the design variables and their bounds.
    n_samples : int
        The number of design samples to be created.

    Returns
    -------
    X : array
        The generated design samples.

    """

    X = lhs(len(var_dict), samples=n_samples)

    bounds = list(var_dict.values())

    for i, bound in enumerate(bounds):
        X[:, i] *= (bound[1] - bound[0])
        X[:, i] += bound[0]

    return X


def write_design_space(case, iteration, var_dict, x):
    """
    A new design space file is created. In this file,
    the model parameters are copied from the original file,
    i.e. file:`design_space`. The design variable names are copied,
    but the bounds are loaded out of the array `x`.
    This function is of interest when evaluating the LOO error
    or Sobol' indices for several design samples.

    Parameters
    ----------
    case : string
        The name of the case.
    iteration : int
        The index of the design sample
        out of the collection of generated design samples.
    var_dict : dict
        A dictionary which includes the design variables and their bounds.
    x : array
        The design sample out of the collection of generated design samples.

    """

    path = os.path.dirname(os.path.abspath(__file__))

    des_var_file = os.path.join(os.path.abspath(os.path.join(path, os.pardir)),
                                'CASES',
                                case,
                                'design_space',
                                )

    new_des_var_file = os.path.join(
        os.path.abspath(
            os.path.join(
                path,
                os.pardir)),
        'CASES',
        case,
        'design_space_%i' %
        iteration,
    )

    if not os.path.isfile(new_des_var_file):
        with open(des_var_file, 'r') as f:
            text = []
            for line in f.readlines():
                found = False
                tmp = line.split()
                for index, name in enumerate(list(var_dict.keys())):

                    if name == tmp[0]:
                        text.append('%s par %f \n' % (name, x[index]))
                        found = True
                if not found:
                    text.append(line)

        with open(new_des_var_file, 'w') as f:
            for item in text:
                f.write("%s" % item)


def run_uq(inputs, design_space='design_space'):
    """
    This function is the main to run uncertainty quantification.
    First, the input distributions are created,
    followed by the reading of previously evaluated samples.
    Thereafter, the new samples are created and evaluated
    in the system model when desired. Finally, the PCE is
    constructed, the statistical moments printed and
    the distributions generated (when desired) for the
    quantity of interest.

    Parameters
    ----------
    inputs : dict
        The dictionary with information on the uncertainty quantification.
    design_space : string, optional
        The design_space filename. The default is 'design_space'.

    """

    check_dictionary(inputs, uq=True)

    objective_position = inputs['objective names'].index(
        inputs['objective of interest'])

    tc, case_obj = load_case(inputs, design_space, uq=True)

    my_data = uq.Data(inputs, tc)

    # acquire information on stochastic parameters
    my_data.read_stoch_parameters()

    # create result csv file to capture all input-output of the samples
    my_data.create_samples_file()

    # create experiment object
    my_experiment = uq.RandomExperiment(my_data, objective_position)

    # create uniform/gaussian distributions and corresponding orthogonal
    # polynomials
    my_experiment.create_distributions()

    my_experiment.n_terms()

    my_experiment.read_previous_samples(inputs['create only samples'])

    # create a design of experiment for the remaining samples
    my_experiment.create_samples(
        size=my_experiment.n_samples - len(my_experiment.x_prev))

    my_experiment.create_only_samples(inputs['create only samples'])

    if not inputs['create only samples']:

        if my_experiment.n_samples > len(my_experiment.x_prev):
            my_experiment.evaluate(case_obj)
        elif my_experiment.n_samples == len(my_experiment.x_prev):
            my_experiment.Y = my_experiment.y_prev
        else:
            my_experiment.Y = my_experiment.y_prev[:my_experiment.n_samples]

        # create PCE object
        my_pce = uq.PCE(my_experiment)

        # evaluate the PCE
        my_pce.uq_run()

        my_pce.uq_calc_LOO()
        my_pce.uq_calc_Sobol()

        # extract results
        my_pce.uq_print()

        if inputs['draw pdf cdf'][0]:
            my_pce.uq_draw(int(inputs['draw pdf cdf'][1]))
