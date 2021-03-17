# -*- coding: utf-8 -*-
"""
Created on Wed Mar 06 02:00:49 2019

@author::   tsiri
@subject::  configuration class object for optimizers


"""


def get_config_obj(optimizer):
    '''

    Returns the configuration class object for the optimizer.

    INPUTS::
    ========

    optimizer : name of the optimizer - string format

    '''

    switcher = {'NSGA2': config_NSGA2,
                }

    return switcher.get(optimizer, NameError('Optimizer is not supported!'))

# BASE CLASSES


class config_NSGA2:
    '''

    Configure NSGA2 optimization scheme

    '''

    def __init__(self, tc, run_dict):

        self.stop_criterion = run_dict['stop'][0]
        self.threshold = run_dict['stop'][1]
        self.n_jobs = run_dict['n jobs']

        # Initiate opt-specific parameters
        self.cx_prob = run_dict['cx prob']
        self.mut_prob = run_dict['mut prob']
        self.eta = run_dict['eta']

        # Create configuration dictionary
        self.create_config_dict(tc)

    def create_config_dict(self, tc):
        '''

        Create the configuration dictionary for PSO2007 optimizer

        '''
        self.config_opt_dict = {
            'evaluate': tc.evaluate,
            'cx prob': self.cx_prob,
            'mut prob': self.mut_prob,
            'eta': self.eta,
            'stop crit': {
                'criterion': self.stop_criterion,
                'threshold': self.threshold},
            'n jobs': self.n_jobs}
