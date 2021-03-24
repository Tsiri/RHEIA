import os
import sys
import collections
import importlib.util

def check_dictionary(run_dict, uq=False):
    """
    
    This function evaluates if the items in the input dictionary are
    properly characterized.

    Parameters
    ----------
    run_dict : dict
        The input dictionary.
    uq : bool, optional
        Boolean that mentions if uncertainty quantification is considered.
        The default is False.

    """

    rob = False

    if not isinstance(run_dict, collections.Mapping):
        raise TypeError('The input dictionary should be a dictionary.')

    requirements = ['case',
                    'n jobs',
                    'results dir',
                    ]

    if 'n jobs' not in run_dict:
        run_dict['n jobs'] = 1

    for key in requirements:
        try:
            run_dict[key]
        except BaseException:
            raise KeyError(
                '"%s" is missing in the input dictionary.' %
                key)

    if not isinstance(run_dict['n jobs'], int):
        raise TypeError(
            'The value of the key "n jobs" should be a positive integer.')

    if not isinstance(run_dict['results dir'], str):
        raise TypeError(
            'The value of the key "results dir" should be a string.')

    if not uq:
        requirements += ['objectives',
                         'stop',
                         'population size',
                         'x0',
                         'cx prob',
                         'mut prob',
                         'eta',
                         'print results light',
                         ]

        if 'x0' not in run_dict:
            run_dict['x0'] = ('AUTO', 'LHS')
        if 'cx prob' not in run_dict:
            run_dict['cx prob'] = 0.9
        if 'mut prob' not in run_dict:
            run_dict['mut prob'] = 0.1
        if 'eta' not in run_dict:
            run_dict['eta'] = 0.2
        if 'print results light' not in run_dict:
            run_dict['print results light'] = [False]

        for key in requirements[3:]:
            try:
                run_dict[key]
            except BaseException:
                raise KeyError(
                    '"%s" is missing in the input dictionary.' %
                    key)

        if not isinstance(run_dict['objectives'], collections.Mapping):
            raise TypeError(
                'The value of the key "objectives" should be a dictionary.')

        if not list(
                run_dict['objectives'].keys())[0] == 'DET' and not list(
                run_dict['objectives'].keys())[0] == 'ROB':
            raise ValueError(
                """Please select "DET" or "ROB" as a key
                for the "objectives" value.""")
        elif not isinstance(list(run_dict['objectives'].values())[0], tuple):
            raise TypeError(
                """The value of the key "DET" or "ROB" should be a tuple
                with the weights for the objectives
                (1 for maximization, -1 for minimization).""")
        elif (not all(isinstance(weight, int) for weight in
                      list(run_dict['objectives'].values())[0])):
            raise TypeError('The weights should be equal to 1 or -1.')
        elif (not all(abs(weight) == 1 for weight in
                      list(run_dict['objectives'].values())[0])):
            raise ValueError('The weights should be equal to 1 or -1.')

        if not isinstance(run_dict['stop'], tuple):
            raise TypeError(
                """The value of the key "stop" should be a tuple
                   with two elements.""")

        if not run_dict['stop'][0] == 'BUDGET':
            raise ValueError(
                """The first element in the tuple related to the key "stop"
                   should be equal to "BUDGET".""")

        if not isinstance(run_dict['stop'][1], int):
            raise TypeError(
                """The second element in the tuple related to the key "stop"
                   should be a positive integer.""")

        if not isinstance(run_dict['population size'], int):
            raise TypeError(
                """The value of the key "population number"
                   should be a positive integer.""")

        if not isinstance(run_dict['x0'], tuple):
            raise TypeError(
                """The value of the key "x0" should be a tuple
                   with two elements.""")

        if (not run_dict['x0'][0] == 'AUTO' and not
                run_dict['x0'][0] == 'CUSTOM'):
            raise ValueError(
                """The first element in the tuple related to the key "x0"
                   should be equal to "AUTO" or "CUSTOM".""")
        elif (run_dict['x0'][0] == 'AUTO' and not
              run_dict['x0'][1] == 'RANDOM' and not
              run_dict['x0'][1] == 'LHS'):
            raise ValueError(
                """When selecting "AUTO", the second element in the tuple
                   related to the key "x0" should be equal
                   to "RANDOM" or "LHS".""")

        if (not run_dict['x0'][0] == 'AUTO' and not
                run_dict['x0'][0] == 'CUSTOM'):
            raise ValueError(
                """The first element in the tuple related to the key "x0"
                   should be equal to "AUTO" or "CUSTOM".""")

        if not isinstance(run_dict['cx prob'], float):
            raise TypeError(
                """The value of the key "cx prob" should be a positive float,
                   lower or equal to 1.""")
        elif run_dict['cx prob'] > 1.:
            raise ValueError(
                """The value of the key "cx prob" should be a positive float,
                   lower or equal to 1.""")

        if not isinstance(run_dict['mut prob'], float):
            raise TypeError(
                """The value of the key "mut prob" should be a positive float,
                   lower or equal to 1.""")
        elif run_dict['mut prob'] > 1.:
            raise ValueError(
                """The value of the key "mut prob" should be a positive float,
                   lower or equal to 1.""")

        if not isinstance(run_dict['eta'], float):
            raise TypeError(
                """The value of the key "eta" should be a positive float,
                   lower or equal to 1.""")
        elif run_dict['eta'] > 1.:
            raise ValueError(
                """The value of the key "eta" should be a positive float,
                   lower or equal to 1.""")

        if not isinstance(run_dict['print results light'], list):
            raise TypeError(
                'The value of the key "print results light" should be a list.')

        if not isinstance(run_dict['print results light'][0], bool):
            raise TypeError(
                """The value of the key "print results light"
                   should be a True or False.""")

        if run_dict['print results light'][0] and not len(
                run_dict['print results light']) == 2:
            raise ValueError(
                """When light result printing is desired,
                provide the step for printing a generation in the form
                of a positive integer as the second list item""")

            if not isinstance(run_dict['print results light'][1], int):
                raise TypeError(
                    """When light result printing is desired,
                       provide the step for printing a generation in the form
                       of a positive integer as the second list item""")

        if list(run_dict['objectives'].keys())[0] == 'ROB':
            rob = True

    else:

        requirements += ['create only samples',
                         'objective of interest',
                         'draw pdf cdf',
                         ]

        if 'create only samples' not in run_dict:
            run_dict['create only samples'] = False

        if 'draw pdf cdf' not in run_dict:
            run_dict['draw pdf cdf'] = [False]

        for key in requirements[-3:]:
            try:
                run_dict[key]
            except BaseException:
                raise KeyError(
                    '"%s" is missing in the input dictionary.' %
                    key)

        if not isinstance(run_dict['create only samples'], bool):
            raise TypeError(
                """The value of the key "create only samples"
                should be a True or False.""")

        if not isinstance(run_dict['draw pdf cdf'], list):
            raise TypeError(
                'The value of the key "draw pdf cdf" should be a list.')

        if not isinstance(run_dict['draw pdf cdf'][0], bool):
            raise TypeError(
                """The value of the first element in the list "draw pdf cdf"
                should be a True or False.""")

        if run_dict['draw pdf cdf'][0] and not len(
                run_dict['draw pdf cdf']) == 2:
            raise ValueError(
                """When creating the distribution is desired,
                   provide the number of samples in the form of
                   a positive integer as the second list item""")

            if not isinstance(run_dict['draw pdf cdf'][1], int):
                raise TypeError(
                    """When creating the distribution is desired,
                    provide the number of samples in the form of
                    a positive integer as the second list item""")

        if not any(name == run_dict['objective of interest']
                   for name in run_dict['objective names']):
            raise ValueError(
                """The value of the key "objective of interest"
                   should be a string equal to any element
                   in the list with key "objective names".""")

    if rob or uq:

        requirements += ['pol order',
                         'sampling method',
                         'objective names',
                         ]

        if 'sampling method' not in run_dict:
            run_dict['sampling method'] = 'SOBOL'

        for key in requirements[-3:]:
            try:
                run_dict[key]
            except BaseException:
                raise KeyError(
                    '"%s" is missing in the input dictionary.' %
                    key)

        if not isinstance(run_dict['pol order'], int):
            raise TypeError(
                """The value of the key "pol order" should be a
                   positive integer.""")

        if (not run_dict['sampling method'] == 'RANDOM' and not
                run_dict['sampling method'] == 'SOBOL'):
            raise ValueError(
                """The value of the key "sampling method"
                   should be equal to "RANDOM" or "SOBOL".""")

        if not isinstance(run_dict['objective names'], list):
            raise TypeError(
                """The value of the key "objective names"
                   should be a list with the names of the
                   quantities of interest as elements in string format.""")

    if rob:
        requirements += ['objective of interest']

        try:
            run_dict[requirements[-1]]
        except BaseException:
            raise KeyError(
                '"%s" is missing in the input dictionary.' %
                key)

        if not isinstance(run_dict['objective of interest'], list):
            raise TypeError(
                """The value of the key "objective of interest"
                   should be a list with the names of the
                   quantities of interest as elements in string format.""")

        if not isinstance(run_dict['objective names'], list):
            raise TypeError(
                """The value of the key "objective names"
                   should be a list with the names of the
                   quantities of interest as elements in string format.""")
        else:
            for name_qoi in run_dict['objective of interest']:
                if not any(name == name_qoi
                           for name in run_dict['objective names']):
                    raise ValueError(
                        """The value of the key "objective of interest"
                           should be a list with strings equal to any element
                           in the list with key "objective names".""")

        if not len(list(run_dict['objectives'].values())[
                   0]) == 2 * len(run_dict['objective of interest']):
            raise ValueError(
                """The number of objectives of interest should be equal to
                           two times the number of weigths. Note that the
                           weigths with uneven indices relate to the standard
                           deviation of the quantity of interest and should
                           therefore be equal to -1 (minimization).""")

    for elem in list(run_dict.keys()):
        if requirements.count(elem) == 0:
            raise ValueError("""" "%s" does not belong in the dictionary.
                                 Try renaming it, such that it corresponds to
                                 an expected dictionary item, or consider
                                 removing it. """ % elem)


def load_case(run_dict, design_space, uq = False):
    """
    For the selected case, the design variables and model parameters
    are loaded based on information from :file:`design_space`.
    In addition, the class object for the selected case is instantiated.

    Parameters
    ----------
    run_dict : dict
        The dictionary with information on the uncertainty quantification.
    design_space : string
        The design_space filename.


    Returns
    -------
    tc_obj : object
        The class object of the case.

    """

    # Define TC directory
    tc_path = os.path.join(
        os.path.split(
            os.path.dirname(
                os.path.abspath(__file__)))[0],
        'CASES')
    # Enlist folders in TC directory
    # This will grouped in a function that checks the consistency of the
    # test case folder
    dir_list = [
        item for item in os.listdir(tc_path) if os.path.isdir(
            os.path.join(
                tc_path,
                item))]

    if uq:
        eval_type = 'UQ'
    elif 'ROB' in run_dict['objectives'].keys():
        eval_type = 'ROB'
    elif 'DET' in run_dict['objectives'].keys():
        eval_type = 'DET'
    else:
        raise ValueError('Error in name of optimization type!')

    space = stochastic_design_space(
        eval_type, run_dict['case'], design_space)

    if not uq:
        space.attach_objectives(run_dict['objectives'])

    if not run_dict['case'] in dir_list:
        raise ValueError(
            'Missing folder: %s folder is not found!' %
            run_dict['case'])

    elif not os.path.isfile(os.path.join(tc_path, run_dict['case'],
                                         'case_description.py')):
        raise ValueError('Missing file: case_description.py not found!')

    else:
        try:
            sys.path.insert(0, os.path.join(tc_path, run_dict['case']))
            import case_description
            tc_obj = case_description.CASE(space)
            
        except BaseException:
            raise ValueError('Error in test case class object!')

    return space, tc_obj


class stochastic_design_space(object):

    def __init__(self, opt_type, case, design_space):
        """
        Class which creates an object which characterizes the stochastic
        design space for the system model evaluation.

        Parameters
        ----------
        opt_type : string
           defines the type run (string format)
           available types = ['DET', 'ROB', 'UQ']

        design_space : string
            the design_space file name considered for evaluation

        Returns
        -------
        None

        """

        self.case = case
        self.design_space = design_space
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.case_path = os.path.join(self.path, case)
        self.opt_type = opt_type
        self.lb = []
        self.ub = []
        self.par_dict = {}
        self.var_dict = {}
        self.upar_dict = {}
        self.obj = None

        self.read_design_space()
        self.read_stochastic_space()

    def read_design_space(self):

        path_to_read = os.path.join(self.case_path, self.design_space)

        if not os.path.isfile(path_to_read):
            raise ValueError(
                """Missing file: "design_space" or the name of the case does
                   not exist. Make sure that the name of the case is equal to
                   the name of the folder in CASES.""")

        with open(path_to_read, 'r') as f:
            for line in f:
                tmp = line.split()
                if tmp[1] == 'par':
                    if len(tmp) != 3:
                        raise IndexError(
                            """ Wrong characterization of the parameter %s.
                                Is it supposed to be a variable?
                                Change "par" into "var".""" % tmp[0])

                    self.par_dict[tmp[0]] = float(tmp[2])

                elif tmp[1] == 'var':
                    if len(tmp) != 4:
                        raise IndexError(
                            """ Wrong characterization of the design variable %s.
                                Is it supposed to be a parameter?
                                Change "var" into "par".""" %
                            tmp[0])
                    self.var_dict[tmp[0]] = [float(tmp[2]), float(tmp[3])]
                    self.lb.append(float(tmp[2]))
                    self.ub.append(float(tmp[3]))
                    if float(tmp[2]) >= float(tmp[3]):
                        raise NameError(
                            """The lower bound is equal or greater than the
                            upper bound of %s""" % tmp[0])
                else:
                    raise ValueError(
                        """ The line %s does not mention if the
                            characterization corresponds to a parameter ("par")
                            or a design variable ("var").""" % tmp)
        self.n_dim = len(self.var_dict.keys())
        self.n_par = len(self.par_dict.keys())

        # ------------------- CHECK IF VAR/PAR ARE CORRECT---------------------
        # ---------------------------------------------------------------------

        if not self.var_dict and 'UQ' not in self.opt_type:
            raise NameError(
                'Variable list is empty! Case cannot be instatiated!')

    def read_stochastic_space(self):

        path_to_read = os.path.join(self.case_path, 'stochastic_space')

        if any(
            x in self.opt_type for x in [
                'ROB',
                'UQ']) and os.path.isfile(path_to_read):

            with open(path_to_read, 'r') as f:
                for line in f:
                    tmp = line.split()
                    self.upar_dict[tmp[0]] = [tmp[1], tmp[2], float(tmp[3])]

        elif 'ROB' in self.opt_type and not os.path.isfile('stochastic_space'):
            raise NameError(
                """Missing file: "stochastic_space".
                   Uncertainty matrix cannot be built!""")

    def attach_objectives(self, obj):
        """
        Attaches the objectives of the configured optimization run.

        Parameters
        ----------
        obj : dict
            the objectives for the optimization run

        Returns
        -------
        None

        """

        self.obj = obj
