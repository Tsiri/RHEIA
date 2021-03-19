import os
from four_bar_truss import four_bar_truss


class CASE(object):

    def __init__(self, stochastic_design_space):
        """
        Class which initiates the information on the stochastic design space
        and evaluates the system model.

        Parameters
        ----------
        stochastic_design_space : string
            The stochastic_design_space object which contains attributes
            with information on the stochastic design space.

        """
        self.stochastic_design_space = stochastic_design_space
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.case = stochastic_design_space.case
        self.lb = stochastic_design_space.lb
        self.ub = stochastic_design_space.ub
        self.par_dict = stochastic_design_space.par_dict
        self.var_dict = stochastic_design_space.var_dict
        self.upar_dict = stochastic_design_space.upar_dict
        self.n_dim = stochastic_design_space.n_dim
        self.n_par = stochastic_design_space.n_par
        self.obj = stochastic_design_space.obj
        self.opt_type = stochastic_design_space.opt_type

    def evaluate(self, x):
        '''
        Evaluation of the system objectives for one given design.

        Parameters
        ----------
        x : tuple
            An enumerate object for the input sample.
            The first element of x
            - the index of the input sample in the list of samples -
            can be used for multiprocessing purposes of executable files
            with input and output text files.
            The second element of x - the input sample -
            is an array with the values for the model parameters
            and design variables, if present:
            x = [xp1, xp2, ..., xpm, xd1, xd2, ..., xdn],
            where xp are the values for the model parameters,
            xd are the values for the design variables,
            m and n equal the number of model parameters
            and design variables, respectively.

        Returns
        -------
        V : float
            truss volume
        d : float
            displacement of outer node
        '''

        V, d = four_bar_truss(x[1])

        return V, d
