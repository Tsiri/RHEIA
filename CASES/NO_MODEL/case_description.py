import os

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
        self.nPar = stochastic_design_space.nPar
        self.obj = stochastic_design_space.obj
        self.opt_type = stochastic_design_space.opt_type
