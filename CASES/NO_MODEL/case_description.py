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
