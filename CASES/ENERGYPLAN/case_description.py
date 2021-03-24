import os
from run_energyplan import energyplan

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
        fuel : float
            primary energy consumption
        co2 : float
            total CO2-emission
        '''

        fuel, co2 = energyplan(x)

        return co2, fuel
