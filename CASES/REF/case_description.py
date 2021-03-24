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
        self.params = []

        self.set_params()

    def set_params(self):
        """
        Set the fixed parameters for each model evaluation.

        """

        self.params = []

    def convert_into_dictionary(self, x):
        """
        Convert the input sample for model evaluation into a dictionary.

        Parameters
        ----------
        x : array
            Input sample for model evaluation.
            x = [xp1, xp2, ..., xpm, xd1, xd2, ..., xdn]
            parameters are included for uncertainty quantification reasons

        Returns
        -------
        {**parameters, **inputs} : dict
            the dictionary with the variable and parameter names as keys,
            and the input sample values as values

        """

        parameters = dict(zip(self.par_dict.keys(), x[:len(self.par_dict)]))
        if self.var_dict:
            inputs = dict(zip(self.var_dict.keys(), x[-len(self.var_dict):]))
        else:
            inputs = {}

        return {**parameters, **inputs}

    def evaluate(self, x, *args):
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
        y1 : float
            model output 1
        y2 : float
            model output 2
        '''

        y1 = 1.
        y2 = 1.
        
        return y1, y2
