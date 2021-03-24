import os
import lib_pv_electrolyzer as pv_elec

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

        filename_climate = os.path.join(os.path.abspath(
                                        os.path.join(self.path,
                                                     os.pardir)),
                                        'DATA',
                                        'climate',
                                        'climate_Brussels.csv')

        my_data = pv_elec.ReadData(filename_climate)
        G = my_data.load_climate()

        self.params = [G]

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
        
        par_dict = self.stochastic_design_space.par_dict
        var_dict = self.stochastic_design_space.var_dict
        parameters = dict(zip(par_dict.keys(), x[:len(par_dict)]))
        if var_dict:
            inputs = dict(zip(var_dict.keys(), x[-len(var_dict):]))
        else:
            inputs = {}

        return {**parameters, **inputs}

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
        lcoh : float
            the levelized cost of hydrogen
        mh2: float
            hydrogen production
        '''

        x_dict = self.convert_into_dictionary(x[1])

        arguments = self.params + [x_dict]

        my_evaluation = pv_elec.Evaluation(*arguments)

        my_evaluation.evaluation()

        lcoh, mh2 = my_evaluation.lcoh, my_evaluation.m_h2

        return lcoh, mh2
