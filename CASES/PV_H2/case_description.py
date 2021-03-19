import lib_pv_h2 as pv_h2
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
        self.n_par = stochastic_design_space.n_par
        self.obj = stochastic_design_space.obj
        self.opt_type = stochastic_design_space.opt_type

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

        filename_demand = os.path.join(os.path.abspath(
                                        os.path.join(self.path,
                                                     os.pardir)),
                                        'DATA',
                                        'demand',
                                        'load_Brussels_dwelling.csv')

        myData = pv_h2.ReadData(filename_climate, filename_demand)
        G = myData.load_climate()
        load_elec = myData.load_demand()
        sell_grid = False

        self.params = [G,load_elec,sell_grid]


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
        lcoe : float
            the levelized cost of electricity
        ssr: float
            the self-sufficiency ratio
        '''

        x_dict = self.convert_into_dictionary(x[1])

        arguments = self.params + [x_dict]

        my_evaluation = pv_h2.Evaluation(*arguments)

        my_evaluation.evaluation()

        lcoe, ssr = my_evaluation.lcoe, my_evaluation.ssr

        return lcoe, ssr
