import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ReadData:

    def __init__(self, filename_climate, filename_demand):
        """

        This class enables to read data from the data files.

        Parameters
        ----------
        filename_climate : str
            The directory of the file with information on the
            solar irradiance.

        """
        self.filename_climate = filename_climate
        self.filename_demand = filename_demand
        self.path = os.path.dirname(os.path.abspath(__file__))

    def load_climate(self):
        """

        This method loads the hourly solar irradiance data,
        situated in the 'sol_irr' column of the climate data file.

        Returns
        -------
        sol_irr : ndarray
            The hourly solar irradiance data for a Typical
            Meteorological Year. (8760 elements)

        """
        data = pd.read_csv(self.filename_climate)
        sol_irr = data['sol_irr'].to_numpy()

        return sol_irr

    def load_demand(self):
        """
        
        This method loads the hourly electricity demand data,
        situated in the 'total electricity' column of the demand data file.

        Returns
        -------
        load_elec : ndarray
            The hourly electricity demand for a typical dwelling (8760 elements)

        """
        data = pd.read_csv(self.filename_demand)
        load_elec = data['total electricity'].to_numpy()

        return load_elec

    def load_parameters(self):
        """

        This method loads the deterministic values of the model
        parameters, defined in the design_space file. This is
        useful when the deterministic performance of a specific
        design needs to be evaluated.

        Returns
        -------
        param_dict : dict
            Dictionary with the names of the model parameters
            and the corresponding deterministic values.

        """
        param_dict = {}
        design_space = os.path.join(self.path, 'design_space')

        with open(design_space, 'r') as f:
            for line in f:
                tmp = line.split()
                if tmp[1] == 'par':
                    param_dict[tmp[0]] = float(tmp[2])

        return param_dict


class Evaluation:

    def __init__(self, G, L_elec, sell_grid, parameters):
        """

        This class evaluates the photovoltaic-hydrogen system.
        For a given design, the solar irradiance, electricity demand
        and the characterization of the model parameters,
        the levelized cost of electricity and the self-sufficiency ratio
        are quantified.

        Parameters
        ----------
        G : ndarray
            The hourly solar irradiance for the evaluated year. 
        L_elec : ndarray
            The hourly electricity demand for the evaluated year.
        sell_grid : bool
            Indicates if selling electricity to the grid is considered.
        parameters : dict
            Dictionary with the model parameters and design variables values.

        """
        self.parameters = parameters
        self.G = G

        self.load_elec = L_elec
        self.sell_grid = sell_grid

        self.length = len(self.G)

        self.G = G * self.parameters['sol_irr']
        self.load_elec *= (1. / sum(self.load_elec)) * \
            self.parameters['load_elec'] * 1e6 * self.length / 8760.
        self.dcac_capacity_array = np.zeros(self.length)
        self.grid_electricity_array = np.zeros(self.length)
        self.sold_electricity_array = np.zeros(self.length)
        self.m_h2_array = np.zeros(self.length)

        self.lifetime_system = 20.
        self.grid_cost = 0.
        self.grid_sold = 0.

        self.running_hours_pem = 0.
        self.running_hours_fc = 0.
        self.m_h2_max = self.tank()
        self.m_h2_min = 0.05 * self.m_h2_max
        self.m_h2 = self.m_h2_min

    def elec_profiles(self):
        """
        Set the grid electricity price for buying and selling electricity.

        Attributes:
            elec_profile (array): grid electricity buying price array
            elec_profile_sale (array): grid electricity selling price array
        """

        self.elec_profile = np.ones(self.length) *  \
            ((self.parameters['elec_cost'] + 20.) * (
                1. / self.parameters['elec_cost_ratio'])) / 1e6

        self.elec_profile_sale = np.ones(
            self.length) * self.parameters['elec_cost'] / 1e6

    def photovoltaic(self):
        """

        The power produced by the photovoltaic array is
        quantified for each hour and returned in an array.

        Returns
        -------
        pv_power : ndarray
            The hourly photovoltaic power production.

        """

        pv_power = np.zeros(self.length)
        for i in range(self.length):
            if self.G[i] > 0.:
                pv_power[i] = min(
                    (1. + self.parameters['power_tol_pv'] / 100.)
                    * self.G[i] * self.parameters['n_pv'],
                    self.parameters['n_dcdc_pv'] * 1e3
                )

        return pv_power

    def net_power(self):
        """
        
        Determine the hourly net power.
        This corresponds to the net power available (or still required)
        after extracting the hourly load from the photovoltaic power,
        considering DC-DC converter and DC-AC inverter efficiency.

        Returns
        -------
        p_net : ndarray
            The hourly net power. [W]

        """

        p_net = np.zeros(self.length)
        self.pv_power = self.photovoltaic()

        for i in range(self.length):
            p_req = self.load_elec[i] / (self.parameters['eff_dcdc']
                                         * self.parameters['eff_dcac'])
            if self.pv_power[i] >= p_req:
                p_net[i] = (self.pv_power[i] - p_req) * \
                    self.parameters['eff_dcdc']
            else:
                p_net[i] = (self.pv_power[i] * self.parameters['eff_dcdc'] *
                            self.parameters['eff_dcac']
                            - self.load_elec[i]) / self.parameters['eff_dcac']

        return p_net

    def tank(self):
        """
        
        Quantify the maximum storage capacity of the hydrogen storage tank.

        Returns
        -------
        m_max : float
            The hydrogen storage capacity. [kg]

        """

        m_max = self.parameters['n_tank'] / 33.33

        return m_max

    def electrolyzer_mh2(self, power):
        """

        The hydrogen production per hour, in function of
        the power supplied to the electrolyzer. The
        produced hydrogen is quantified in kg.

        Parameters
        ----------
        power : float
            The power supplied to the electrolyzer array in W.

        Returns
        -------
        m_h2 : float
            The hourly hydrogen production in kg.

        """

        m_h2 = self.parameters['eff_pem'] * power * 3600. / 120e6

        return m_h2

    def electrolyzer_power(self, m_h2):
        """

        The power required to produce the desired hydrogen.

        Parameters
        ----------
        m_h2 : float
            The hourly hydrogen production in kg.

        Returns
        -------
        power : float
            The power supplied to the electrolyzer array in W.

        """

        power = m_h2 * 120e6 / (self.parameters['eff_pem'] * 3600.)

        return power

    def charge_electrolyzer(self, Prem):
        """

        This method evaluates if the power generated by the
        photovoltaic array lies within the operating range
        of the DC-DC converter and of the electrolyzer array.
        If yes, the power is supplied to the electrolyzer array and the
        produced hydrogen is quantified. If the supplied power is larger than
        the electrolyzer capacity, than the nominal power is supplied.
        If the produced hydrogen cannot be stored in the storage tank, the
        supplied power is recalculated, such that the produced hydrogen matches
        the initial available space left in the hydrogen storage tank.
        Finally, the operating hours of the electrolyzer array is
        increased by 1.

        Parameters
        ----------
        power_elec : float
            The power supplied to the electrolyzer array.

        Returns
        -------
        p_consumed : float
            The actual power consumed by the electrolyzer array.
        """
        
        power_elec = min(Prem, self.parameters['n_dcdc_pem'] * 1e3) * \
            self.parameters['eff_dcdc']

        if (self.m_h2 < self.m_h2_max and
                power_elec > self.parameters['n_pem'] * 10.):

            p_consumed = min(power_elec, self.parameters['n_pem'] * 1e3)

            m_h2 = self.electrolyzer_mh2(p_consumed)
            self.running_hours_pem += 1.
            self.m_h2 += m_h2

            if self.m_h2 > self.m_h2_max:
                avail_m_h2 = self.m_h2_max - (self.m_h2 - m_h2)
                p_consumed = self.electrolyzer_power(avail_m_h2)
                self.m_h2 = self.m_h2_max

        else:
            p_consumed = 0.

        return p_consumed / self.parameters['eff_dcdc']

    def fuel_cell_mh2(self, power):
        """

        The hydrogen consumption per hour, in function of
        the power demanded from the fuel cell. The
        required hydrogen is quantified in kg.

        Parameters
        ----------
        power : float
            The power required from the fuel cell array in W.

        Returns
        -------
        m_h2 : float
            The hourly hydrogen supply in kg.

        """

        m_h2 = power / (self.parameters['eff_fc'] * 120e6) * 3600.

        return m_h2

    def fuel_cell_power(self, m_h2):
        """

        The power produced in function of the avialable hydrogen mass flow rate.

        Parameters
        ----------
        m_h2 : float
            The hourly hydrogen supply in kg.

        Returns
        -------
        power : float
            The power required from the fuel cell array in W.

        """

        power = m_h2 * 120e6 * self.parameters['eff_fc'] / 3600.

        return power

    def charge_fuel_cell(self, power_req):
        """

        This method evaluates if the power required from the fuel cell
        lies within the operating range of the DC-DC converter and of the
        fuel cell array. If yes, the power is supplied and the
        consumed hydrogen is quantified. If the required power is larger than
        the fuel cell capacity, than the nominal power is supplied.
        If the consumed hydrogen is larger than the available hydrogen in
        the hydrogen storage tank, the supplied power is recalculated,
        such that the consumed hydrogen matches the initial available hydrogen
        left in the hydrogen storage tank.
        Finally, the operating hours of the fuel cell array is
        increased by 1.

        Parameters
        ----------
        power_req : float
            The power demanded from the fuel cell array.

        Returns
        -------
        p_produced : float
            The actual power produced by the fuel cell array.
        """

        power_fc = min(power_req, self.parameters['n_dcdc_fc'] * 1e3) / \
            self.parameters['eff_dcdc']
        if (self.m_h2 > self.m_h2_min and
                power_fc > self.parameters['n_fc'] * 10.):

            p_produced = min(power_fc, self.parameters['n_fc'] * 1e3)

            m_h2 = self.fuel_cell_mh2(p_produced)
            self.running_hours_fc += 1.
            self.m_h2 -= m_h2

            if self.m_h2 < self.m_h2_min:
                avail_m_h2 = self.m_h2 + m_h2 - self.m_h2_min
                p_produced = self.fuel_cell_power(avail_m_h2)
                self.m_h2 = self.m_h2_min
        else:
            p_produced = 0.

        return p_produced * self.parameters['eff_dcdc']

    def evaluation(self):
        """

        This is the main method of the Evaluation class.
        For each hour, the power management strategy is applied.
        Finally, the electrolyzer lifetime,
        self-sufficiency ratio and the system cost are determined.

        """
        self.elec_profiles()
        self.pv_power = self.photovoltaic()
        p_net = self.net_power()

        for t in range(self.length):
            e_grid_buy = 0.
            e_grid_sold = 0.
            if p_net[t] > 0.:
                p_consumed = self.charge_electrolyzer(p_net[t])
                p_rem = p_net[t] - p_consumed

                e_grid_sold = p_rem * self.parameters['eff_dcac']
                self.grid_sold += e_grid_sold * self.elec_profile_sale[t]
                self.dcac_capacity_array[t] += self.load_elec[t] + e_grid_sold

            elif p_net[t] < 0.:
                p_req = abs(p_net[t])
                p_produced = self.charge_fuel_cell(p_req)
                p_req -= p_produced

                e_grid_buy += p_req * self.parameters['eff_dcac']
                self.dcac_capacity_array[t] += (self.load_elec[t] - e_grid_buy)
                self.grid_cost += e_grid_buy * self.elec_profile[t]

            self.grid_electricity_array[t] = e_grid_buy
            self.sold_electricity_array[t] = e_grid_sold
            self.m_h2_array[t] = (self.m_h2 - self.m_h2_min) / \
                ((self.m_h2_max - self.m_h2_min))

        self.lifetime()
        self.self_sufficiency_ratio()
        self.cost()

    def lifetime(self):
        """

        The lifetime method determines the lifetime of
        the electrolyzer array and fuel cell array, based on the number of
        operating hours during the evaluated year.

        """

        if self.running_hours_pem == 0.:
            self.lifeELEC = 1e8
        else:
            self.lifeELEC = self.parameters['life_pem'] / self.running_hours_pem

        if self.running_hours_fc == 0.:
            self.lifeFC = 1e8
        else:
            self.lifeFC = self.parameters['life_fc'] / self.running_hours_fc

    def self_sufficiency_ratio(self):
        """
        
        The self-sufficiency ratio is quantified.
        
        """
        self.ssr = 1. - sum(self.grid_electricity_array) / sum(self.load_elec)

    def cost(self):
        """

        Based on the capital recovery factor, the CAPEX,
        OPEX and replacement cost of the system components,
        the levelized cost of electricity is determined.

        """

        lifetime = self.lifetime_system

        inv_rate = (
            self.parameters['int_rate'] - self.parameters['infl_rate']) / (
            1. + self.parameters['infl_rate'])
        CRF = (((1. + inv_rate)**lifetime - 1.) /
               (inv_rate * (1. + inv_rate)**lifetime))**(-1)

        self.pv_cost = self.parameters['n_pv'] * (
            CRF * self.parameters['capex_pv'] + self.parameters['opex_pv'])
        self.pv_dcdc_cost = self.parameters['n_dcdc_pv'] * (
            self.parameters['capex_dcdc'] *
            (CRF + self.parameters['opex_dcdc']))
        components_cost = self.pv_cost + self.pv_dcdc_cost

        self.pem_cost = self.parameters['n_pem'] * (
            self.parameters['capex_pem'] *
            (CRF + self.parameters['opex_pem']))
        self.pem_dcdc_cost = self.parameters['n_dcdc_pem'] * (
            self.parameters['capex_dcdc'] *
            (CRF + self.parameters['opex_dcdc']))
        components_cost += self.pem_cost + self.pem_dcdc_cost
        self.fc_cost = self.parameters['n_fc'] * (
            self.parameters['capex_fc'] * CRF + self.parameters['opex_fc'] *
            self.running_hours_fc)
        self.fc_dcdc_cost = self.parameters['n_dcdc_fc'] * (
            self.parameters['capex_dcdc'] *
            (CRF + self.parameters['opex_dcdc']))
        components_cost += self.fc_cost + self.fc_dcdc_cost

        self.tank_cost = self.parameters['n_tank'] * (
            self.parameters['capex_tank'] *
            (CRF + self.parameters['opex_tank']))
        components_cost += self.tank_cost

        self.dcac_cost = max(self.dcac_capacity_array) * \
            (self.parameters['capex_dcac'] *
             (CRF + self.parameters['opex_dcac']))
        components_cost += self.dcac_cost

        ARC = 0
        ARC += CRF * sum([(1. + inv_rate)**(-(i + 1.) * self.lifeELEC) *
                          self.parameters['n_pem'] *
                          self.parameters['repl_pem'] *
                          self.parameters['capex_pem'] for i in
                          range(int(lifetime / self.lifeELEC))])
        ARC += CRF * sum([(1. + inv_rate)**(-(i + 1.) * self.lifeFC) *
                          self.parameters['n_fc'] *
                          self.parameters['repl_fc'] *
                          self.parameters['capex_fc'] for i in
                          range(int(lifetime / self.lifeFC))])

        if not self.sell_grid:
            self.grid_sold = 0.

        cost = ARC + components_cost + self.grid_cost - self.grid_sold

        self.lcoe = cost / (sum(self.load_elec)) * 1e6

    def print_results(self):
        """

        This method prints the levelized cost of electricity,
        the self-sufficiency ratio, the annual energy produced
        by the photovoltaic array and the lifetime of the
        electrolyzer array and fuel cell array.

        """
        print('outputs:')
        print('LCOE:'.ljust(30) + '%.2f euro/MWh' % self.lcoe)
        print('SSR:'.ljust(30) + '%.2f %%' % (self.ssr * 100.))
        print('PV electricity generated:'.ljust(30) +
              '%.2f MWh' % (sum(self.pv_power) / 1e6))
        print('life electrolyzer:'.ljust(30) + '%.2f year' % self.lifeELEC)
        print('life fuel cell:'.ljust(30) + '%.2f year' % self.lifeFC)

        plt.plot(self.m_h2_array)
        plt.show(block=False)
