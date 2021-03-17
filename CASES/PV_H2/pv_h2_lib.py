# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:36:14 2019

@author: Diederik Coppitters
"""

'''

Short description ::
-----------------

component models for HRES


- python2.7
- numpy 
- scipy
- matplotlib


'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

#import pvlib


class ReadData:
    
    def __init__(self,filename_climate, filename_demand):
        self.filename_climate = filename_climate
        self.filename_demand = filename_demand
        self.path = os.path.dirname(os.path.abspath(__file__)) 

    def load_climate(self):
        data = pd.read_csv(self.filename_climate)
        S_irr = data['sol_irr'].to_numpy()

        return S_irr

    def load_demand(self):
        data = pd.read_csv(self.filename_demand)
        load_elec = data['total electricity'].to_numpy()

        return load_elec
        
    def load_parameters(self):
        param_dict = {}
        design_space = os.path.join(self.path,'design_space')
        
        with open(design_space, 'r') as f:
            for l in f:
                tmp = l.split()
                if tmp[1] == 'par':
                    param_dict[tmp[0]] = float(tmp[2])
        
        return param_dict
    

class Evaluation:

    def __init__(self,G,L_elec,sell_grid,parameters):
        self.parameters = parameters
        self.G = G
        
        self.load_elec = L_elec
        self.sell_grid = sell_grid

        self.length = len(self.G)

        self.G = G*self.parameters['sol_irr']
        self.load_elec = self.load_elec/sum(self.load_elec) *self.parameters['load_elec']*1e6 * self.length/8760.
        self.dcac_capacity_array = np.zeros(self.length)
        self.grid_electricity_array = np.zeros(self.length)
        self.sold_electricity_array = np.zeros(self.length)
        self.m_h2_array = np.zeros(self.length)

        self.lifetime_system = 20.
        self.grid_cost = 0.
        self.grid_sold = 0.

        self.runningHoursEL = 0.
        self.runningHoursFC = 0.
        self.m_H2_max = self.tank()
        self.m_H2_min = 0.05*self.m_H2_max
        self.m_H2 = self.m_H2_min                

    def elec_profiles(self):
        """
        Set the grid electricity price for buying and selling electricity. 
            
        Attributes:
            elec_profile (array): grid electricity buying price array 
            elec_profile_sale (array): grid electricity selling price array 
        """

        self.elec_profile = np.ones(self.length) * ((self.parameters['elec_cost']+20.)
                            * (1./self.parameters['elec_cost_ratio'])) / 1e6

        self.elec_profile_sale = np.ones(self.length) * self.parameters['elec_cost'] / 1e6 
        
    def photovoltaic(self):

        pv_power = np.zeros(self.length)
        for i in range(self.length):
            if self.G[i] > 0.:
                pv_power[i] = min(
                              (1.+self.parameters['power_tol_pv']/100.) 
                              * self.G[i]*self.parameters['n_pv'],
                              self.parameters['n_dcdc_pv']*1e3
                              )

        return pv_power


    def net_power(self):
        """
        Determine the hourly net power.
        This corresponds to the net power available (or still required) after extracting
        the hourly load from the photovoltaic power, considering DC-DC converter and
        DC-AC inverter efficiency.
        
        Attributes:
            pv_power (array): hourly photovoltaic power [W]

        Returns:
            p_net (array): the hourly net power [W]

        """

        p_net = np.zeros(self.length)
        self.pv_power = self.photovoltaic()
    
        for i in range(self.length):
            p_req = self.load_elec[i] / (self.parameters['eff_dcdc']*self.parameters['eff_dcac'])
            if self.pv_power[i] >= p_req:
                p_net[i] = (self.pv_power[i]-p_req) * self.parameters['eff_dcdc']
            else:
                p_net[i] = (self.pv_power[i]*self.parameters['eff_dcdc']*self.parameters['eff_dcac']
                            - self.load_elec[i]) / self.parameters['eff_dcac']
        
        return p_net

    def tank(self):

        m_max = self.parameters['n_tank']/33.33

        return m_max
    

    def electrolyzer_mh2(self,power):
        
        m_H2 = self.parameters['eff_elec']*power*3600./120e6
        
        return m_H2

    def electrolyzer_power(self,m_H2):
        
        power = m_H2*120e6/(self.parameters['eff_elec']*3600.)
        
        return power

    def charge_electrolyzer(self,Prem):
    
        power_elec = min(Prem,self.parameters['n_dcdc_elec']*1e3)*self.parameters['eff_dcdc']
        #print('power na dcdc elec:', power_elec)
        if self.m_H2 < self.m_H2_max and power_elec > self.parameters['n_elec']*10.:
        
            p_consumed = min(power_elec,self.parameters['n_elec']*1e3)          
            #print('power nr elec:', p_consumed)
                        
            m_H2 = self.electrolyzer_mh2(p_consumed)                                                
            self.runningHoursEL += 1.
            self.m_H2 += m_H2
            
            if self.m_H2 > self.m_H2_max:
                avail_m_H2 = self.m_H2_max - (self.m_H2 - m_H2)
                p_consumed = self.electrolyzer_power(avail_m_H2)
                self.m_H2 = self.m_H2_max
                #print('POWER ELEC NIEUW:', p_consumed)
        else:
            p_consumed = 0.

        return p_consumed/self.parameters['eff_dcdc']
                
    def fuel_cell_mh2(self,power):
        
        m_H2 = power/(self.parameters['eff_fc']*120e6)*3600.
        
        return m_H2

    def fuel_cell_power(self,m_H2):
        
        power = m_H2*120e6*self.parameters['eff_fc']/3600.

        #eff = P_stack/(n_H2*2.02/1000.*LHV)
        
        return power

    def charge_fuel_cell(self,Preq):
    
        power_fc = min(Preq,self.parameters['n_dcdc_fc']*1e3)/self.parameters['eff_dcdc']
        if self.m_H2 > self.m_H2_min and power_fc > self.parameters['n_fc']*10.:
        
            p_produced = min(power_fc,self.parameters['n_fc']*1e3)          
            #print('power nr fc:', p_produced)
                        
            m_H2 = self.fuel_cell_mh2(p_produced)                                                
            self.runningHoursFC += 1.
            self.m_H2 -= m_H2
            
            if self.m_H2 < self.m_H2_min:
                avail_m_H2 = self.m_H2 + m_H2 - self.m_H2_min
                #print(m_H2,self.m_H2,avail_m_H2,self.m_H2_min)
                p_produced = self.fuel_cell_power(avail_m_H2)
                self.m_H2 = self.m_H2_min
                #print('power nr fc nieuw:', p_produced)
        else:
            p_produced = 0.

        return p_produced*self.parameters['eff_dcdc']
                

    ##########################################
    ## model evaluation
    ##########################################

    def evaluation(self):
        self.elec_profiles()
        self.pv_power = self.photovoltaic()
        p_net = self.net_power()

        for t in range(self.length):
            e_grid_buy = 0.
            e_grid_sold = 0.
            #print('pnet: ', p_net[t])
            if p_net[t] > 0.:
                #print('pnet: ', p_net[t])
                p_consumed = self.charge_electrolyzer(p_net[t])
                p_rem = p_net[t]-p_consumed
                #print('prem: ', p_rem)

                e_grid_sold = p_rem*self.parameters['eff_dcac']
                self.grid_sold += e_grid_sold * self.elec_profile_sale[t]
                self.dcac_capacity_array[t] += self.load_elec[t] + e_grid_sold

            elif p_net[t] < 0.:
                p_req = abs(p_net[t])
                #print('pnet: ', p_net[t])
                p_produced = self.charge_fuel_cell(p_req)
                p_req -= p_produced 
                #print('preq: ', p_req)
                
                e_grid_buy += p_req*self.parameters['eff_dcac']
                self.dcac_capacity_array[t] += (self.load_elec[t] - e_grid_buy)
                self.grid_cost += e_grid_buy * self.elec_profile[t]

            self.grid_electricity_array[t] = e_grid_buy
            self.sold_electricity_array[t] = e_grid_sold
            self.m_h2_array[t] = (self.m_H2 - self.m_H2_min)/((self.m_H2_max - self.m_H2_min))
                
        self.lifetime()
        self.self_sufficiency_ratio()
        self.cost()

    def lifetime(self):

        if self.runningHoursEL == 0.:
            self.lifeELEC = 1e8
        else:    
            self.lifeELEC = self.parameters['life_elec']/self.runningHoursEL

        if self.runningHoursFC == 0.:
            self.lifeFC = 1e8
        else:    
            self.lifeFC = self.parameters['life_fc']/self.runningHoursFC

    def self_sufficiency_ratio(self):
        self.ssr = 1. - sum(self.grid_electricity_array)/sum(self.load_elec)

    def cost(self):

        lifetime = self.lifetime_system

        inv_rate = (self.parameters['int_rate'] - self.parameters['infl_rate'])/(1. + self.parameters['infl_rate'])        
        CRF = (((1.+inv_rate)**lifetime-1.)/
              (inv_rate*(1.+inv_rate)**lifetime))**(-1)

        self.PVcost = self.parameters['n_pv'] * (CRF*self.parameters['capex_pv'] + self.parameters['opex_pv'])
        self.PVDCDCcost = self.parameters['n_dcdc_pv'] * (self.parameters['capex_dcdc'] * (CRF + self.parameters['opex_dcdc']))
        components_cost = self.PVcost + self.PVDCDCcost

        self.ELCost = self.parameters['n_elec'] * (self.parameters['capex_elec'] * (CRF + self.parameters['opex_elec']))
        self.ELDCDCCost = self.parameters['n_dcdc_elec'] * (self.parameters['capex_dcdc'] * (CRF + self.parameters['opex_dcdc']))                
        components_cost += self.ELCost + self.ELDCDCCost
        #print('opex', self.parameters['opex_fc'])
        #print('running hours', self.runningHoursFC)
        self.FCCost = self.parameters['n_fc'] * (self.parameters['capex_fc'] * CRF + self.parameters['opex_fc']*self.runningHoursFC)
        self.FCDCDCCost = self.parameters['n_dcdc_fc'] * (self.parameters['capex_dcdc'] * (CRF + self.parameters['opex_dcdc']))                
        components_cost += self.FCCost + self.FCDCDCCost

        self.tankCost = self.parameters['n_tank'] * (self.parameters['capex_tank'] * (CRF + self.parameters['opex_tank']))
        components_cost += self.tankCost

        self.DCACCost = max(self.dcac_capacity_array) * (self.parameters['capex_dcac'] * (CRF + self.parameters['opex_dcac']))                
        components_cost += self.DCACCost

        ARC = 0
        ARC += CRF*sum([ (1.+inv_rate)**(-(i+1.)*self.lifeELEC)*self.parameters['n_elec']*self.parameters['repl_elec']*self.parameters['capex_elec'] for i in range(int(lifetime/self.lifeELEC)) ])
        ARC += CRF*sum([ (1.+inv_rate)**(-(i+1.)*self.lifeFC)*self.parameters['n_fc']*self.parameters['repl_fc']*self.parameters['capex_fc'] for i in range(int(lifetime/self.lifeFC)) ])
        
        if not self.sell_grid:
            self.grid_sold = 0.
            
        cost = ARC + components_cost + self.grid_cost - self.grid_sold
        
        self.lcoe = cost/(sum(self.load_elec))*1e6
        
    def print_results(self):
        print( 'outputs:' )
        print( 'LCOE:'.ljust(30) + '%.2f euro/MWh' %self.lcoe) 
        print( 'SSR:'.ljust(30) + '%.2f %%' %(self.ssr*100.))
        print('PV electricity generated:'.ljust(30) + '%.2f MWh' %(sum(self.pv_power)/1e6))
        print( 'life electrolyzer:'.ljust(30) + '%.2f year' %self.lifeELEC)
        print( 'life fuel cell:'.ljust(30) + '%.2f year' %self.lifeFC)
        
        plt.plot(self.m_h2_array)
        plt.show(block=False)
        
    def get_objectives(self):
        return self.lcoe,self.ssr
        
        
        

