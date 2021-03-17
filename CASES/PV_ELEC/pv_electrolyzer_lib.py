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
    
    def __init__(self,filename_climate):
        self.filename_climate = filename_climate
        self.path = os.path.dirname(os.path.abspath(__file__)) 

    def load_climate(self):
        data = pd.read_csv(self.filename_climate)
        S_irr = data['sol_irr'].to_numpy()

        return S_irr
        
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

    def __init__(self,G,parameters):
        self.parameters = parameters
        self.G = G*self.parameters['sol_irr']
        self.length = len(self.G)

        self.lifetime_system = 20.
        self.pv_power_consumed = 0.
        self.m_H2 = 0.
        self.runningHoursEL = 0.
        
    def photovoltaic(self):

        pv_power = np.zeros(self.length)
        for i in range(self.length):
            if self.G[i] > 0.:
                pv_power[i] = min(
                              (1.+self.parameters['power_tol_pv']/100.) 
                              * self.G[i]*self.parameters['n_pv'],
                              self.parameters['n_dcdc_pv']*1e3
                              )*self.parameters['eff_dcdc']

        return pv_power
    
    def electrolyzer(self,power):
        
        m_H2 = self.parameters['eff_elec']*power*3600./120e6
        
        return m_H2

    def fill_electrolyzer(self,Prem):
    
        power_elec = min(Prem,self.parameters['n_dcdc_elec']*1e3)*self.parameters['eff_dcdc']

        if power_elec > self.parameters['n_elec']*10.:
        
            power_elec = min(power_elec,self.parameters['n_elec']*1e3)          
                        
            m_H2 = self.electrolyzer(power_elec)                                    
            
            self.runningHoursEL += 1.
            
            self.pv_power_consumed += power_elec

        else:
            m_H2 = 0.

        return m_H2
                

    ##########################################
    ## model evaluation
    ##########################################

    def evaluation(self):
        self.pv_power = self.photovoltaic()

        for i in self.pv_power:

            self.m_H2 += self.fill_electrolyzer(i)
            
        self.lifetime()
        self.cost()

    def lifetime(self):

        if self.runningHoursEL == 0.:
            self.lifeELEC = 1e8
        else:    
            self.lifeELEC = self.parameters['life_elec']/self.runningHoursEL
            
    def cost(self):

        lifetime = self.lifetime_system

        inv_rate = (self.parameters['int_rate'] - self.parameters['infl_rate'])/(1. + self.parameters['infl_rate'])        
        CRF = (((1.+inv_rate)**lifetime-1.)/(inv_rate*(1.+inv_rate)**lifetime))**(-1)

        self.PVcost = self.parameters['n_pv'] * (CRF*self.parameters['capex_pv'] + self.parameters['opex_pv'])
        self.PVDCDCcost = self.parameters['n_dcdc_pv'] * (self.parameters['capex_dcdc'] * (CRF + self.parameters['opex_dcdc']))
        components_cost = self.PVcost + self.PVDCDCcost

        self.ELCost = self.parameters['n_elec'] * (self.parameters['capex_elec'] * (CRF + self.parameters['opex_elec']))
        self.ELDCDCCost = self.parameters['n_dcdc_elec'] * (self.parameters['capex_dcdc'] * (CRF + self.parameters['opex_dcdc']))                
        components_cost += self.ELCost + self.ELDCDCCost

        ARC = 0
        ARC += CRF*sum([ (1.+inv_rate)**(-(i+1.)*self.lifeELEC)*self.parameters['n_elec']*self.parameters['repl_elec']*self.parameters['capex_elec'] for i in range(int(lifetime/self.lifeELEC)) ])
        cost = ARC + components_cost
        if self.m_H2 < 1e-5:
            self.LCOH = 1e3 + 1e5*random.random()
        else:    
            self.LCOH = cost/self.m_H2

    def print_results(self):
        print( 'outputs:' )
        print( 'LCOH:'.ljust(30) + '%.2f euro/kg' %self.LCOH) 
        print( 'm_H2:'.ljust(30) + '%.2f kg' %self.m_H2)
        print('PV electricity generated:'.ljust(30) + '%.2f MWh' %(sum(self.pv_power)/1e6))
        print('PV electricity consumed:'.ljust(30) + '%.2f MWh' %(self.pv_power_consumed/1e6))
        print('self-consumption ratio:'.ljust(30) + '%.2f %%' %(1e2*self.pv_power_consumed/sum(self.pv_power)))
        print( 'life electrolyzer:'.ljust(30) + '%.2f year' %self.lifeELEC)
                    
    def get_objectives(self):
        return self.LCOH,self.m_H2
        
        
        

