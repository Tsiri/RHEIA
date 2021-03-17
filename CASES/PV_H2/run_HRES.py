
import pv_h2_lib as lb
import numpy as np
import matplotlib.pyplot as plt
import time
import os

path = os.path.dirname(os.path.abspath(__file__)) 	

filename_climate = os.path.join(os.path.abspath(os.path.join(path, os.pardir)),
                                'DATA',
                                'climate',
                                'climate_Brussels.csv')

filename_load = os.path.join(os.path.abspath(os.path.join(path, os.pardir)),
                                'DATA',
                                'demand',
                                'load_Brussels_dwelling.csv')

myData = lb.ReadData(filename_climate, filename_load)
G = myData.load_climate() #W/m2,degC
load_elec = myData.load_demand() #W/m2,degC
parameters = myData.load_parameters()


x = [39.871049, 16.658545, 0.011440, 1.011904, 0.875798, 1.510053, 189.065948,1.,]

x = [25.922670, 13.307593, 2.692685, 3.741924, 0.000662176, 1.625991, 132.346403,] 

x = [25.922670, 13.307593, 1e-8,1e-8,1e-8,1e-8,1e-8] 

x = [5.288813, 1.883958, 0.467710, 2.102333, 0.236113, 3.115239, 62.925609,]

x = [5.288813, 1.883958, 1e-8,1e-8,1e-8,1e-8,1e-8]

ll = 8760
t = time.time()
inputs = {'n_pv': x[0],
          'n_dcdc_pv': x[1],
          'n_elec': x[2],
          'n_dcdc_elec': x[3],
          'n_fc': x[4],
          'n_dcdc_fc': x[5],
          'n_tank': x[6], 
          }

myEvaluation = lb.Evaluation(G[:ll], load_elec[:ll], {**parameters, **inputs})
myEvaluation.evaluation()
print('time: %d' %(time.time()-t))
myEvaluation.print_results()
#LCOE,LCOH,LCOEn,LCOX,SSR,SSRH,SSRex,ex_eff = myEvaluation.getObjectives()


