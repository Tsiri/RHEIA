
import pv_electrolyzer_lib as lb
import numpy as np
import matplotlib.pyplot as plt
import time
import os

path = os.path.dirname(os.path.abspath(__file__)) 	
filenameClimate = "climate_Brussels.csv"

myData = lb.ReadData(filenameClimate)
G = myData.load_climate() #W/m2,degC
parameters = myData.load_parameters()


x = [1.9304161672, 1.6993071716, 1.7560465536]

ll = 8760
t = time.time()
inputs = {'n_dcdc_pv': x[0], 'n_elec': x[1], 'n_dcdc_elec': x[2]}

myEvaluation = lb.Evaluation(G[:ll], {**parameters, **inputs})
myEvaluation.evaluation()
print('time: %d' %(time.time()-t))
myEvaluation.print_results()
#LCOE,LCOH,LCOEn,LCOX,SSR,SSRH,SSRex,ex_eff = myEvaluation.getObjectives()


