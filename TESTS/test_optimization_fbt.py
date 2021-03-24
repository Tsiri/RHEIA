import os, sys
import shutil
import pytest
import multiprocessing as mp 
import gc

path = os.path.split(
    os.path.dirname(
        os.path.abspath(__file__)))[0]
sys.path.insert(0, os.path.join(path,'OPT'))
from optimization import run_opt

def test_fbt():
    dict_opt = {'case': 'FOUR_BAR_TRUSS',
                'objectives':          {'DET': (-1, -1)}, 
                'stop':                ('BUDGET', 100),
                'population size':     20,
                'results dir':      'testdir',
               }

    run_opt(dict_opt)

def test_fbt_rob():
    dict_opt = {'case': 'FOUR_BAR_TRUSS',
                'objectives':          {'ROB': (-1, -1)}, 
                'stop':                ('BUDGET', 200),
                'population size':     6,
                'results dir':      'testdir',
                'pol order': 1,
                'objective names': ['V','D'],
                'objective of interest': ['D'],
                'n jobs':              int(mp.cpu_count()/2), 
               }
    
    run_opt(dict_opt)