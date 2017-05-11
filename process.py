# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:47:15 2017

@author: CStorm
"""

import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    
    X = data[:, :-1]
    Y = data[:, -1]
    
    X[:,1] = (X[:,1]-X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2]-X[:,2].mean()) / X[:,2].std()

