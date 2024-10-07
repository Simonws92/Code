# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:29:22 2024

@author: Simon
"""

import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

import Tolletaten_defines as td
import Tolletaten_model as tm

"Global variables"
Training_X = {}
Training_Y = {}
Training_Y_2 = {}

"""Column names:
0    'declaration_date',
1    'statistical_value',
2    'net_weight',
3    'commodity_code',
4    'destination_country_full',
5    'mode_of_transport_full',
6    'domestic_party_id_unique'    
"""


if True:
    path = "C:/Users/Simon/Desktop/Tolletaten/export_kongekrabbe.csv"

    Dict_of_countries_with_data, Total_data, \
        Dict_of_ID_with_data, unknown_destination = td.load_csv(path)





"Test a specific country"
if True:
    for country in Dict_of_countries_with_data:
        
        if country == 'GB':
            country_dict = Dict_of_countries_with_data[country]
            _ = td.test(country_dict, country)





"""
Revenue, Ratio, Amount
"""

"This goes through every country"
if False:
    for country in Dict_of_countries_with_data:
        td.test_cyclical(Dict_of_countries_with_data[country], country=country,control = "Ratio")



"This goes through every ID"
if False:
    for id_ in Dict_of_ID_with_data:
        
        
        td.test_cyclical(Dict_of_ID_with_data[id_], country=id_, scrutinize = "ID",control = "Revenue")






#######################################################
### DEEP LEARNING MODEL TO PREDICT REASONABLE VALUE ###
#######################################################

"We create a model to predict when events may occur based on if the snapshot has an event"

"The events are defined using a Z-score evaluation"

"The likelihood of a number of events is based on the poisson distribution"

if False:
    X,Y = td.compile_data(Dict_of_countries_with_data)
    
    "Convert from list to np.array"
    X = np.array(X)
    Y = np.array(Y)
    
    "Normalizes the data wrt every batch"
    X = ( X - np.average(X) ) / np.std(X)
    Y = ( Y - np.average(Y) ) / np.std(Y)
    
    "Adds an extra feature dimension"
    np_X, np_Y = np.array(X)[:,:,None], np.array(Y)[:,:,None]
    
    
    training_loss = tm.Training(np_X,np_Y)
    
    plt.figure()
    plt.plot(training_loss)
    




















