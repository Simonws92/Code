# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:12:14 2024

@author: Simon
"""

import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import random



def month_day_fcn(month, year):
    "We sum the nr of days up to and excl the current year"
    sum_days = 0
    if year == 2020:
        sum_days += 1
    
    "2019 is the initial year"
    for i in range(year-2019):
        sum_days += 365
        
    
    "We sum the nr of days up to and excluding the current month"
    if year == 2020:    
        nr_days = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        nr_days = [31,28,31,30,31,30,31,31,30,31,30,31]
    for i in range(month):
        sum_days += nr_days[i]
    
    return sum_days


def sort_dict(dict_to_sort):
    "sort wrt days"
    for entry in dict_to_sort:
        sort_list = np.array( dict_to_sort[entry] )
        sorted_list = sort_list[sort_list[:, 0].argsort()]
        dict_to_sort[entry] = sorted_list
    return dict_to_sort


def load_csv(path):
    
    data_list_raw = []
    Full_information = []
    unknown_destination = []
    
    statistical_value   = []
    commodity_code      = []
    destination_country = []
    mode_of_transport   = []
    domestic_party_id   = []
    
    Dict_of_countries_with_data = {}
    Dict_of_ID_with_data = {}
    
    nr_of_missing_entries = 0
    
    with open(path, newline='') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=',')
        
        
        for i, row in enumerate(csv_file):
            
            if i == 0:
                "We remove the first entry which are column names"
                pass
            else:
                
                "Store raw data"
                data_list_raw.append( row )
                
                "We list different categories which we will sort the data by later"
                st_val, amount, comm_code, dest_entry, mode_of_tr, dom_prty_id = \
                    row[1:]
                
                dest_entry, st_val, amount = dest_entry[:2], int(st_val), int(amount)
                
                if '' in row:
                    nr_of_missing_entries += 1
                
                
                if st_val not in statistical_value:
                    statistical_value.append( int(st_val) )
                if comm_code not in commodity_code:
                    commodity_code.append(comm_code)
                if dest_entry not in destination_country:
                    destination_country.append(dest_entry)
                if mode_of_tr not in mode_of_transport:
                    mode_of_transport.append(mode_of_tr)
                if dom_prty_id not in domestic_party_id:
                    domestic_party_id.append(dom_prty_id)
                
                
                "We need to sort the data with respect to date"
                date = row[0].split('-')
                year,month,day = date
                
                
                "Converts date to day"
                year_month_days = month_day_fcn( int(month),int(year))
                DAYS = year_month_days + int(day)
                
                "We assign the date and value to the destination country"
                date_str = str(row[0])
                date_str = date_str.replace("-", "")
                date_int = float(date_str)
                
                
                "Some of the id numbers have negative values"
                id_number = str('id_{}'.format( np.abs(int(dom_prty_id))))
                #print(id_number)
                information_list = [DAYS, st_val, amount, date_int]
                try:
                    Dict_of_countries_with_data[dest_entry].append( information_list )
                    
                    Dict_of_ID_with_data[id_number].append( information_list )
                    
                except:
                    "if it doesnt exist. we create one"
                    Dict_of_countries_with_data[dest_entry] = [ information_list ]
                    
                    Dict_of_ID_with_data[id_number] = [ information_list ]

                
                
                "Total sales"
                Full_information.append( information_list )
             
                
                "no destination given"
                if dest_entry == '':
                    unknown_destination.append( [DAYS,st_val,amount, np.abs(int(dom_prty_id))] )
        
    
        "Sort the relevant dicts"
        Dict_of_countries_with_data = sort_dict(Dict_of_countries_with_data)
        Dict_of_ID_with_data        = sort_dict(Dict_of_ID_with_data)

        
        "Same sorting procedure"
        Full_information = np.array(Full_information)
        Full_info_sorted = Full_information[Full_information[:, 0].argsort()]
        
        "unknown: Sorted by dates"
        unknown_destination = np.array(unknown_destination)
        unknown_sorted      = unknown_destination[unknown_destination[:, 0].argsort()]
        
        "unknown: Sorted by id"
        #unknown_destination = np.array(unknown_destination)
        #unknown_sorted      = unknown_destination[unknown_destination[:, -1].argsort()]
        
        
     
    return Dict_of_countries_with_data, Full_info_sorted, \
            Dict_of_ID_with_data, unknown_sorted



"TEST CASE:"
"For selected countries"
def test(country_dict, country):
    
    "We can now perform statistics on the data"
    date = country_dict[:,0]
    revenue = country_dict[:,1]
    amount = country_dict[:,2]
    
    
    avg_revenue = np.average(revenue)
    std_revenue = np.std(revenue)
    
    avg_amount = np.average(amount)
    std_amount = np.std(amount)
    
    
    #plt.hist( norway_amount , bins = 100 )
    #plt.figure()
    #plt.hist( norway_revenue, bins = 100 )
    
    plt.scatter(  date, revenue )
    
    #s_A = np.sum(amount)
    #s_R = np.sum(revenue)
    
    revenue_ratio = avg_revenue / avg_amount
    
    
    "We want it to be greater than 1"
    print(type(avg_revenue))
    print(country, ": revenue:",  f'{ avg_revenue:.2f}',' amount:' , f'{avg_amount:.2f}' )
    print(revenue_ratio)
    return revenue_ratio




def compute_poisson(L,k):
    """
    L: Number of expected events
    k: Number of occured events
    """
    return ((L**k) * np.exp( -L )) / np.prod(np.arange(1,k))




"We collect snapshot data for our deep learning model to predict reasonable values"
def data_collector(X,Y,Y_2, country):
    
    Training_X, Training_Y, Training_Y_2 = [],[],[]
    
    "Stores data into a training set."
    try:
        Training_X[country].append(X)
        Training_Y[country].append(Y)
        Training_Y_2[country].append(Y_2)
    except:
        Training_X[country]=[X]
        Training_Y[country]=[Y]
        Training_Y_2[country]=[Y_2]
    
    return Training_X, Training_Y, Training_Y_2




"Perform cyclical normal distribution"
def test_cyclical(country_dict, country = None, scrutinize="COUNTRY", control="Revenue"):
    
    
    print()
    print("PROCESSING {}:".format(scrutinize), country)
    "We can now perform statistics on the data"
    k = 0 # our number of events placeholder
    days    = np.array( country_dict[:,0] )
    revenue = np.array( country_dict[:,1] )
    amount  = np.array( country_dict[:,2] )
    
    #date = country_dict[:,3]

    plot_ = False
    
    "We perform avg and std on a snapshot wrt time"
    "Larger snapshot for more chaotic but regular data"
    
    snapshot_length = 25
    if len(revenue) < snapshot_length:
        snapshot_length = len(revenue)
    else:
        pass
    
    
    for i in range(  len(revenue)-snapshot_length  ):
            
        "Grabs a snapshot of sequential data wrt days"
        snapshot_revenue = revenue[ i:i+snapshot_length ]
        snapshot_amount  = amount[  i:i+snapshot_length]
        snapshot_ratio = snapshot_revenue / snapshot_amount
        
        ratio = revenue / amount
        
        "Performs Mean and Standard deviation of snapshot data"
        avg_R = np.average(snapshot_revenue)
        std_R = np.std(snapshot_revenue)
        
        avg_A = np.average(snapshot_revenue)
        std_A = np.std(snapshot_revenue)
        
        
        "We normalize the snapshot data wrt Z-score"
        revenue_N = ( snapshot_revenue  - avg_R) / std_R
        amount_N  = ( snapshot_amount  - avg_A) / std_A
        ratio_N   = ( snapshot_ratio - np.average(snapshot_ratio) ) / np.std(snapshot_ratio)
        
        
        if control == "Revenue":
            check_value = revenue_N[0]
        if control == "Ratio":
            check_value = ratio_N[0]
        if control == "Amount":
            check_value = amount_N[0]
            
        
        "What is the likelihood of observing a value N#s above normal-distribution standard deviation"
        "We can use the number of occured events as training data"
        if check_value > 2:
            " ~5% chance to occur if above 2"
            " <1% chance to occur if above 3"
            #print( "date:", date[i], "entry:", i)
            #print("Something 'fishy' in country:", country, "at day:", days[i])
            #print("Revenue:", revenue[i], "amount:", amount[i])
            #print( "ratio:" , revenue[i] / amount[i])
            
            "We record the number of events"
            if k > 10:
                pass
            else:
                k += 1
            
            Y_nr_events = 1
            
        else:
            Y_nr_events = 0
        
        """
        Not in use
        
        if len(revenue) >= 2*snapshot_length:
            "We collect data"
            X = snapshot_revenue
            Y = revenue[ i + int(snapshot_length/2) : i + int(1.5*snapshot_length) ]
            data_collector( X,Y, Y_nr_events,country )
        else:
            "We dont collect any data"
            pass
        """
            
    
    "Poisson distribution"
    "We use the poisson distribution to determine the likelihood of number of events"
    if k > 0:
        L = 1
        p = compute_poisson(L,k)
        
        if p < 0.05:
            plot_ = True
            #print("Check country:", country) 
            #print("Probability of ",'{}'.format(k), "events occured in this interval:", "{:.2f}".format(p))
            
        
    if plot_ == True:
        
        if control == "Revenue":
            plt.scatter(days,revenue)
        if control == "Ratio":
            plt.scatter(days,ratio)
        
        plt.figure()
        plt.ylabel(control, fontsize=20)
        plt.xlabel("Days from 1 jan 2019", fontsize=20)
        plt.title("{}: {}".format(scrutinize,country), fontsize=20)








def compile_data(Dict_of_countries_with_data):
    "We compile a dataset"
    X = []
    Y = []
    for country in Dict_of_countries_with_data:
        
        "not every country have more than 25 timesteps"
        try:
            "Convert the country data to usable training data"
            "Data dimension: nr_batches, timesteps, feature_data"
            snapshot_length = 25
            country_data = Dict_of_countries_with_data[country][:,1]
            
            "We pick a random number between 0 and len(country_1) - 25"
            r = np.arange(0, len(country_data)-25)
            np.random.shuffle(r)
            
            r_list = list(r)
            
            for i in range(len(r)):
                "We pick this number as our starting integer for our snapshot"
                rv = r_list.pop(0)
                
                input_batch  = country_data[ rv : rv + snapshot_length ]
                target_batch = country_data[ rv + snapshot_length: rv + 2*snapshot_length ]
                
                if len(target_batch)< snapshot_length:
                    pass
                else:
                    X.append( input_batch  )
                    Y.append( target_batch )
                
                
                
                "TBC: compute z-score and poisson for current batch"
                "TBC: Compute reasonable output values, then compare with event value"
            
        except:
            pass
        
    return X,Y