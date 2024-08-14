import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as st
import sys
import pandas as pd
import pytz as pytz
from scipy.stats import linregress
from datetime import datetime, timezone
from matplotlib.dates import MonthLocator, DateFormatter
import matplotlib.dates as mdates

def separate_str(str):
    # Create empty strings for storing the alphabets and numbers
    numbers = ''
    alphabets = ''
     
    # Iterate through each character in the given string
    for char in str:
        # Check if the character is an alphabet
        if char.isalpha():
            # If it is an alphabet, append it to the alphabets string
            alphabets += char
        # Check if the character is a number
        elif char.isnumeric():
            # If it is a number, append it to the numbers string
            numbers += char
    
    return numbers, alphabets
##############################################################################################################################################################################################################################

# Test 2
begin = '2024-07-07 08:20:00'
end = '2024-07-08 14:20:00'

freq='10Min' #what time average data you want to be displayed on the plots

time, time_period = separate_str(freq)
##############################################################################################################################################################################################################################

lbl_size = 15
tick_size = 13

# Plot Details for the CO concentration vs Datetime plot
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.set_xlabel('DateTime', fontsize = lbl_size)
ax1.set_ylabel('CO Concentration (ppm)', fontsize = lbl_size)
ax1.set_title(f'Effect of RH and T at ~10ppm CO ({time} {time_period} averaged)', fontsize = tick_size)

twin1 = ax1.twinx() #RH
twin2 = ax1.twinx() #Temp
twin2.spines.right.set_position(("axes", 1.1))
twin1.set_ylabel("Relative Humidity (%)", fontsize = lbl_size)
twin2.set_ylabel("Temperature (°C)", fontsize = lbl_size)
twin1.yaxis.label.set_color('blue')
twin2.yaxis.label.set_color('red')
twin1.set_ylim(20, 80)
twin2.set_ylim(24, 36)

# Plot Details for the CO mean error vs humidity
#fig2, ax2 = plt.subplots(figsize=(12,6))
fig2, ax2 = plt.subplots(figsize=(12,8))
ax2.set_xlabel('Relative Humidity (%)', fontsize = lbl_size)
ax2.set_ylabel('CO Mean Error (ppm)', fontsize = lbl_size)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)
ax2.set_title(f'Effect of RH and T ({time}-{time_period} averaged) at ~10ppm of CO', fontsize="15", weight='bold')

##############################################################################################################################################################################################################################

# Chamber (Reference) Data

time_avg = '1min' #1min or 5min
group_num = 2
test_num = 3 # 1, 2, 3, or 4 

data_dir = os.path.abspath('../../../Data/CO_Testing_2024/chamber_data/group2/')

group_num = 'group' + str(group_num)
test_num = 'test' + str(test_num)

flnm = test_num + '_' + group_num + '_' + time_avg + '.csv'
flnm_dir = os.path.join(data_dir,flnm)

chamber_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                 dtype={'Date_CT':str, 'Time_CT':str ,'CO_ppm':float, 'T_c':float, 'RH_percent': float},
                                 usecols = [0, 1, 2, 4, 6], skiprows=2,
                                 names = ['Date_CT', 'Time_CT', 'CO_ppm', 'T_c', 'RH_percent']))

#add leading zeros to the date and time columns because for some reason it needs it
chamber_df['Date_CT'] = chamber_df['Date_CT'].str.replace(r'\b(\d)\b', r'0\1', regex=True) #https://stackoverflow.com/questions/77313012/how-to-add-leading-zero-to-day-and-month-in-pandas-core-series-series-object
chamber_df['Time_CT'] = chamber_df['Time_CT'].str.zfill(8) #https://stackoverflow.com/questions/68609499/pandas-check-column-and-add-leading-zeros

#combine date and time columns
chamber_df['DateTime_CT'] = chamber_df['Date_CT'] + ' ' + chamber_df['Time_CT']
chamber_df = chamber_df.drop(labels=['Date_CT'], axis=1)
chamber_df = chamber_df.drop(labels=['Time_CT'], axis=1)

#change the datetime col to a pandas datetime
idx_fmt = '%m/%d/%Y %I:%M %p'
chamber_df['DateTime_CT'] = pd.to_datetime(chamber_df['DateTime_CT'], format=idx_fmt)

# removing the seconds in the datetime
chamber_df['DateTime_CT'] = chamber_df['DateTime_CT'].dt.floor('T')

chamber_df = chamber_df[~(chamber_df['DateTime_CT'] < begin)]
chamber_df = chamber_df[~(chamber_df['DateTime_CT'] >= end)]

chamber_df = chamber_df.set_index(keys='DateTime_CT', drop=True)

#get rid of the random -9999 values when for whatever reason it wasn't working
chamber_df = chamber_df[chamber_df['T_c'] != -9999]
chamber_df = chamber_df[chamber_df['RH_percent'] != -9999]  

chamber_conc = chamber_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')
chamber_temp = chamber_df['T_c'].groupby([pd.Grouper(freq=freq)]).agg('mean')
chamber_rh = chamber_df['RH_percent'].groupby([pd.Grouper(freq=freq)]).agg('mean')

#ax1.plot(chamber_datetime, chamber_co, color='black', label = 'Analyzer (Teledyne T300U)')
ax1.plot(chamber_conc, color='black', label = 'Analyzer (Teledyne T300U)')
twin2.plot(chamber_temp, color='red', label = 'Temperature')
twin1.plot(chamber_rh, color='blue', label = 'Relative Humidity')

##############################################################################################################################################################################################################################

# Lascar Data

time_avg = '1min' #1min or 5min
group_num = 2
test_num = 3 # 1, 2, 3, or 4 

group_num = 'group' + str(group_num)
test_num = 'test' + str(test_num)

sensor_flnm = ['Chamber_30458', 'SN_30433', 'USB64_30391']
sensor_nm = ['30458', '30433', '30391']
# sensor_flnm = ['Chamber_30458', 'SN_30433']
# sensor_nm = ['30458', '30433']
color = plt.cm.gray(np.linspace(0.2,0.7,len(sensor_nm)))
for i in range(len(sensor_nm)):
    data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group2/lascar/')

    flnm = sensor_flnm[i] + '.txt'
    flnm_dir = os.path.join(data_dir, flnm)

    lcar_df = pd.read_csv(filepath_or_buffer=flnm_dir, 
                          delimiter=",", usecols=[1,2], skiprows=1, 
                          names=['DateTime', 'CO_ppm'])

    idx_fmt = '%Y-%m-%d %H:%M:%S'
    lcar_df['DateTime'] = pd.to_datetime(lcar_df['DateTime'], format=idx_fmt)

    # removing the seconds in the datetime
    lcar_df['DateTime'] = lcar_df['DateTime'].dt.floor('T')

    if sensor_nm[i] == '30391':
        lcar_df['DateTime'] = lcar_df['DateTime'] - pd.Timedelta(hours=3)
    else:
        lcar_df['DateTime'] = lcar_df['DateTime'] - pd.Timedelta(hours=1)
    # lcar_df['DateTime'] = lcar_df['DateTime'] - pd.Timedelta(hours=1)
    lcar_df = lcar_df[~(lcar_df['DateTime'] < begin)]
    lcar_df = lcar_df[~(lcar_df['DateTime'] >= end)]

    lcar_df = lcar_df.set_index(keys='DateTime', drop=True)

    error_df = abs(lcar_df['CO_ppm'].subtract(chamber_conc))
    error_df = error_df.groupby([pd.Grouper(freq=freq)]).agg('mean')

    begin_sub = ['2024-07-07 09:20:00', '2024-07-07 11:20:00', '2024-07-07 13:20:00', '2024-07-07 19:20:00', '2024-07-07 21:20:00', '2024-07-07 23:20:00']
    end_sub = ['2024-07-07 10:20:00', '2024-07-07 12:20:00', '2024-07-07 14:20:00', '2024-07-07 20:20:00', '2024-07-07 22:20:00', '2024-07-08 00:20:00']
    mean_errors = []
    for j in range(len(begin_sub)):
        val1 = error_df[~(error_df.index < begin_sub[j])]
        val1 = val1[~(val1.index >= end_sub[j])]
        mean_error = val1.sum()/len(val1.index)
        mean_errors.append(mean_error)

    rhs = [30, 50, 70]

    lcar_df = lcar_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')
    ax1.plot(lcar_df, color=color[i], linestyle='None', marker='.', label = 'Lascar ' + sensor_nm[i])

    ax2.plot(rhs, mean_errors[:3], label = 'Lascar ' + sensor_nm[i] + '; 25C', marker = '.', color=color[i])
    ax2.plot(rhs, mean_errors[-3:], label = 'Lascar ' + sensor_nm[i] + '; 35C', marker = '^', markersize=12, color=color[i], linestyle = "dashed")

##############################################################################################################################################################################################################################

# Canary Data

# sensor_nm = ['1', '3', '4']
sensor_nm = ['4']
#color = plt.cm.gray(np.linspace(0.2,0.7,len(sensor_nm)))
for i in range(len(sensor_nm)):
    data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group2/canary/')

    # flnm = 'LOGGER1_Canary-00' + sensor_nm[i] + '.txt'
    flnm = 'Canary00' + sensor_nm[i] + '.csv'
    flnm_dir = os.path.join(data_dir, flnm)

    can_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                 dtype={'DateTime': str, 'CO_ppm':float},
                                 usecols = [1, 5], skiprows=1,
                                 names = ['DateTime', 'CO_ppm']))

    idx_fmt = '%Y-%m-%d %H:%M:%S'
    can_df['DateTime'] = pd.to_datetime(can_df['DateTime'], format=idx_fmt)

    # removing the seconds in the datetime
    can_df['DateTime'] = can_df['DateTime'].dt.floor('T')

    can_df['DateTime'] = can_df['DateTime'] - pd.Timedelta(hours = 4 )

    can_df = can_df[~(can_df['DateTime'] < begin)]
    can_df = can_df[~(can_df['DateTime'] >= end)]

    can_df = can_df.set_index(keys='DateTime', drop=True)

    error_df = abs(can_df['CO_ppm'].subtract(chamber_conc))
    error_df = error_df.groupby([pd.Grouper(freq=freq)]).agg('mean')

    begin_sub = ['2024-07-07 09:20:00', '2024-07-07 11:20:00', '2024-07-07 13:20:00', '2024-07-07 19:20:00', '2024-07-07 21:20:00', '2024-07-07 23:20:00']
    end_sub = ['2024-07-07 10:20:00', '2024-07-07 12:20:00', '2024-07-07 14:20:00', '2024-07-07 20:20:00', '2024-07-07 22:20:00', '2024-07-08 00:20:00']
    mean_errors = []
    for j in range(len(begin_sub)):
        val1 = error_df[~(error_df.index < begin_sub[j])]
        val1 = val1[~(val1.index >= end_sub[j])]
        mean_error = val1.sum()/len(val1.index)
        mean_errors.append(mean_error)

    rhs = [30, 50, 70]

    can_df = can_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

    ax1.plot(can_df, color='green', linestyle='None', marker='.', label = 'Canary ' + sensor_nm[i])

    ax2.plot(rhs, mean_errors[:3], label = 'Canary ' + sensor_nm[i] + '; 25C', marker = '.', color='green')
    ax2.plot(rhs, mean_errors[-3:], label = 'Canary ' + sensor_nm[i] + '; 35C', marker = '^', markersize=12, color='green', linestyle = "dashed")

##############################################################################################################################################################################################################################

# VAMMS Data

# sensor_nm = ['1', '3', '4']
sensor_nm = ['1', '2', '3', '4', '5', '6', '7']
slopes = []
r_sqrs = []
#color = plt.cm.gray(np.linspace(0.2,0.7,len(sensor_nm)))
data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group2/VAMMS/')

vamms_df = pd.DataFrame()
for i in range(len(sensor_nm)):
    flnm = '1519010' + sensor_nm[i] + '.CSV'
    flnm_dir = os.path.join(data_dir, flnm)

    vamm_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                    dtype={'DateTime_UTC': str, 'CO_ppm':float},
                                    usecols = [0, 9], skiprows=1,
                                    names = ['DateTime_UTC', 'CO_ppm']))

    vamms_df = pd.concat([vamms_df, vamm_df], axis=0)

vamms_df['CO_ppm'] = vamms_df['CO_ppm']/10

idx_fmt = '%Y-%m-%dT%H:%M:%S-00:00'
vamms_df['DateTime_UTC'] = pd.to_datetime(vamms_df['DateTime_UTC'], format=idx_fmt)

# removing the seconds in the datetime
vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'].dt.floor('T')

vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'] + pd.Timedelta(days = 2010)
vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'] + pd.Timedelta(hours = 11)
vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'] + pd.Timedelta(minutes = 51)

vamms_df = vamms_df[~(vamms_df['DateTime_UTC'] < begin)]
vamms_df = vamms_df[~(vamms_df['DateTime_UTC'] >= end)]

vamms_df = vamms_df.set_index(keys='DateTime_UTC', drop=True)

error_df = abs(vamms_df['CO_ppm'].subtract(chamber_conc))
error_df = error_df.groupby([pd.Grouper(freq=freq)]).agg('mean')

begin_sub = ['2024-07-07 09:20:00', '2024-07-07 11:20:00', '2024-07-07 13:20:00', '2024-07-07 19:20:00', '2024-07-07 21:20:00', '2024-07-07 23:20:00']
end_sub = ['2024-07-07 10:20:00', '2024-07-07 12:20:00', '2024-07-07 14:20:00', '2024-07-07 20:20:00', '2024-07-07 22:20:00', '2024-07-08 00:20:00']
mean_errors = []
for j in range(len(begin_sub)):
    val1 = error_df[~(error_df.index < begin_sub[j])]
    val1 = val1[~(val1.index >= end_sub[j])]
    mean_error = val1.sum()/len(val1.index)
    mean_errors.append(mean_error)

rhs = [30, 50, 70]

vamms_df = vamms_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')
ax1.plot(vamms_df, color='orange', linestyle='None', marker='.', label = 'VAMMS')

ax2.plot(rhs, mean_errors[:3], label = 'VAMMS' + '; 25C', marker = '.', color='orange')
ax2.plot(rhs, mean_errors[-3:], label = 'VAMMS' + '; 35C', marker = '^', markersize=12, color='orange', linestyle = "dashed")

##############################################################################################################################################################################################################################

# Other figure formatting
hours = mdates.HourLocator(interval = 4)
ax1.xaxis.set_major_locator(hours)
ax1.xaxis.set_major_formatter(DateFormatter("%I:%M%p\n%m/%d/%Y"))

ax1.legend(loc = 'upper left')
twin1.legend(loc = 'upper right')
twin2.legend(loc = 'upper right', bbox_to_anchor=(1, 0.93))
fig1.tight_layout()

####
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig2.tight_layout()

##############################################################################################################################################################################################################################
# Saving Plots
analyzed_dir = os.path.abspath('../../../Analyzed/group2/')
fig_flnm = f'{test_num}_{group_num}_{freq}-avg.png'
fig1.savefig(os.path.join(analyzed_dir, fig_flnm), format='png')

####

fig_flnm2 = f'{test_num}_{group_num}_{freq}-avg_mean-error.png'
fig2.savefig(os.path.join(analyzed_dir, fig_flnm2), format='png')