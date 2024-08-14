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
begin = '2024-06-27 17:00:00'
end = '2024-06-28 23:01:00'

freq='10Min' #what time average data you want to be displayed on the plots

time, time_period = separate_str(freq)
##############################################################################################################################################################################################################################

lbl_size = 15
tick_size = 13

# Plot Details for the CO concentration vs Datetime plot
fig1, ax1 = plt.subplots(figsize=(12,6))
ax1.set_xlabel('DateTime', fontsize = lbl_size)
ax1.set_ylabel('CO Concentration (ppm)', fontsize = lbl_size)
ax1.set_title(f'Effect of RH and T at ~5ppm CO ({time} {time_period} averaged)', fontsize = tick_size)

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
ax2.set_title(f'Effect of RH and T ({time}-{time_period} averaged) at ~5ppm of CO', fontsize="15", weight='bold')

##############################################################################################################################################################################################################################

# Chamber (Reference) Data

time_avg = '1min' #1min or 5min
group_num = 1
test_num = 2 # 1, 2, 3, or 4 

data_dir = os.path.abspath('../../../Data/CO_Testing_2024/chamber_data/group1/')

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

# BlueSky Data

sensor_nm = ['5009', '5021', '5050']
color = plt.cm.winter(np.linspace(0.3,1,len(sensor_nm)))
#color2 = plt.cm.winter(np.linspace(0,1,len(sensor_nm)))

for i in range(len(sensor_nm)):
    data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group1/bluesky/BlueSky 8145 ' + sensor_nm[i] + '/')

    flnm = '814500231' + sensor_nm[i] + '-' + '202426' + '.csv'
    flnm_dir = os.path.join(data_dir, flnm)

    flnm2 = '814500231' + sensor_nm[i] + '-' + '202427' + '.csv'
    flnm_dir2 = os.path.join(data_dir, flnm2)

    bs1_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                 dtype={'DateTime_utc':str , 'CO_ppm':float},
                                 usecols = [0, 14], skiprows=59,
                                 names = ['DateTime_utc', 'CO_ppm']))
    
    bs2_df = (pd.read_csv(filepath_or_buffer=flnm_dir2,
                                 dtype={'DateTime_utc':str , 'RH_percent': float, 'T_c':float, 'CO_ppm':float},
                                 usecols = [0, 14], skiprows=59,
                                 names = ['DateTime_utc', 'CO_ppm']))

    bs_df = pd.concat([bs1_df, bs2_df], axis=0)

    idx_fmt = '%Y-%m-%d %H:%M:%S'
    bs_df['DateTime_utc'] = pd.to_datetime(bs_df['DateTime_utc'], format=idx_fmt)
    # removing the seconds in the datetime
    bs_df['DateTime_utc'] = bs_df['DateTime_utc'].dt.floor('T')

    bs_df['DateTime_utc'] = bs_df['DateTime_utc'] - pd.Timedelta(hours=5)
    bs_df = bs_df[~(bs_df['DateTime_utc'] < begin)]
    bs_df = bs_df[~(bs_df['DateTime_utc'] >= end)]

    bs_df = bs_df.set_index(keys='DateTime_utc', drop=True)


    error_df = abs(chamber_df['CO_ppm'].subtract(bs_df['CO_ppm']))
    error_df = error_df.groupby([pd.Grouper(freq=freq)]).agg('mean')

    begin_sub = ['2024-06-27 18:00:00', '2024-06-27 20:00:00', '2024-06-27 22:00:00', '2024-06-28 04:00:00', '2024-06-28 06:00:00', '2024-06-28 08:00:00']
    end_sub = ['2024-06-27 19:00:00', '2024-06-27 21:00:00', '2024-06-27 23:00:00', '2024-06-28 05:00:00', '2024-06-28 07:00:00', '2024-06-28 09:00:00']
    mean_errors = []
    for j in range(len(begin_sub)):
        val1 = error_df[~(error_df.index < begin_sub[j])]
        val1 = val1[~(val1.index >= end_sub[j])]
        mean_error = val1.sum()/len(val1.index)
        mean_errors.append(mean_error)

    rhs = [30, 50, 70]


    bs_df = bs_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

    ax1.plot(bs_df, color=color[i], linestyle='None', marker='.', label = 'BlueSky ' + sensor_nm[i])


    ax2.plot(rhs, mean_errors[:3], label = 'BlueSky ' + sensor_nm[i] + '; 25C', marker = '.', color=color[i])
    ax2.plot(rhs, mean_errors[-3:], label = 'BlueSky ' + sensor_nm[i] + '; 35C', marker = '^', markersize=12, color=color[i], linestyle = "dashed")

##############################################################################################################################################################################################################################

# QuantAQ Data

time_avg = '1min' #1min or 5min
group_num = 1
test_num = 2 # 1, 2, 3, or 4 

data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group1/quantaq/')

group_num = 'group' + str(group_num)
test_num = 'test' + str(test_num)

sensor_nm = ['MOD-00751', 'MOD-00752', 'MOD-00753']
color = plt.cm.gray(np.linspace(0.4,0.8,len(sensor_nm)))
for i in range(len(sensor_nm)):
    flnm = test_num + '_' + group_num + '_' + sensor_nm[i] + '_' + time_avg + '.csv'
    flnm_dir = os.path.join(data_dir, flnm)

    qaq_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                 dtype={'DateTime_lst':str, 'CO_ppm':float},
                                 usecols = [2, 12], skiprows=2,
                                 names = ['DateTime_lst', 'CO_ppm']))
    qaq_df = qaq_df.dropna()

    # Raw data is in ppb, so changing to ppm
    qaq_df['CO_ppm'] = qaq_df['CO_ppm']/1000

    idx_fmt = '%Y-%m-%dT%H:%M:%SZ'
    qaq_df['DateTime_lst'] = pd.to_datetime(qaq_df['DateTime_lst'], format=idx_fmt)
    
    # removing the seconds in the datetime
    qaq_df['DateTime_lst'] = qaq_df['DateTime_lst'].dt.floor('T')

    qaq_df['DateTime_lst'] = qaq_df['DateTime_lst'] - pd.Timedelta(hours=1)
    qaq_df = qaq_df[~(qaq_df['DateTime_lst'] < begin)]
    qaq_df = qaq_df[~(qaq_df['DateTime_lst'] >= end)]

    qaq_df = qaq_df.set_index(keys='DateTime_lst', drop=True)
    qaq_df = qaq_df.sort_values(by='DateTime_lst')

    error_df = abs(chamber_df['CO_ppm'].subtract(qaq_df['CO_ppm']))
    error_df = error_df.groupby([pd.Grouper(freq=freq)]).agg('mean')


    begin_sub = ['2024-06-27 18:00:00', '2024-06-27 20:00:00', '2024-06-27 22:00:00', '2024-06-28 04:00:00', '2024-06-28 06:00:00', '2024-06-28 08:00:00']
    end_sub = ['2024-06-27 19:00:00', '2024-06-27 21:00:00', '2024-06-27 23:00:00', '2024-06-28 05:00:00', '2024-06-28 07:00:00', '2024-06-28 09:00:00']
    mean_errors = []
    for j in range(len(begin_sub)):
        val1 = error_df[~(error_df.index < begin_sub[j])]
        val1 = val1[~(val1.index >= end_sub[j])]
        mean_error = val1.sum()/len(val1.index)
        mean_errors.append(mean_error)

    rhs = [30, 50, 70]

    qaq_df = qaq_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

    ax1.plot(qaq_df, color=color[i], linestyle='None', marker='x', label = 'QuantAQ ' + sensor_nm[i])

    ax2.plot(rhs, mean_errors[:3], label = 'QuantAQ ' + sensor_nm[i] + '; 25C', marker = '.', color=color[i])
    ax2.plot(rhs, mean_errors[-3:], label = 'QuantAQ ' + sensor_nm[i] + '; 35C', marker = '^', markersize=12, color=color[i], linestyle = "dashed")

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
analyzed_dir = os.path.abspath('../../../Analyzed/group1/')
fig_flnm = f'{test_num}_{group_num}_{freq}-avg.png'
fig1.savefig(os.path.join(analyzed_dir, fig_flnm), format='png')

####

fig_flnm2 = f'{test_num}_{group_num}_{freq}-avg_mean-error.png'
fig2.savefig(os.path.join(analyzed_dir, fig_flnm2), format='png')