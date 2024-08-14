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
from matplotlib.ticker import MaxNLocator

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

freq='10Min' #what time average data you want to be displayed on the plots

time, time_period = separate_str(freq)
##############################################################################################################################################################################################################################

lbl_size = 15
tick_size = 13

# Plot Details for the CO concentration vs Datetime plot
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.set_xlabel('DateTime', fontsize = lbl_size)
ax1.set_ylabel('CO Concentration (ppm)', fontsize = lbl_size)
ax1.tick_params(axis='both', which='major', labelsize=tick_size)
####
# Plot Details for reference vs sensor concentration plot
fig2, ax2 = plt.subplots(figsize=(7,5))
ax2.set_xlabel('Reference CO Concentration (ppm)', fontsize = lbl_size)
ax2.set_ylabel('Sensor CO Concentration (ppm)', fontsize = lbl_size)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)
####

fig3, ax3 = plt.subplots(figsize=(7,4))
ax3.set_xlabel('Sensor ID', fontsize = lbl_size)
ax3.set_ylabel('Slope', fontsize = lbl_size)
ax3.tick_params(axis='both', which='major', labelsize=tick_size)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

fig4, ax4 = plt.subplots(figsize=(7,4))
ax4.set_xlabel('Sensor ID', fontsize = lbl_size)
ax4.set_ylabel('$R^2$', fontsize = lbl_size)
ax4.tick_params(axis='both', which='major', labelsize=tick_size)
ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

##############################################################################################################################################################################################################################

# Chamber (Reference) Data

time_avg = '1min' #1min or 5min
group_num = 3
test_num = 1 # 1, 2, 3, or 4 

data_dir = os.path.abspath('../../../Data/CO_Testing_2024/chamber_data/group3/')

group_num = 'group' + str(group_num)
test_num = 'test' + str(test_num)

flnm = test_num + '_' + group_num + '_' + time_avg + '.csv'
flnm_dir = os.path.join(data_dir,flnm)

chamber_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                 dtype={'Date_CT':str, 'Time_CT':str ,'CO_ppm':float},
                                 usecols = [0, 1, 2], skiprows=2,
                                 names = ['Date_CT', 'Time_CT', 'CO_ppm']))

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

begin = '2024-08-02 15:50:00'
end = '2024-08-03 02:50:00'
chamber_df = chamber_df[~(chamber_df['DateTime_CT'] < begin)]
chamber_df = chamber_df[~(chamber_df['DateTime_CT'] >= end)]

chamber_df = chamber_df.set_index(keys='DateTime_CT', drop=True)

# #get rid of the random -9999 values when for whatever reason it wasn't working
# chamber_df = chamber_df[chamber_df['T_c'] != -9999]
# chamber_df = chamber_df[chamber_df['RH_percent'] != -9999]  

chamber_df = chamber_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')
# print(chamber_df)

n = len(chamber_df)
ax1.plot(chamber_df, color='black', label = 'Reference', linewidth=5.0)

##############################################################################################################################################################################################################################

# QuantAQ Data

time_avg = '1min' #1min or 5min
group_num = 3
test_num = 1 # 1, 2, 3, or 4 

data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group3/quantaq/')

group_num = 'group' + str(group_num)
test_num = 'test' + str(test_num)

slopes = []
r_sqrs = []
sensor_nm = ['MOD-00751', 'MOD-00752', 'MOD-00753']
color = plt.cm.gray(np.linspace(0.3,0.7,len(sensor_nm)))
for i in range(len(sensor_nm)):
    flnm = test_num + '_' + group_num + '_' + sensor_nm[i] + '_' + time_avg + '.csv'
    flnm_dir = os.path.join(data_dir, flnm)

    qaq_df = (pd.read_csv(filepath_or_buffer=flnm_dir,
                                 dtype={'DateTime_lst':str , 'RH_percent': float, 'T_c':float, 'CO_ppm':float},
                                 usecols = [2, 4, 5, 12], skiprows=2,
                                 names = ['DateTime_lst', 'RH_percent', 'T_c', 'CO_ppm']))
    qaq_df = qaq_df.dropna()

    # Raw data is in ppb, so changing to ppm
    qaq_df['CO_ppm'] = qaq_df['CO_ppm']/1000

    idx_fmt = '%Y-%m-%dT%H:%M:%SZ'
    qaq_df['DateTime_lst'] = pd.to_datetime(qaq_df['DateTime_lst'], format=idx_fmt)

    qaq_df['DateTime_lst'] = qaq_df['DateTime_lst'] - pd.Timedelta(hours=1)
    begin = '2024-08-02 15:50:00'
    end = '2024-08-03 02:50:00'
    qaq_df = qaq_df[~(qaq_df['DateTime_lst'] < begin)]
    qaq_df = qaq_df[~(qaq_df['DateTime_lst'] >= end)]

    qaq_df = qaq_df.set_index(keys='DateTime_lst', drop=True)
    qaq_df = qaq_df.sort_values(by='DateTime_lst')

    qaq_df = qaq_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')
    # print(qaq_df)

    slope, intercept, r, p, se = linregress(chamber_df, qaq_df)
    r_sqr = r**2
    slopes.append(slope)
    r_sqrs.append(r_sqr)

    ax1.plot(qaq_df, color=color[i], marker='.', label = 'QuantAQ ' + sensor_nm[i])
    ax2.plot(chamber_df, qaq_df, label = 'QuantAQ ' + sensor_nm[i], marker = 'X', linestyle='None', color=color[i])

qaq_nums = [1, 2, 3]
ax3.plot(qaq_nums, slopes, label = 'QuantAQ', marker = 'X', linestyle='None', color='grey')
ax4.plot(qaq_nums, r_sqrs, label = 'QuantAQ', marker = 'X', linestyle='None', color='grey')

##############################################################################################################################################################################################################################

# Other figure formatting
hours = mdates.HourLocator(interval = 3)
ax1.xaxis.set_major_locator(hours)
ax1.xaxis.set_major_formatter(DateFormatter("%I:%M%p\n%m/%d/%Y"))

ax1.set_title(f'Concentration Ramp 1 ({time}-{time_period} averaged), N = {n}', fontsize="15", weight='bold')

ax1.legend()
fig1.tight_layout()

#######
lims = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax2.plot(lims, lims, 'k-', zorder=0, label='1:1', linewidth=5.0)
ax2.grid(visible='True')
ax2.set_title(f'Concentration Ramp 1 ({time}-{time_period} averaged), N = {n}', fontsize="15", weight='bold')
ax2.legend()
fig2.tight_layout()

####

ax3.set_ylim(0,2.2)
ax3.axhspan(0.8,1.2, alpha=0.5, color='gray')
ax3.legend(fontsize="15", loc='lower left')
fig3.tight_layout()

####

ax4.set_ylim(0.6,1)
ax4.axhspan(0.8,1, alpha=0.5, color='gray')
ax4.legend(fontsize="15", loc='lower left')
fig4.tight_layout()

##############################################################################################################################################################################################################################
# Saving Plots
analyzed_dir = os.path.abspath('../../../Analyzed/group3/')
fig_flnm = f'{test_num}_{group_num}_{freq}-avg.png'
fig1.savefig(os.path.join(analyzed_dir, fig_flnm), format='png')

#######
fig_flnm2 = f'{test_num}_{group_num}_{freq}-avg_sensor-ref.png'
fig2.savefig(os.path.join(analyzed_dir, fig_flnm2), format='png')

#######
fig_flnm = f'{test_num}_{group_num}_{freq}_slopes.png'
fig3.savefig(os.path.join(analyzed_dir,fig_flnm),format='png')

#######
fig_flnm = f'{test_num}_{group_num}_{freq}_rsqrs.png'
fig4.savefig(os.path.join(analyzed_dir,fig_flnm),format='png')