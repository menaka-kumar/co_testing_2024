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

begin = '2024-07-03 14:51:00'
end = '2024-07-04 04:21:00'
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
#
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
group_num = 2
test_num = 1 # 1, 2, 3, or 4 

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

chamber_df = chamber_df[~(chamber_df['DateTime_CT'] < begin)]
chamber_df = chamber_df[~(chamber_df['DateTime_CT'] >= end)]

chamber_df = chamber_df.set_index(keys='DateTime_CT', drop=True)

#get rid of the random -9999 values when for whatever reason it wasn't working
chamber_df = chamber_df[chamber_df['T_c'] != -9999]
chamber_df = chamber_df[chamber_df['RH_percent'] != -9999]  

chamber_df = chamber_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

n = len(chamber_df)
ax1.plot(chamber_df, color='black', label = 'Reference', linewidth=5.0)

##############################################################################################################################################################################################################################

# LASCAR Data

time_avg = '1min' #1min or 5min
group_num = 2
test_num = 1 # 1, 2, 3, or 4 

group_num = 'group' + str(group_num)
test_num = 'test' + str(test_num)

sensor_flnm = ['Chamber_30458', 'SN_30433', 'USB64_30391']
sensor_nm = ['30458', '30433', '30391']
# sensor_flnm = ['Chamber_30458', 'SN_30433']
# sensor_nm = ['30458', '30433']
slopes = []
r_sqrs = []
color = plt.cm.gray(np.linspace(0.2,0.7,len(sensor_nm)))
for i in range(len(sensor_nm)):
    data_dir = os.path.abspath('../../../Data/CO_Testing_2024/sensor_data/group2/lascar/')

    flnm = sensor_flnm[i] + '.txt'
    flnm_dir = os.path.join(data_dir, flnm)

    lcar_df=pd.read_csv(filepath_or_buffer=flnm_dir, delimiter=",", usecols=[1,2], skiprows=1, names=['DateTime', 'CO_ppm'])

    idx_fmt = '%Y-%m-%d %H:%M:%S'
    lcar_df['DateTime'] = pd.to_datetime(lcar_df['DateTime'], format=idx_fmt)

    if sensor_nm[i] == '30391':
        lcar_df['DateTime'] = lcar_df['DateTime'] - pd.Timedelta(hours=1)
    else:
        lcar_df['DateTime'] = lcar_df['DateTime'] - pd.Timedelta(hours=1)
    # lcar_df['DateTime'] = lcar_df['DateTime'] - pd.Timedelta(hours=1)
    lcar_df = lcar_df[~(lcar_df['DateTime'] < begin)]
    lcar_df = lcar_df[~(lcar_df['DateTime'] >= end)]

    lcar_df = lcar_df.set_index(keys='DateTime', drop=True)

    lcar_df = lcar_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

    slope, intercept, r, p, se = linregress(chamber_df, lcar_df)
    r_sqr = r**2
    slopes.append(slope)
    r_sqrs.append(r_sqr)

    ax1.plot(lcar_df, color=color[i], marker='.', label = 'Lascar ' + sensor_nm[i])
    ax2.plot(chamber_df, lcar_df, label = 'Lascar ' + sensor_nm[i], marker = 'X', linestyle='None', color=color[i])

ls_nums = [1, 2, 3]
ax3.plot(ls_nums, slopes, label = 'Lascar', marker = 'X', linestyle='None', color='grey')
ax4.plot(ls_nums, r_sqrs, label = 'Lascar', marker = 'X', linestyle='None', color='grey')

##############################################################################################################################################################################################################################

# Canary Data

# sensor_nm = ['1', '3', '4']
sensor_nm = ['4']
slopes = []
r_sqrs = []
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

    can_df['DateTime'] = can_df['DateTime'] - pd.Timedelta(hours = 4)
    can_df = can_df[~(can_df['DateTime'] < begin)]
    can_df = can_df[~(can_df['DateTime'] >= end)]

    can_df = can_df.set_index(keys='DateTime', drop=True)

    can_df = can_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

    slope, intercept, r, p, se = linregress(chamber_df, can_df)
    r_sqr = r**2
    slopes.append(slope)
    r_sqrs.append(r_sqr)

    ax1.plot(can_df, color='green', marker='.', label = 'Canary ' + sensor_nm[i])
    ax2.plot(chamber_df, can_df, label = 'Canary ' + sensor_nm[i], marker = 'X', linestyle='None', color='green')

can_nums = [4]
ax3.plot(can_nums, slopes, label = 'Canary', marker = 'X', linestyle='None', color='green')
ax4.plot(can_nums, r_sqrs, label = 'Canary', marker = 'X', linestyle='None', color='green')

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


vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'] + pd.Timedelta(days = 2010)
vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'] + pd.Timedelta(hours = 11)
vamms_df['DateTime_UTC'] = vamms_df['DateTime_UTC'] + pd.Timedelta(minutes = 51)

vamms_df = vamms_df[~(vamms_df['DateTime_UTC'] < begin)]
vamms_df = vamms_df[~(vamms_df['DateTime_UTC'] >= end)]

vamms_df = vamms_df.set_index(keys='DateTime_UTC', drop=True)

vamms_df = vamms_df['CO_ppm'].groupby([pd.Grouper(freq=freq)]).agg('mean')

slope, intercept, r, p, se = linregress(chamber_df, vamms_df)
r_sqr = r**2
slopes.append(slope)
r_sqrs.append(r_sqr)

ax1.plot(vamms_df, color='orange', marker='.', label = 'VAMMS')
ax2.plot(chamber_df, vamms_df, label = 'VAMMS', marker = 'X', linestyle='None', color='orange')

vamms_nums = [0]
ax3.plot(vamms_nums, slopes, label = 'VAMMS', marker = 'X', linestyle='None', color='orange')
ax4.plot(vamms_nums, r_sqrs, label = 'VAMMS', marker = 'X', linestyle='None', color='orange')


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
analyzed_dir = os.path.abspath('../../../Analyzed/group2/')
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