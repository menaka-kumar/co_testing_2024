08/14/24

Canary Folder: Placed 3 canary sensors in the chamber. Sensors 001 and 003 would not connect online. The files for sensors 1, 3, and 4 "LOGGER1_Canary-001.CSV", "LOGGER1_Canary-003.CSV", "LOGGER1_Canary-004.CSV"  were from the SD cards in the sensors. However, for some reason it not have a column for CO data for some reason. Sensor 004 was the only one that could connect online so I downloaded data from the dashboard that is "Canary004.csv". This file contains data for tests 1-4. 

Lascar Folder: Places 3 Lascars in the chamber. These files contain data for each sensor ID for tests 1-4.

VAMMS Folder: Placed 1 VAMMS in the chamber. The data for tests 1-4 is split into many different files. One thing to note is that since the VAMMS could not get a GPS signal while in the chamber it could not record an accurate time. So that data is off. To get it to match with the time that the chamber displays (Eastern without daylight savings) I added 2010 days, 11 hours, and 51 minutes to the DateTimes.