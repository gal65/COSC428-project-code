# Author: Gordon Lay
# May 2021

# Compares ground truth and recognition data spreadsheets line by line 
# and prints matching readings as a fraction of the total number of readings

import csv

total_frames = 521 # take this from the output of show_frames.py
count = 0

with open('ground_truth_data.csv', 'r') as t1, open('recognition_data.csv', 'r') as t2:
    line = 0
    while True:
        lineT1 = t1.readline().strip()
        lineT2 = t2.readline().strip()
        line += 1
     
        if (lineT1 or lineT2):
            if (lineT1 == lineT2):
                count += 1
        else:
            t1.close()
            t2.close()
            break
                    
print('Shot Recognition Accuracy: ', count / total_frames)