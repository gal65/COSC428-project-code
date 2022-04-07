# Author: Gordon Lay
# May 2021

# Generates ground_truth_data.csv 
# Manually set intervals 
# Iterate through and assign respective shot type to each interval

import csv
import pandas as pd

total_frames = 521

ground_truth_array = []


for i in range(0, total_frames):
    if i < 36:
        ground_truth_array.append('Idle')
    elif i < 56:
        ground_truth_array.append('Overhead')  
    elif i < 243:
        ground_truth_array.append('Idle')    
    elif i < 274:
        ground_truth_array.append('Backhand')     
    elif i < 335:
        ground_truth_array.append('Idle')       
    elif i < 353:
        ground_truth_array.append('Net Shot')       
    elif i < 400:
        ground_truth_array.append('Idle')   
    elif i < 413:
        ground_truth_array.append('Net Shot')    
    elif i < 492:
        ground_truth_array.append('Idle')    
    elif i < 506:
        ground_truth_array.append('Defensive Shot')    
    else:
        ground_truth_array.append('Idle')
    
df = pd.DataFrame(list(zip(*[ground_truth_array]))).add_prefix('Ground Truth ')

df.to_csv('ground_truth_data.csv', index=False)