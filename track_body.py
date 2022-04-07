# Author: Gordon Lay
# May 2021

# Functions for shot recognition 

import numpy as np

def test_y_separation(keypoint1, keypoint2, threshold):
    # Check that the absolute distance between two key points is greater than 
    # the threshold, and that the y-component of keypoint1 is higher than that of keypoint2 in the frame.
    if keypoint1[1] > keypoint2[1] and (abs(keypoint1[1] - keypoint2[1]) > threshold):
        return True
    else:
        return False
    
def check_x_separation(keypoint1, keypoint2, threshold):
    # Check the separation between two key points is greater than the threshold
    if (abs(keypoint1[0] - keypoint2[0]) > threshold):
        return True
    else:
        return False

def test_linearity(keypoint1, keypoint2, keypoint3, threshold): 
    # Check the error (difference) in gradient between two sets of keypoints is less than a threshold
    gradient1 = (keypoint1[1] - keypoint2[1]) / (keypoint1[0] - keypoint2[0])
    gradient2 = (keypoint2[1] - keypoint3[1]) / (keypoint2[0] - keypoint3[0])
    if abs(gradient1) - abs(gradient2) < threshold:
        return True
    else:
        return False

def calculate_shoulder_width(shoulder1, shoulder2):
    # Compute shoulder-width based
    return np.sqrt((shoulder1[0] - shoulder2[0])**2 + (shoulder1[1] - shoulder2[1])**2)

def test_reach_low(knee1, knee2, ankle1, ankle2, wrist, threshold):
    # Check if right wrist is within range of knee or ankle point
    if (abs(knee1[1] - wrist[1]) < threshold or abs(knee2[1] - wrist[1]) < threshold or abs(ankle1[1] - wrist[1]) < threshold or abs(ankle2[1] - wrist[1]) < threshold):
        return True
    else:
        return False
    
def get_orientation(right_shoulder, left_shoulder, right_hip, left_hip):
    # returns True if right side of human has largest x-coordinate in frame
    # returns False if left side of human has largest x-coordinate in frame
    if ((right_shoulder[0] > left_shoulder[0]) and (right_hip[0] > left_hip[0])):
        return True
    else:
        return False
    
def check_outstretched_arm(right_shoulder, right_wrist):
    # [Checks if gradient from right shoulder to right wrist is negative]
    # Checks if gradient is less than 0.8 
    if (abs((right_shoulder[1] - right_wrist[1]) / (right_shoulder[0] - right_wrist[0])) < 0.8): #((right_shoulder[1] - right_wrist[1]) / (right_shoulder[0] - right_wrist[0])) < 0
        return True
    else:
        return False

def check_y_proximity(keypoint1, keypoint2, threshold):
    # Checks if absolute difference between the y-components of two key points
    # is less than a threshold
    if (abs(keypoint1[1] - keypoint2[1]) < threshold):
        return True
    else:
        return False    
