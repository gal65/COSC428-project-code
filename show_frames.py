import cv2

# Author: Gordon Lay
# May 2021

# Allows user to view current frame number and total number of frames as an overlay
# Click on cv2.imshow window (as if to drag the window) in order to 'pause' 
# Current frame can be observed and intervals of different shot types can be recorded for ground truth generation

# Load the video file.
cap = cv2.VideoCapture('test_demo_Trim.mp4') # video file path
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0

while cap.isOpened():
    count += 1
    ret, frame = cap.read()  
    if not ret:
        break
    cv2.putText(frame, str(count) + '/' + str(num_frames), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (100, 255, 255), 2)   
    cv2.imshow('frame', frame)      

    # 60 ms delay so that the individual frames can be observed more easily
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Release the video file, and close the GUI.
cap.release()
cv2.destroyAllWindows()