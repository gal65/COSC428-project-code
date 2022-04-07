# webcam_to_file.py

import cv2

# Open the first camera connected to the computer.
cap = cv2.VideoCapture(1)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'x264')
out = cv2.VideoWriter('output_demo.mp4', fourcc, float(frames_per_second), (width, height))

while True:
    ret, frame = cap.read()  # Read an frame from the webcam.  
    #frame = cv2.resize(frame, (1080, 720))   

    out.write(frame)  # Write the frame to the output file.

    cv2.imshow('frame', frame)  # While we're here, we might as well show it on the screen.

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera device and output file, and close the GUI.
cap.release()
out.release()
cv2.destroyAllWindows()
