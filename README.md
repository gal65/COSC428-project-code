# COSC428 Project Code 

<em> This is where the fun begins </em>

## What all the files do! 

<h3> bashScriptforDemo.txt </h3>
This contains some helful bash commands for running the demo.py file from webcam or video input, with the correct configuration files. 
Open bash in COSC428_project_demo/demo and run the command you want.
  
<br>Errors concerning the config files is really common, with 'invisible' characters in the directory paths or missing slashes. Here is the bash command you would use to run the program on the provided test video:

```
python3 demo.py --video-input test_video.mp4 --confidence-threshold 0.95 --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl
```

### demo.py
The main file of the whole operation. Does not perform inferences but handles the arguments given to it from bash.

### demo_output.mp4 
A demonstration of the badminton shot recognition output in the intended application environment. You can generate your own by running the demo script. 

### show_frames.py, ground_truth.py, and measure_accuracy.py
Used to analyse test videos, manually generate ground truth in the form of CSV, and obtain an accuracy measurement.  
Mostly just for creating confusion matrices etc.

### predictor.py 
Most of the code is implemented here. Key point data is generated for every frame so this is leveraged and used to holistically determine the shot being performed by the player in frame.

### test_video.mp4 
A sample video that can be run through demo.py to obtain an output video identical to demo_output.mp4

### track_body.py
A module that contains helper functions that carry out the condition checking required for the criteria-based algorithm.

### webcam_to_file.py
Non-essential to the function of the code, but is useful for capturing sample videos from webcam.
