# COSC428 Project Code

## Description 

Code for my computer vision project: Badminton Shot Recognition through Application of Traditional Analysis 
Model on 2D Human Pose Estimation. The goal of this project was to modify the Detectron2 human pose estimation to recognise five classes of badminton shot: overhead, backhand, net shot, defensive shot and idle/other.

## Installing Detectron2

### Option 1 (really hope this works):
Follow the installation instructions in https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### Option 2 (if you give up on Option 1):
This is what I found worked:

1. Install miniconda
2. Create an env.yml file with the following:

```
name: detectron2
channels:
  - pytorch
  - conda-forge
  - anaconda
  - defaults
dependencies:
  - python=3.8
  - numpy
  - pywin32
  - cudatoolkit=11.3
  - pytorch==1.8.1
  - torchvision
  - git
  - pip
  - pip:
    - git+https://github.com/facebookresearch/detectron2.git@v0.3
```

You may need to adjust the package versions according to your requirements. 

3. In miniconda, navigate to the env.yml file and run ```conda env create -f env.yml```
4. Activate the environment with ```conda activate detectron2```

Any missing package errors can be resolved by uninstalling the particular module and reinstalling it within the mininconda prompt as follows:
```
pip uninstall numpy; pip install numpy
```

If CUDA, PyTorch or Torchvision produce errors, note that they must be built together due to dependency. It is best to uninstall each (check success with ```conda list``` and ```pip list``` and then reinstall with the commands found at https://pytorch.org/: for example, I used ```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch```. Ensure the version of CUDA specified here matches that installed on your machine (For example, I installed CUDAv11.3 from https://developer.nvidia.com/cuda-11.3.0-download-archive). 

5. Check the success of your install with the following command in the miniconda prompt:
```
python
import torch; torch.__version__; torch.cuda.is_available()
```

Ensure the version of Torch is what you expect, and that ```torch.cuda.is_available()``` returns True. If it does not, repeat Step 4 and ensure Torch is built with CUDA support.

6. Either install Detectron2 with: ```python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'``` 
Or from a local clone: ```git clone https://github.com/facebookresearch/detectron2.git; python -m pip install -e detectron2```

Any errors may be due to GCC/G++ or VisualStudio versions. 

7. Check successful install of Detectron2 with 
```
python
import detectron2
```

## Adding my code to the Detectron2 library

1. Clone this repo with ```git clone https://github.com/gal65/COSC428-project-code.git```
2. Delete the contents of detectron2/demo and replace with that of the COSC428-project-code directory (the code in this repo)

<em> This is where the fun begins </em>

## Running the demo

Run the following code in the miniconda prompt (ensuring the correct environment is activated):

```
python3 demo.py --video-input test_video.mp4 --confidence-threshold 0.95 --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --opts MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl
```

This will process the provided sample video and display the recognition in a pop-up window. See bashScriptforDemo.txt for commands to run from webcam, and save the output videos.

### Other notes
The inclusion of openh264-1.8.0-win64.dll in this directory is a work-around to a video encoding error: if an error is thrown where a different version of openh264 is specified, download it from https://github.com/cisco/openh264/releases/tag/vX.X.X where 'vX.X.X is replaced by your required version. 
