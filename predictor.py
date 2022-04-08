# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted by Gordon Lay

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import pandas as pd
import torch
import cv2
import random
import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from track_body import test_y_separation
from track_body import check_x_separation
from track_body import test_linearity
from track_body import calculate_shoulder_width
from track_body import test_reach_low
from track_body import get_orientation
from track_body import check_outstretched_arm
from track_body import check_y_proximity

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        self.mode = 1   # Mode 0: Shot recognition
                        # Mode 1: Match suggested shot (training/coaching mode)
                        # Mode 2: Accuracy test mode
                      
        self.is_setup = 0
        
        # if user is left handed (self.is_left_handed = 1), flip the image
        self.is_left_handed = 0
        
        self.VERT_THRESHOLD = 15            # Number of pixels for one point to be considered 'above' another in frame
        self.LIN_THRESHOLD = 10             # Max difference between two gradients to be considered linear
        self.SEPARATED_THRESHOLD = 50       # Min distance (in pixels) for two points to be considered separated
        self.PROX_THRESHOLD = 40            # Max separation for two points to be considered in proximity to each other
        
        if (self.mode == 2):
            self.outputList = [] 
        
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)     
            
        self.counter = 0
        self.previous_ID = 'Idle'
        self.message = 'Idle'
            
    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return predictions, vis_output  

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                # Resize frame to 480p for controlled resolution and faster inference
                frame = cv2.resize(frame, (640, 480))     
                if (self.is_left_handed == 1):
                    frame = cv2.flip(frame, 1)
                yield frame
            elif (self.mode == 2):
                # On last frame: print recognition data list and save to .csv for accuracy measurements
                print(self.outputList)
                df = pd.DataFrame(list(zip(*[self.outputList]))).add_prefix('Recognition ')
                df.to_csv('recognition_data.csv', index=False)                
                break
            else:
                break
        
    def shot_recognition(self, predictions, frame):
        """
        Perform shot recognition.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                    model. Following fields will be used:
                    "pred_keypoints"
            frame (ndarray): an BGR image of shape (H, W, C), in the range [0, 255].
            
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        
        if getattr(self, 'recognise_shot', None) is None:
            default = 'Idle' # No shot being played
            string = default
            self.identify = string # Recognised shot type to be overlayed   
        
        # Initialise keypoints
        if len(predictions) > 0:
            keypoint_names = self.metadata.get("keypoint_names")
            keypoints = predictions.get('pred_keypoints').squeeze()  
            
            left_shoulder_index = keypoint_names.index('left_shoulder')
            right_shoulder_index = keypoint_names.index('right_shoulder')
            left_shoulder = keypoints[left_shoulder_index]
            right_shoulder = keypoints[right_shoulder_index]             
            
            left_elbow_index = keypoint_names.index('left_elbow')
            right_elbow_index = keypoint_names.index('right_elbow')
            left_elbow = keypoints[left_elbow_index]
            right_elbow = keypoints[right_elbow_index]              
            
            left_wrist_index = keypoint_names.index('left_wrist')
            right_wrist_index = keypoint_names.index('right_wrist')
            left_wrist = keypoints[left_wrist_index]
            right_wrist = keypoints[right_wrist_index]       
            
            left_hip_index = keypoint_names.index('left_hip')
            right_hip_index = keypoint_names.index('right_hip')
            left_hip = keypoints[left_hip_index]
            right_hip = keypoints[right_hip_index]       
            
            left_knee_index = keypoint_names.index('left_knee')
            right_knee_index = keypoint_names.index('right_knee')
            left_knee = keypoints[left_knee_index]
            right_knee = keypoints[right_knee_index]   
            
            left_ankle_index = keypoint_names.index('left_ankle')
            right_ankle_index = keypoint_names.index('right_ankle')
            left_ankle = keypoints[left_ankle_index]
            right_ankle = keypoints[right_ankle_index]   
        
            # run only once, initially. Should be in __init__() along with key
            # point initialisation
            if self.is_setup == 0: 
                self.SEPARATED_THRESHOLD = left_shoulder[0] - right_shoulder[0]
                self.is_setup = 1
        
        # Criteria-based algorithm 
        if get_orientation(right_shoulder, left_shoulder, right_hip, left_hip):
            if check_y_proximity(right_shoulder, right_wrist, self.PROX_THRESHOLD) and test_linearity(left_shoulder, left_elbow, left_wrist, self.LIN_THRESHOLD):
                string = 'Backhand'
                self.identify = string   
        elif not get_orientation(right_shoulder, left_shoulder, right_hip, left_hip) and (right_wrist[0] > left_ankle[0]):
            if test_linearity(right_shoulder, right_elbow, right_wrist, self.LIN_THRESHOLD) and not test_reach_low(left_knee, right_knee, left_ankle, right_ankle, right_wrist, self.PROX_THRESHOLD) and check_outstretched_arm(right_shoulder, right_wrist):
                string = 'Net Shot'
                self.identify = string 
            elif test_reach_low(left_knee, right_knee, left_ankle, right_ankle, right_wrist, self.PROX_THRESHOLD):
                string = 'Defensive Shot'
                self.identify = string    
        elif test_y_separation(right_shoulder, right_elbow, self.VERT_THRESHOLD):
                string = 'Overhead'
                self.identify = string    
        elif test_linearity(right_shoulder, right_elbow, right_wrist, self.LIN_THRESHOLD) and check_x_separation(left_ankle, right_ankle, self.SEPARATED_THRESHOLD):
            if test_reach_low(left_knee, right_knee, left_ankle, right_ankle, right_wrist, self.PROX_THRESHOLD):
                string = 'Defensive Shot'
                self.identify = string
            elif not get_orientation(right_shoulder, left_shoulder, right_hip, left_hip) and check_outstretched_arm(right_shoulder, right_wrist):  
                string = 'Net Shot'
                self.identify = string
        else:
            string = default 
            self.identify = string
        
        # if in training/coach mode 
        # trainer picks a shot to suggest at random      
        if (self.mode == 1):   
            shot_list = ['Overhead','Backhand','Net Shot', 'Defensive Shot']    
            if (self.message == 'Idle'):
                self.message = shot_list[random.randint(0, 3)]
            if (self.identify == self.previous_ID):   
                if (self.identify == self.message):
                    self.counter += 1
            elif (self.counter >= 15):
                self.message = shot_list[random.randint(0, 3)] 
                self.counter = 0
                self.previous_ID = self.identify
            else:
                self.counter = 0
                self.previous_ID = self.identify
                
            # Overlay the current suggested shot type
            if (self.message == 'Overhead'):
                cv2.putText(frame, str('Play an ' + self.message + '!'), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, str('Play a ' + self.message + '!'), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                
        # if in shot recognition mode
        elif (self.mode == 0):
            cv2.putText(frame, str(self.identify), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            
        # if in accuracy measurement mode
        elif (self.mode == 2):
            cv2.putText(frame, str(self.identify), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            # Append recognition output to list
            self.outputList.append(self.identify)
        
        # Check CUDA usage
        # print(torch.cuda.get_device_name(0))
        # print('Using device:', self.cpu_device)
        # print('Memory Usage:')
        # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        # print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')         
        
        return frame

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: RGB visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            # Move the predictions tensor off the GPU so we can access the
            # data with the CPU.                         
            if "instances" in predictions:
                predictions = predictions["instances"].to(torch.device('cpu'))
                # Perform the checks for the shot recognition algorithm, and draws
                # the associated overlays.
                vis_frame = self.shot_recognition(predictions, frame)
    
                # Draw the neural network overlay (object bounding box, and body
                # keypoints).
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                vis_frame = video_visualizer.draw_instance_predictions(vis_frame, predictions)
                vis_frame = vis_frame.get_image()
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)                
           
            else:
                vis_frame = frame

            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
