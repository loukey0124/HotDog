import os
import cv2
import numpy as np
from time import sleep
from threading import Thread, Event
from tensorflow.lite.python.interpreter import Interpreter
    
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(650,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture("http://192.168.0.241:8080/video_feed")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


class Predict:
    THRESHHOLD = 0.6
    
    def __init__(self):
        self.event = Event()
        
        self.url = 'http://192.168.0.241:8080'
        
        MODEL_PATH = './model/'
        MODEL_NAME = 'DetectionModel.tflite'
        LABELMAP_NAME = 'labels.txt'
        
        self.IMW = 650
        self.IMH = 480
        
        self.isFinddog = False
        self.isFindcat = False
        
        CWD_PATH = os.getcwd()
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_PATH, MODEL_NAME)
        PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_PATH, LABELMAP_NAME)
        
        with open(PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
            
        self.interpreter = Interpreter(model_path=PATH_TO_CKPT)
        
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        self.input_mean = 127.5
        self.input_std = 127.5
        
        self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        
    def FindDog(self):
        videostream = VideoStream(resolution=(self.IMW, self.IMH), framerate=20).start()
        sleep(1)
        
        while True:
            if self.event.is_set():
                return
            
            frame = videostream.read()
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                continue
            
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            input_data = np.expand_dims(frame_resized, axis=0)
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0]
            
            self.isFinddog = False
            for i in range(len(scores)):
                if self.labels[int(classes[i])] == 'dog':
                    if ((scores[i] > self.THRESHHOLD) and (scores[i] <= 1.0)):
                        self.dogloc = (boxes[i][3] * self.IMW + boxes[i][1] * self.IMW) / 2
                        self.dogscale = (boxes[i][3] * self.IMW - boxes[i][1] * self.IMW)
                        self.isFinddog = True                        
    
    def GetDogLocaiton(self):
        try:
            if self.isFinddog == False:
                return 'NONE'
            
            if self.dogscale < 100:
                return 'go'
            
            if self.dogloc > self.width / 2 + 200:
                return 'right'
            elif self.dogloc < self.width / 2 + 0:
                return 'left'
            else:
                return 'stop'
        except:
            return 'NONE'
    
    def DogActivate(self):
        self.event.clear()
        Thread(target=self.FindDog, daemon=True).start()
    
    def DogDeactivate(self):
        self.event.set()
        
    def FindCat(self):
        videostream = VideoStream(resolution=(self.IMW, self.IMH), framerate=20).start()
        sleep(1)
        
        while True:
            if self.event.is_set():
                return 0
            
            frame = videostream.read()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            input_data = np.expand_dims(frame_resized, axis=0)
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0]
            
            self.isFindcat = False
            for i in range(len(scores)):
                if self.labels[int(classes[i])] == 'cat':
                    if ((scores[i] > self.THRESHHOLD) and (scores[i] <= 1.0)):
                        self.catloc = (boxes[i][3] * self.IMW + boxes[i][1] * self.IMW) / 2
                        self.catscale = (boxes[i][3] * self.IMW - boxes[i][1] * self.IMW)
                        self.isFindcat = True
    
    def GetCatLocaiton(self):
        try:
            if self.isFindcat == False:
                return 'NONE'
            
            if self.catscale < 200:
                return 'go'
            
            if self.catloc > self.width / 2 + 150:
                return 'right'
            elif self.catloc < self.width / 2 -250:
                return 'left'
            elif self.catscale < 200:
                return 'go'
            else:
                if self.catscale >= 200:
                    print(self.catscale)
                    return 'END'
                return 'stop'
        except:
            return 'NONE'
    
    def CatActivate(self):
        self.event.clear()
        Thread(target=self.FindCat, daemon=True).start()
    
    def CatDeactivate(self):
        self.event.set()
        
if __name__ == "__main__":
    p = Predict()
    print("start")
    
    p.Activate()
    while True:
        print(p.GetLocaiton())
        sleep(1)
        if cv2.waitKey(1) != -1:
            break
    
    p.Deactivate()      