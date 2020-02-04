import pickle
import numpy as np
import cv2
import time
import socket
from io import StringIO
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Lambda

from collections import deque
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from IPython.display import HTML
from moviepy.editor import VideoFileClip

%matplotlib inline

from utils import load_weights, process_output, draw_boxes, preprocess, remove_duplicates

keras.backend.set_image_dim_ordering('th')
model = Sequential()

#normalize data
model.add(Lambda(lambda x: x/128 - 1,input_shape=(3,448,448)))

#layer 1
model.add(Convolution2D(16, 3, 3,border_mode='same',subsample=(1,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

#layer 2
model.add(Convolution2D(32,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 3
model.add(Convolution2D(64,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 4
model.add(Convolution2D(128,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 5
model.add(Convolution2D(256,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 6
model.add(Convolution2D(512,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))

#layer 7
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))

#layer 8
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))

#layer 9
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())

#layer 10
model.add(Dense(256))

#layer 11
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))

#output layer
model.add(Dense(1470))
model.summary()

load_weights(model,"yolo.weights")

class Pipeline:
    def __init__(self):
        self.boxes = []
        self.dropped = 0
        self.history = deque(maxlen=8)
        self.first_frames = True
    
    def apply_threshold(self, boxes):
        if len(boxes) == 0 and len(self.history) > 0:
            self.history.popleft()
        else:
            #create blank heatmap to store the most recent boxes
            heatmap = np.zeros([720,1280], np.int)
            for box in boxes:
                # Add += 1 for all pixels inside each box
                heatmap[box.y1:box.y2, box.x1:box.x2] += 1
            
            #add current heatmap to deque of heatmaps
            self.history.append(heatmap)
            
            #make sure to only use heatmap detection if there is enough information
            if len(self.history) > 4:
                self.first_frames = False
            else: 
                self.first_frames = True
            
            #add heatmaps together to make full heatmap
            if not self.first_frames:
                full_heatmap = np.zeros([720,1280], np.int)
                for preheat in self.history:
                    full_heatmap = np.add(full_heatmap, preheat)
                
                #if the center point of a box is not in an area the heatmap recognizes, remove that box
                new_boxes = []
                for box in boxes:
                    if full_heatmap[int(box.y), int(box.x)] > 2:
                        new_boxes.append(box)
                return new_boxes

        return boxes

        
    
    def pipeline(self,img):
        # each detection is ordered in terms of priority. A detection higher on the list will override a similar
        # detection lower down.
        # initial detection preformed on entire image.
        transposed, padhw, shavedim, resized = preprocess(img,shave=False)
        yolo_output = model.predict(np.array([transposed]))[0]
        boxes = process_output(yolo_output, threshold=0.20, padhw=padhw,shaved=False, shavedim=shavedim)
        
        #preform detection on each box in the stored previous boxes
        for box in self.boxes:
            offset = 150
            shavedim = [box.y1 - offset, box.y2 + offset, box.x1 - offset, box.x2 + offset]
            transposed, padhw, shavedim, resized = preprocess(img, shave=True, shavedim=shavedim)
            yolo_output = model.predict(np.array([transposed]))[0]
            boxes2 = process_output(yolo_output, threshold=0.35, padhw=padhw, shaved=True, shavedim=shavedim)
            boxes += boxes2 
        
        # last detection preformed picture shaved to just the highway ahead
        transposed, padhw, shavedim, resized = preprocess(img, shave=True, shavedim=(350,500, 500,1000))
        yolo_output = model.predict(np.array([transposed]))[0]
        boxes2 = process_output(yolo_output, threshold=0.30, padhw=padhw, shaved=True, shavedim=shavedim)
        boxes += boxes2  
        
                    
        #remove duplacate boxes        
        boxes = remove_duplicates(boxes, img)
        boxes = self.apply_threshold(boxes)
        if(len(boxes) < len(self.boxes)) and self.dropped < 3:
            self.dropped += 1
            boxes = self.boxes
        else:
            self.dropped = 0
            
        drawn = draw_boxes(boxes, img)
        plt.imshow(drawn)
#         print(boxes)
#         self.boxes = boxes
        return drawn
        
video_src = 'project_video.mp4'
cap = cv2.VideoCapture(video_src)
while True:
    print(3)
    s = socket.socket() 
    print ("Socket successfully created")        
    port = 12345               
    s.bind(('', port))
    print ("socket binded to %s" %(port))        
    s.listen(5)
    print ("socket is listening")
    c, addr = s.accept()
    print('Got connection from', addr)
    img_t = np.zeros((512,512,3), np.uint8)
    l=[]
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    pipe = Pipeline()
    test_image2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes= pipe.pipeline(test_image2)
    for box in boxes:
        cv2.circle(img_t,(x+ w//2,y+ h//2), 1, (0,0,255), 1)
        l.append(((box[0]+box[1])//2,(box[2]+box[3])//2))
#         print(l)
#         print(box[0])
    data = pickle.dumps(l)

    c.send(data)
    c.close()
    cv2.imshow('video', img_t)
    f = StringIO()
    np.savez_compressed(f,frame=img_t)
    f.seek(0)
    out = f.read()
    client_socket.sendall(out)

    if cv2.waitKey(33) == 27:
        break
    print(4)
cv2.destroyAllWindows()
