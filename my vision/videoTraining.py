import os
import sys
import cv2
import math
import glob
import random
import datetime
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
 
sys.path.append("/home/alex/machine_learning_tutorial/Mask_RCNN-master")

from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


ROOT_DIR = "/home/alex/machine_learning_tutorial/Mask_RCNN-master"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

#Video Directory
VIDEO_DIR = os.path.join(ROOT_DIR, "videos")

#Save Video to Directory
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")

#make directory if not exists...
if not os.path.exists(VIDEO_DIR):
   os.makedirs(VIDEO_DIR) 
     
#make directory if not exists...   
if not os.path.exists(VIDEO_SAVE_DIR):
   os.makedirs(VIDEO_SAVE_DIR)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("/home/alex/Desktop/mask_rcnn_vision_0020.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


#generate random color 
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors
   
   
#apply mask to images...   
def apply_mask(image, mask, color, alpha=0.5):
    
    for i, c in enumerate(color):
        image[:, :, i] = np.where(
            mask == 1,
            image[:, :, i] * (1 - alpha) + alpha * c,
            image[:, :, i]
        )
    return image



def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image
CLASSES =  ['lacRosseGround','lacRosseStick', 'lacRosseBall','lacRosseHelmet',  'squAshBall',  'squAshFloor', 'squAshWall', 'lawnTennisNet', 'lawnTennisCourt','lawnTennisBall' , 'tableTennisBall', 'tableTennisNet', 'tableTennisBat', 'racQuetFloor', 'racQuetBall',  'racQuetWall',  'badMintonCock','badMintonNet','badMintonCourt' ]
EVALUATION_CLASSES = []
EVALUATION_CLASSES.append('BG')
for CLASS in CLASSES:
    EVALUATION_CLASSES.append(CLASS)

coco_classes = ['person', 'cat', 'dog', 'handbag', 'suitcase', 'bottle', 'ball', 'cup', 'fork', 'knife', 'spoon', 'apple', 'orange', 'potted plant', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'cell phone', 'microwave', 'toaster', 'refrigerator', 'clock', 'vase']

for CLASS in coco_classes:
    EVALUATION_CLASSES.append(CLASS)




class InferenceConfig(coco.CocoConfig):
        
       GPU_COUNT = 1
       IMAGES_PER_GPU = 1

       # Skip detections with < 90% confidence
       DETECTION_MIN_CONFIDENCE = 0.95
       NUM_CLASSES = len(EVALUATION_CLASSES)
       IMAGE_MAX_DIM = 768

       # Give the configuration a recognizable name
       NAME = "vision"

config = InferenceConfig()
config.display()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
def train_video(model, VIDEO_DIR):
    capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, '0gkxTQGR6zI.mp4'))
    frames=[]
    frame_count=0
    success = True
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while success:
          ret, frame = capture.read()
          if not ret:
             break
          frame_count +=1
          frames.append(frame)
          print('frame_count:{0}'.format(frame_count))
          if len(frames)==1:
             results = model.detect(frames, verbose=0)
          print('Predicted')
          for i , item in enumerate(zip(frames, results)):
              frame = item[0]
              r = item[1]
              frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], EVALUATION_CLASSES, r['scores'])
                     
              name = '{0}.jpg'.format(frame_count + i - 1)
              name = os.path.join(VIDEO_SAVE_DIR, name)
              cv2.imwrite(name, frame)
              print('writing to file:{0}'.format(name))
          frames=[] 
    capture.release()
    video = cv2.VideoCapture(os.path.join(VIDEO_DIR, '0gkxTQGR6zI.mp4'))
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)< 3:
       fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
       print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
       
    else:
        fps = capture.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    video.release()
       
#will join all the predicted images into a video      
def join_Images(outvid, images=None, fps=30, size=None, is_color=True, format="FMP4"):
   fourcc = VideoWriter_fourcc(*format)
   vid = None
   for image in images:
       if not os.path.exists(image):
          raise FileNotFoundError(image)
       img = imread(image)
       if vid is None:
          if size is None:
             size = img.shape[1], img.shape[0]
             vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
       if size[0] != img.shape[1] and size[1] != img.shape[0]:
          img = resize(img, size)
       vid.write(img)
   vid.release()
   return vid
    

train_video(model, VIDEO_DIR)



VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "save")
images = list(glob.iglob(os.path.join(VIDEO_SAVE_DIR, '*.*')))
# Sort the images by integer index
images = sorted(images, key=lambda x: float(os.path.split(x)[1][:-3]))

outvid = os.path.join(VIDEO_DIR, "out_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now()))
join_Images(outvid, images, fps=30)


    
          
          
   


   

    
    
    
    
