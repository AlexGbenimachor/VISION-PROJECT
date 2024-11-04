import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append("/home/alex/machine_learning_tutorial/Mask_RCNN-master")

from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize



#choose the gpu device for training
'''
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
     raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))
'''
# Root directory of the project
ROOT_DIR = "/home/alex/machine_learning_tutorial/Mask_RCNN-master"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join("/home/alex/Desktop/mask_rcnn_vision_0020.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_FOLDER="/home/alex/Desktop/VisionComplete/Dataset/dataset"
IMAGE_DIR = os.path.join(IMAGE_FOLDER, "dataset")



class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    
    #name of project
    NAME = "vision"
    
    GPU_COUNT = 1
    
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 1 + 19
    
    
    
    
    
config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])


class_names=['BG','lacRosseGround','lacRosseStick', 'lacRosseBall','lacRosseHelmet',  'squAshBall',  'squAshFloor', 'squAshWall', 'lawnTennisNet', 'lawnTennisCourt','lawnTennisBall' , 'tableTennisBall', 'tableTennisNet', 'tableTennisBat', 'racQuetFloor', 'racQuetBall',  'racQuetWall',  'badMintonCock','badMintonNet','badMintonCourt' ]





# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=2)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
                            
                            
                            
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, 'image9.png'))

# Run detection
results = model.detect([image], verbose=2)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])








