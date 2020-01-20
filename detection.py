import mrcnn
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from keras.models import load_model




from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#%matplotlib inline


class PredictionConfig(Config):
    NAME = "crossings_cfg20200105T1348"
    # Number of classes (background + crossings+ no_crossings )
    NUM_CLASSES = 1 +1 +1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 80
    DETECTION_MIN_CONFIDENCE = 0.80
     # setting Max ground truth insances
    MAX_GT_INSTANCES=5
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#cfg = PredictionConfig()

def make_prediction(img, model):

   # img = load_img('/home/jyoti/Downloads/2380358_090517-kabc-6pm-broad-crosswalk-vid.jpg')
    # nz2.PNG
    img = img_to_array(img)
    img = img[:,:,:3]
    #model._make_predict_function()
    # detecting objects in the image
    result= model.detect([img])
    r = result[0]
    return r
