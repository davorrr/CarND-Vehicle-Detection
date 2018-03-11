# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:34:15 2018

@author: davor
"""

from keras import backend as K  # Importing tf/th backend to handle tensor operations

from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
import cv2
from moviepy.editor import VideoFileClip
import time

# User made funtions imports
from keras_ssd300 import ssd_300 # SSD300 model import
from keras_ssd_loss import SSDLoss # Loss function for SSD
img_height = 300 # Set this in accordance with the video
img_width = 300 # Same

####
# 1: Building the model and loading the weights
####

K.clear_session() # Clears previous models from memory

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20, # Set this in accordance with video data
                mode='inference', # In 'inference' mode raw predictions are decoded into absolute coordinates and filtered via confidence  thresholding, 
                #non-maximum suppression and top-k filtering. Other modes are training and inference_fast.
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], 
                # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05])
                # These are anchor boxes scales
                aspect_ratios_per_layer =[[1.0, 2.0, 0.5],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                          [1.0, 2.0, 0.5],
                                          [1.0, 2.0, 0.5]],
                # Up are aspect ratios for each prediction layer
                two_boxes_for_ar1=True, # 2 anchors are generated for aspect ratio 1
                steps=[8, 16, 32, 64, 100, 300], # Number of elements in each prediction layer. 
                #They represent distance in pixels between centers of anchor boxes for each predictior layer
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], # Represents distance of centers of edge anchor boxes from
                # edges of the image as the fraction of the steps - so for 1. prediction layer 8*0.5=4 pixels
                limit_boxes=False, # If true then it limits box coordinates to stay within the image boundaries
                variances=[0.1, 0.1, 0.2, 0.2], # Scaling divisors for the encoded predicted box coordinates. These values (<1) upscale encoded predictions.
                coords='centroids', # Box coordinates formate to be used
                normalize_coords=True, # If True model uses relative instead of absolute coordinates (so coords. between 0 and 1)
                subtract_mean=[123, 117, 104], # Elements to be subtracted from the image pixel intensity values. 
                # Performs per channel mean normalization for color images.
                swap_channels=True, # If True color channel order is reversed; i.e. RGB -> BGR
                confidence_thresh=0.5, # Minimum classification confidence in a specific positive class in order to be considered for non-maximum suppression stage.
                iou_threshold=0.45, # All boxes that have a Jaccard similarity greater than 'iou_threshold' with a locally maximal box will be removed from the set 
                # of predictions for a given class, where 'maximal' refers to the box confidence score.
                top_k=200, # The number of highest scoring predictions to be kept for each batch item after the non-maximum suppression stage.
                nms_max_output_size=400) # The maximal number of predictions that will be left over after the NMS stage.
####
# 2: Loading trained weights
####


weights_path = 'C:/Users/davor/CarND-Vehicle-Detection-SSD-weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5'

model.load_weights(weights_path, by_name=True) # by_name flag enables loading weights for a different architecture then the one for which they were saved
# Very important for transfer learning.      

####
# 3: Compile the model so that Keras won't complain the next time you load it.
#### 
                
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0) # Instatiation of SSDLoss class
# neg_pos_ratio - Maximum ration of negative (i.e. background) to positive ground truth boxes to include in loss computation. Predicted boxes usually contain anchor boxes labeled
#                 with the background class. Number of these boxes is much greater then the number of positive boxes and it is necesarry to balance their influence
#                 on the loss.
# n_neg_min     - Minimum number of negative ground truth boxes to enter the loss computation per batch. This can be used to ensure that the model learns from a minimum number of
#                 negatives in batches in which there are very few, or even none at all, positive ground truth boxes.
# alpha         - Weight factor for the localisation loss in the computation of total loss

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

############################
def draw_boxes(img):
  orig_images = []  # For storing images
  input_images = [] # For storing resized images

  orig_images.append(img)
  im_res = cv2.resize(img, (300, 300))
  input_images.append(im_res)
  input_images = np.array(input_images)
  

  # predict on the image
  y_pred = model.predict(input_images)

  # set confidence threshold
  confidence_threshold = 0.5
  y_pred = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
  
  # set colors - this is an instance for 21 classes - we need only one
  colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist() # Making a np array of 21 different colors for 21 classes and then converting it to a list

  for x in range(len(colors)):
    colors[x] = [int(y * 255) for y in colors[x]]

  classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
             'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']
  
  for box in y_pred[0]:
    # Transform the predicted bounding boxes for 300x300 image to the original image dimensions.
    if(int(box[0])) == 7:
      xmin = int(box[-4] * img.shape[1] // img_width)
      ymin = int(box[-3] * img.shape[0] // img_height)
      xmax = int(box[-2] * img.shape[1] // img_width)
      ymax = int(box[-1] * img.shape[0] // img_height)
      color = colors[int(box[0])]
      label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
      cv2.rectangle(img,(xmin, ymin),(xmax, ymax), color ,2)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, label ,(xmin,ymin), font, 0.7 , (255,255,255,1),2 ,cv2.LINE_AA)
       
  return img


def process_image(img):
  start_time = time.time()
  img2 = draw_boxes(img)
  fps = 'FPS:  {:.2f}'.format(1.0 / (time.time() - start_time))
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img2, fps, (5,20), font, 0.7, (255,255,255,1), 1,cv2.LINE_AA)
  
  return img2 
   

output_video = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4")

video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)  



