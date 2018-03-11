# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 17:34:15 2018

@author: davor
"""

from keras import backend as K  # Importing tf/th backend to handle tensor operations
from keras.models import load_model
from keras.preprocessing import image # Module for real-time data augmentation
from keras.optimizers import Adam
from scipy.misc import imread ## Exchange with openCV
import numpy as np
from matplotlib import pyplot as plt

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
                coords='centroids', # Box coordinates format to be used
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


### Load some images
orig_images = []  # For storing images
input_images = [] # For storing resized images

img_path = 'test_images/test4.jpg'

# Resizing the image to 300x300 pixels to fit into the SSD300
orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width)) # Loads image in PIL format
img = image.img_to_array(img) # Converts PIL image instance to Numpy array
input_images.append(img)
input_images = np.array(input_images)

### Make predictions

y_pred = model.predict(input_images)
# y_pred contains a fixed number of predictions per batch item (200 in original model configuration), many
# of which are low confidence predicions or dummy entries. We therefore need to apply a confidence threshold
# to filter out the bad predictions.

confidence_threshold = 0.5  ### Already done ?

y_pred = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf   xmin   ymin   xmax   ymax')
print(y_pred[0])


### Display the image and draw the predicted boxes onto it

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist() # Making a np array of 20 different colors for 20 classes and then converting it to a list
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

plt.figure(figsize=(20,12))
plt.imshow(orig_images[0])

current_axis = plt.gca() # Creating the axes instances

for box in y_pred[0]:
  # Transform the predicted bounding boxes for 300x300 image to the original image dimensions.
  xmin = box[-4] * orig_images[0].shape[1] / img_width
  ymin = box[-3] * orig_images[0].shape[0] / img_height
  xmax = box[-2] * orig_images[0].shape[1] / img_width
  ymax = box[-1] * orig_images[0].shape[0] / img_height
  color = colors[int(box[0])]
  label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
  current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                       color=color, fill=False, linewidth=2)) # Drawing rectangle around detected object
  current_axis.text(xmin, ymin, label, size='x-large', color='white', 
                    bbox={'facecolor':color, 'alpha':1.0})
                
                
                