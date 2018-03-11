**Vehicle Detection Project**

The goals / steps of this project are the following:

* Implement a real-time deep learning algorithm for object detection.
* Start with an available Tensorflow API solution and then try do obtain better results using a more customized solution.
*  Integrate the implemented real-time objecte detection with Advanced Lane Lines finding project.

[//]: # (Image References)
[image1]: ./output_images/image1.jpg "Object detection architectures overview"
[image2]: ./output_images/image2.jpg "SSD300"
[image3]: ./output_images/image3.jpg "Detection and Classification Block"
[image4]: ./output_images/image4.jpg "Detection and Classification Procedure"
[image5]: ./output_images/image5.png "Tensorflow's Object Detection API"
[image6]: ./output_images/image6.png "False detection" 
[image7]: ./output_images/image7.png "Unsuccessful detection 1" 
[image8]: ./output_images/image8.png "Unsuccessful detection 2" 
[image9]: ./output_images/single_image_inference.PNG "SSD300 inference on single image"
[image10]: ./output_images/image9.png "Real-time video inference"
[image11]: ./output_images/image10.png "Merged projects result" 
[video1]: ./project_video.mp4

---
### README

### Reasons for chosing Deep Learning instead of Classical Computer Vision

The specifications of this project were based with presuption that the Computer Vision procedures for object detection such as Histogram of Oriented Gradients (HOG) would be used along with Machine Learning procedures such as Support Vector Machine or Decision Tree. However having in mind the recent development of Deep Learning arcitectures which enable a more acurrate object classification along with an ability to work in real-time I decided to explore them and gain more insight into these techniques.

### Model architecture

#### 1 Architecture selection

After a review of several current object detection models such as: YOLO, YOLO9000, SSD and Faster RNN the Single Shot multibox Detector (SSD) was chosen for reasons that his speed and accuracy was shares the current state-of-the-art together with the YOLO9000, but is easier to implement from the initial solution because SSD uses caffe implementation which is easier to port to Tensorflow then the YOLO9000 darknet implementation. On the graph below different object detection algorithms are shown, albeit without YOLO9000, where we can see that for example Faster-RCNN has higher accuracy than SSD but SSD is much faster and therefore more suitable for a real-time implementation.

![alt text][image1]

#### 2. Single Shot multibox Detector (SSD)

Single Shot multibox Detector is the one of two architectures (YOLO family being the second) that does both object classification and object localization in a single pass. Earlier architecture such as those from the Region-Convolutional Neural Network (RCNN) family employ a two step process where they usually first do the object classification and then the localisation which makes the algorithm to slow and unapproapriate for real-time implementations.

SSD consists of two distinct networks architectures - first network is used for object classification and second one is used for object localization. For object classification we can use any of the famous object detection algorithms such as VGG, ResNet etc. On the image below is a variant of SSD300 given in the [original paper](https://arxiv.org/pdf/1512.02325v5.pdf) using the VGG-16 for object classification:

![alt text][image2]

Here we can see that on the input we have a 300x300 image on which we detect objects. The image is passed into the VGG-16 network which goes down in size from 300x300x3 to 1x1x256. This approach resembles the approach used by Google in their Inception family of object classifier architectures. This enables the detection of various object sizes where for example 38x38x512 layer of VGG can detect only relatively small objects while as the size of the layers is going down we are able to detect bigger and bigger objects. Then results from each convolutional layer we pass to detector and classifier block.

![alt text][image3]

The detector and classifer block in one pass proposes a serie of bounding boxes (in the case above 75). Here we have a 5x5 size feature map which corresponds to the 25 bounding box locations on the image. Bounding box is determined by it's center given in x and y coordinates and by it's width and height. For each cetral positon the arcitecture generates 3 bounding boxes which on the end gives us 75 bounding boxes. That gives us 5x5x12 feature maps in the localisation block with (x,y,w,h) parameters for for 75 bounding boxes. In the confidence block we have 5x5x63 feature maps with detection confidence results that the object in one of the 21 classes (20 plus background) is detected for each of the proposed bounding boxes.

![alt text][image4]

After these steps filtration by confidence results is done which reduces the number of proposed bounding boxes to top few results. The results are then passed to the Fast Non-Maximum Suppression block which provides the end result.


### Model architecture implementation

#### 1. Tensorflow Object Detection API

![alt text][image5]

In june 2017 Google published the Object Detection API for Tensorflow which enables an easy out of the box way to deploy a model for object detection. The API consists of a set of pretrained object detection models such as different variants of SSD and Faster-RCNN. I chose this API as the starting point from which I wanted to go deeper into the implementation of SSD architecture.

| Model name  | Speed (ms) | COCO mAP[^1] | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: |
| [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) | 30 | 21 | Boxes |
| [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz) | 42 | 24 | Boxes |
| [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 58 | 28 | Boxes |
| [faster_rcnn_resnet50_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) | 89 | 30 | Boxes |
| [faster_rcnn_resnet50_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz) | 64 |  | Boxes |
| [rfcn_resnet101_coco](http://download.tensorflow.org/models/object_detection/rfcn_resnet101_coco_2018_01_28.tar.gz)  | 92 | 30 | Boxes |
| [faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz) | 106 | 32 | Boxes |
| [faster_rcnn_resnet101_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz) | 82 |  | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz) | 620 | 37 | Boxes |
| [faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz) | 241 |  | Boxes |
| [faster_rcnn_nas](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz) | 1833 | 43 | Boxes |
| [faster_rcnn_nas_lowproposals_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz) | 540 |  | Boxes |
| [mask_rcnn_inception_resnet_v2_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz) | 771 | 36 | Masks |
| [mask_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz) | 79 | 25 | Masks |
| [mask_rcnn_resnet101_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz) | 470 | 33 | Masks |
| [mask_rcnn_resnet50_atrous_coco](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz) | 343 | 29 | Masks |

For this project the SSD combined with the MobileNet arcitecture trained on the Microsoft's COCO dataset was chosen being the fastest.

Pipeline for real-time object detection was is given below: 
```python
def process_image(image_np):
  start_time = time.time()
  #ret, image_np = cap.read()
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  # Each box represents a part of the image where a particular object was detected.
  boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  # Each score represent how level of confidence for each of the objects.
  # Score is shown on the result image, together with the class label.
  scores = detection_graph.get_tensor_by_name('detection_scores:0')
  classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  # Actual detection.
  (boxes, scores, classes, num_detections) = sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True, line_thickness=8)
  img = cv2.resize(image_np, (480,360))
  label = 'FPS:  {:.2f}'.format(1.0 / (time.time() - start_time))
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img, label ,(5,20), font, 0.7 , (255,255,255,1), 1 ,cv2.LINE_AA)
   
  return img

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    output_video = 'C:/Users/davor/CarND-Vehicle-Detection/output_API.mp4'
    clip1 = VideoFileClip("C:/Users/davor/CarND-Vehicle-Detection/project_video.mp4")

    video_clip = clip1.fl_image(process_image)
    video_clip.write_videofile(output_video, audio=False)  
```

The entire script can be viewed in [`object_detection_API.py`](./object_detection_API.py). It simply loads the frozen model and then uses it for the detection of objects on the video frame by frame. Since it was trained on COCO dataset it is capable of detecting much bigger set of objects that just vehicles on the road.

The speed of the model is reasonable ranging from 12 to 16 FPS but the detection results are quite unsatisfactory since the model fails to detect a car in front of him at various instances, once for full 3 seconds (0:11 - 0:14 in the video) and also makes several false detections of persons and cars.
![alt text][image6]
![alt text][image7]
![alt text][image8]

It should be noted that the script works only with the object detection API present, which is not included in this repository.

Full video can be viewed  in [here](./output_API.mp4).


#### 2. Custom made SSD implementation

In this step the Tensorflow's API was replaced with custom made implementation of SSD architecture. This was done to obtain better performance as well as to gain more insight into the SSD arcitecture. As a starting point for this was used a [Github repository](https://github.com/pierluigiferrari/ssd_keras) made by Pierreluigi Ferrari. The repository contains Keras implementations of SSD300 and SSD512 architectures based on the original SSD paper and using a VGG-16 architecture for object classification.

In the repo there are alredy provided weights for the models trained on the VOC 2007, VOC 2012 and COCO datasets for both SSD300 and SSD512 implementations.

|Evaluated on        | VOC 2007 test     |VOC 2007 test                | VOC 2012 test  |
|:------------------:|:-----------------:|:---------------:|:--------------:|
|trained on IoU rule | 07+12  (0.5)      | 07+12+COCO (0.5)|07+12+COCO (0.5)|
| SSD300             | 77.6              |81.2             |79.4            |
| SSD512             | 79.8              |83.2             |82.3            |

Based on these results as a starting starting architecture a SSD300 trained on the VOC 2007, VOC 2012 and COCO and tested on the VOC 2007 was chosen for inference.

The SSD300 architecture was implemented fully using Keras wrapper and can be viewed in [`keras_ssd300.py`](./keras_ssd300.py). It was implemented as a port of the original Caffe implementation of the architecture. 

Besides the Keras implementation of the model itself it was necessary to implement:
- Loss function in [`keras_ssd_loss.py`](./keras_ssd_loss.py)
- Anchor boxes generation layer in [`keras_layer_AnchorBoxes.py`](./keras_layers/keras_layer_AnchorBoxes.py)
- L2 Normalisation layer in [`keras_layer_L2Normalization.py`](./keras_layers/keras_layer_L2Normalization.py)
- Detections decoder layer in [`keras_layer_DecodeDetections2.py`](./keras_layers/keras_layer_DecodeDetections2.py)


For the purpouse of inference first step was to load the pretrained weights contained in the HDF5 file.

#### 1. Single image inference results

As a first step in testing the approach was to do a single image inference for which the test image provided by Udacity was used. In the script [`single_image_inference_SSD300.py`](./single_image_inference_SSD300.py) the model is constructed by calling the _SSD300()_ function like this:

```python
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
```
Afterwards the trained model weights are loaded using:

```python
weights_path = 'C:/Users/davor/CarND-Vehicle-Detection-SSD-weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5'

model.load_weights(weights_path, by_name=True) # by_name flag enables loading weights for a different architecture then the one for which they were saved
```

Weights themselves are too large for this repository but they can be downloaded [this link](https://drive.google.com/open?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox).

After the weights are loaded the model was compiled using following lines of code:
```python
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0) # Instatiation of SSDLoss class
# neg_pos_ratio - Maximum ration of negative (i.e. background) to positive ground truth boxes to include in loss computation. Predicted boxes usually contain anchor boxes labeled
#                 with the background class. Number of these boxes is much greater then the number of positive boxes and it is necesarry to balance their influence
#                 on the loss.
# n_neg_min     - Minimum number of negative ground truth boxes to enter the loss computation per batch. This can be used to ensure that the model learns from a minimum number of
#                 negatives in batches in which there are very few, or even none at all, positive ground truth boxes.
# alpha         - Weight factor for the localisation loss in the computation of total loss

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
```
Now the inference on a single frame could be done:
```python
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
```
It should be noted that the image resizing done here does not maintain aspect ratio. In order to try to achieve better results resizing that maintains aspect ratio was attempted but that interestingly lead to lower confidence scores so the first approach was kept.

Results are really good and both vehicles on the image were detected with a very large confidence score.
![alt text][image9]


#### 2. Real-time video inference results

Next step was to build the pipeline that would attepmt the real-time inference on the project video. The pipeline improves on the single image inference pipeline and uses the function _draw_boxes()_ to draw detection boxes on the video frames:
```python
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
```
After that in the _process_image()_ function the frame rate is calculated and drawn on the frame:
```python
def process_image(img):
  start_time = time.time()
  img2 = draw_boxes(img)
  fps = 'FPS:  {:.2f}'.format(1.0 / (time.time() - start_time))
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(img2, fps, (5,20), font, 0.7, (255,255,255,1), 1,cv2.LINE_AA)
  
  return img2 
```
The entire script can be viewed in the [`video_inference_SSD300.py`](./video_inference_SSD300.py).

The example of successful detection of vehicles on a frame is given below:
![alt text][image10]

As can be seen the speed of inference is around 10 fps which is lower than in the Tensorflow's API which is probably because the API's model uses MobileNets for image classification which is faster than the VGG-16 from the original implementation used here. Also in the original code of SSD when all the classes are kept there are several instances of false detections railings as a train or a boat. In order to avoid that only one class, car was left in the code for detection.

The entire video can be viewed [here](./output_video.mp4).

#### 3. Merging the project with the Advanced Lane Lines project

It is also interesting to merge Advanced Lane Lines project which using computer vision techinques detects lane lines with this project so we would have both lane lines and vehicle detection in one project.

![alt text][image11]

As it can be seen from the picture this two projects were successfuly merged and now we have an ability to simultanously detect lane lines together with the vehicles. Only problem is now the frame rate which is quite low of cca 5 fps which is unsuitable for real time implementations. It must be mentioned that all this was done with no regard on optimization of the code for speed and probably siginficantly higher fps can be achieved with some code optimization.

The video of the merged projects results can be viewed [here](./output_video2.mp4).

### Discussion

In the approach I took I decided to use deep learning techniques in order to obtain real-time object detection. This was done step by step first choosing the Tensorflow's Object Detection API. He proved to be the fastest because of the fact that the SSD implementation there used MobileNets for image classification, but also this meant that the quality of detection was the lowest. To improve on the quality in the next step I used Pierluigi Ferrari's Keras implementation of original SSD architecture written in Caffe. This proved to have much better detection confidence but the speed was lower. On the end project was successfuly merged with Advanced Lane Lines project thereby creating the ability to simultaniously detect both lane and vehicles.

In all the implementations the standard models trained on the usual datasets such as V0C 2007, V0C 2012 and COCO were used. They provided good results but if even better results are required some finetuning of the model using the Udacity's dataset would probably improve on both the number of detected vehicles in the frame as well as the confidence of the detections.

The pipeline itself is likely to fail in the conditions that are significantly different than the ones it was trained on. Some of these conditions would be and situation where the visibilty is low, such as night time, adverse weather such as heavy rain, fog, and snow. 

The advantage of using deep neural networks in regard to classical computer vision techniques is that in comparison to the computer vision techniques deep neural networks almost need almost no feature extraction and data preprocessing.

My next step in the project will be to go deeper into the model and attempt to improve on the inference speed by using some of the algorithms for efficient inference such as pruning, weight sharing, quantization, and other techniques.


