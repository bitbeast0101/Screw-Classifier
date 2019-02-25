####### Image Object Detection Using Tensorflow-trained Classifier #########
# AUTHOR : Rohith C H
# Date : 24/09/2018
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

##some from EdjeElectronics example at 
##https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/Object_detection_image.py

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import collections

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'orig.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()
print("current directory" + CWD_PATH)
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 4

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `1`, we know that this corresponds to `allen screw`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# detection and recognition of variety of screws
final_score = np.squeeze(scores) #reduce dimensionality in mutli dimensional lists
count = 0
for i in range(100):
    if scores is None or final_score[i] > 0.8:
        count = count + 1
        
objects = []    # list of dictionaries with detected screws and its confidence values
uniq = []          # listof unique elements
ans = []
seen = []
for index, value in enumerate(classes[0]):
  object_dict = {}
  if scores[0, index] > 0.8:    #threshold = 0.8
     
    object_dict[(category_index.get(value)).get('name')] = \
                        scores[0, index]
    objects.append(object_dict)
    for key in object_dict.keys():
        ans.append(key)
print("ans" + str(ans))    
print("objects" + str(objects))

for ele in ans:
    if ele not in uniq: 
        uniq.append(ele)
    else:
        seen.append(ele)
print("unique elements :" + str(uniq))
print("multiple detections :" + str(seen))
width,height = image.shape[:2]
if seen:
    print("red flag")
    cv2.line(image,(height,0),(width,height),(0,0,255),25)
else:
    print("green flag")
    cv2.line(image,(height,0),(width,height),(0,255,0),25)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image,"Object count : " + str(count),(0,512),font,0.5,(0,0,255),2)

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)
print("number of objects detected : " + str(count))
# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
