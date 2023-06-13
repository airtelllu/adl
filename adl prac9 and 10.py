#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Aim: Write a program for object detection from the image.


# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load the VGG16 model
model = VGG16()
# Load the image to detect objects in
img = load_img('objectdetectimage.jpg', target_size=(224, 224))
img_arr = img_to_array(img)
img_arr = np.expand_dims(img_arr, axis=0)
img_arr = preprocess_input(img_arr)
# Predict the objects in the image
preds = model.predict(img_arr)
decoded_preds = decode_predictions(preds, top=5)[0]
# Print the predicted objects and their probabilities
for pred in decoded_preds:
    print(f"{pred[1]}: {pred[2]*100:.2f}%")


# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
# Load the VGG16 model with pre-trained weights
model = VGG16()
# Load the image
image = load_img('objectdetectimage2.jpg', target_size=(224, 224))
# Convert the image to a numpy array
image = img_to_array(image)
# Reshape the image data for VGG
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# Preprocess the image
image = preprocess_input(image)
# Make predictions on the image using the VGG model
predictions = model.predict(image)
# Decode the predictions
decoded_predictions = decode_predictions(predictions, top=2)
# Print the predictions with their probabilities
for i, prediction in enumerate(decoded_predictions[0]):
    print("Object ", i+1, ": ", prediction[1], ", Probability: ", prediction[2])


# In[ ]:


Aim: Write a program for object detection using pre-trained models to use object detection.
Theory: VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network
(CNN) architecture with multiple layers. The “deep” refers to the number of layers with VGG-16 or VGG19 consisting of 16 and 19 convolutional layers.
Code:

