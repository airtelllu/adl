#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

# Load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding format 
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the model architecture
model = keras.Sequential([
keras.Input(shape=(32, 32, 3)),
layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),
layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),layers.MaxPooling2D(pool_size=(2, 2)),
layers.Flatten(),
layers.Dropout(0.5),
layers.Dense(10, activation="softmax"),
])

# Compile the model
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Train the model 
model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test,y_test))

import numpy as np
from PIL import Image
# Load the saved model
model = keras.models.load_model("cifar10_model.h5")
# Load and preprocess the test image
img = Image.open("two.png")
img = img.resize((32, 32))
img_array = np.array(img)
img_array = img_array.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)
# Make predictions on the test image
predictions = model.predict(img_array)
# Get the predicted class label
class_label = np.argmax(predictions)
# Print the predicted class label
print("Predicted class label:", class_label)


# In[4]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert the labels to one-hot encoded vectors
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Define a modified model architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with modified parameters
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Save the trained model to a file
model.save("cifar10_model.h5")

# Load the saved model
model = keras.models.load_model("cifar10_model.h5")

# Load and preprocess a test image
img = Image.open("two.png")
img = img.resize((32, 32))
img_array = np.array(img)
img_array = img_array.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make predictions on the test image
predictions = model.predict(img_array)

# Get the predicted class label
class_label = np.argmax(predictions)

# Print the predicted class label
print("Predicted class label:", class_label)


# In[ ]:


: Implement deep learning for recognizing classes for datasets like CIFAR-10 images for previously
unseen images and assign them to one of the 10 classes.


# In[ ]:


Theory: The CIFAR-10 dataset (Canadian Institute for Advanced Research) is a collection of images
that are commonly used to train machine learning and computer vision algorithms. It is one of the most
widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color
images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs,
frogs, horses, ships, and trucks. There are 6,000 images of each class.
Computer algorithms for recognizing objects in photos often learn by example. CIFAR-10 is a set of
images that can be used to teach a computer how to recognize objects. Since the images in CIFAR-10 are
low-resolution (32x32), this dataset can allow researchers to quickly try different algorithms to see what
works.
CIFAR-10 is a labeled subset of the 80 million Tiny Images dataset from 2008, published in 2009. When
the dataset was created, students were paid to label all of the images. Various kinds of convolutional
neural networks tend to be the best at recognizing the images in CIFAR-10.

