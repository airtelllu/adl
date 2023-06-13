#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Define the CNN architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Show predictions for a sample input image
sample_img = x_test[0]
sample_label = y_test[0]
sample_img = np.expand_dims(sample_img, 0)
pred = model.predict(sample_img)
pred_label = np.argmax(pred)
print("Sample image true label:", sample_label)
print("Sample image predicted label:", pred_label)

# Display the sample image
plt.imshow(sample_img.squeeze(), cmap='gray')
plt.show()


# In[ ]:


Aim: Implement Convolutional Neural Network for Digit Recognition on the MNIST Dataset.


# In[1]:


Theory: A Convolutional Neural Network (CNN) is a type of deep learning algorithm that is particularly
well-suited for image recognition and processing tasks. It is made up of multiple layers, including
convolutional layers, pooling layers, and fully connected layers.
The convolutional layers are the key component of a CNN, where filters are applied to the input image to
extract features such as edges, textures, and shapes. The output of the convolutional layers is then passed
through pooling layers, which are used to down-sample the feature maps, reducing the spatial dimensions
while retaining the most important information. The output of the pooling layers is then passed through
one or more fully connected layers, which are used to make a prediction or classify the image.
Convolutional Neural Network Design:
• The construction of a convolutional neural network is a multi-layered feed-forward neural
network, made by assembling many unseen layers on top of each other in a particular order.
• It is the sequential design that give permission to CNN to learn hierarchical attributes.
• In CNN, some of them followed by grouping layers and hidden layers are typically convolutional
layers followed by activation layers.
• The pre-processing needed in a ConvNet is kindred to that of the related pattern of neurons in the
human brain and was motivated by the organization of the Visual Cortex.


# In[ ]:




