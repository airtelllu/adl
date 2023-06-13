#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

# Load the data
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
train_data = train_data.reshape((60000, 784)) / 255.0
test_data = test_data.reshape((10000, 784)) / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,),
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=128, validation_data=(test_data, test_labels))


# In[3]:


import tensorflow as tf  # Import TensorFlow library

# Load the data
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()  # Load MNIST dataset

'''Loads the MNIST dataset using the load_data() function provided by Keras, a high-level API of TensorFlow.
The MNIST dataset contains 70,000 images of handwritten digits that are split into 60,000 training images
and 10,000 testing images.'''

# Preprocess the data
train_data = train_data.reshape((60000, 784)) / 255.0  # Reshape and normalize training data
test_data = test_data.reshape((10000, 784)) / 255.0  # Reshape and normalize testing data
train_labels = tf.keras.utils.to_categorical(train_labels)  # Convert training labels to one-hot encoding
test_labels = tf.keras.utils.to_categorical(test_labels)  # Convert testing labels to one-hot encoding

'''Preprocess the data. The images are first reshaped from a 3D array (28x28 pixels) to a 2D array (784 pixels).
Then, the pixel values are normalized to be between 0 and 1 by dividing by 255.
The labels are converted to one-hot encoding format using the to_categorical() function provided by Keras.
This is done to make it easier for the model to classify the images into 10 different classes (one for each digit).'''

# Define the model architecture
model = tf.keras.models.Sequential([
    # Define sequential model
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    # Add a fully connected layer with 128 units, ReLU activation, and L2 regularization
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    # Add another fully connected layer with 64 units, ReLU activation, and L2 regularization
    tf.keras.layers.Dense(10, activation='softmax')  # Add a final output layer with 10 units (one for each class), softmax activation
])

'''This code defines the architecture of the neural network model.
The Sequential() function is used to create a sequential model, meaning that the layers are added in sequence.
Three fully connected layers are defined using the Dense() function.
The first layer has 128 units, ReLU activation, and L2 regularization with a regularization parameter of 0.01.
The second layer has 64 units, ReLU activation, and L2 regularization with a regularization parameter of 0.01.
The third and final layer has 10 units, softmax activation, and is used for the classification task.'''

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Use Adam optimizer with learning rate 0.001
              loss='categorical_crossentropy',  # Use categorical cross-entropy loss function
              metrics=['accuracy'])  # Monitor accuracy during training

'''This code compiles the model. The compile() function configures the model for training by specifying the optimizer,
loss function, and metrics to monitor during training.
In this case, the Adam optimizer is used with a learning rate of 0.001, categorical cross-entropy is used as the loss function,
and accuracy is monitored during training.'''

# Train the model
history = model.fit(train_data, train_labels, epochs=10, batch_size=128, validation_data=(test_data, test_labels))

'''This code trains the model using the fit() function.The training data and␣labels are passed in as arguments,
along with the number of epochs to train for, the batch size to use, and the validation data to use for
monitoring model performance during training. The fit() function returns a history object that contains information
about the training process, such as the loss and accuracy at each epoch.
The purpose of this program is to demonstrate how to implement a neural network model for image classification
using TensorFlow/Keras. The model uses regularization techniques to prevent overfitting and achieves high accuracy
on the MNIST dataset.'''


# In[ ]:


Write a Program to implement regularization to prevent the model from overfitting 


# In[ ]:


Theory: Regularization is a technique which makes slight modifications to the learning algorithm such
that the model generalizes better. This in turn improves the model’s performance on the unseen data as
well. L1 and L2 are the most common types of regularization. These update the general cost function by
adding another term known as the regularization term.
Cost function = Loss (say, binary cross entropy) + Regularization term
Due to the addition of this regularization term, the values of weight matrices decrease because it assumes
that a neural network with smaller weight matrices leads to simpler models. Therefore, it will also reduce
overfitting to quite an extent. However, this regularization term differs in L1 and L2.
In L2, we have:
Here, lambda is the regularization parameter. It is the hyperparameter whose value is optimized for better
results. L2 regularization is also known as weight decay as it forces the weights to decay towards zero
(but not exactly zero).
In L1, we have:
In this, we penalize the absolute value of the weights. Unlike L2, the weights may be reduced to zero
here. Hence, it is very useful when we are trying to compress our model. Otherwise, we usually prefer
L2 over it.

