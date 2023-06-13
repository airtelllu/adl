#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# In[2]:


# Load Iris dataset
iris = load_iris()


# In[10]:


# get features and output
X = iris.data
y = iris.target


# In[11]:


# One-hot encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)


# In[12]:


from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


# In[14]:


optimizers = ['sgd', 'adam', 'rmsprop']

for optimizer in optimizers:
    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model with different optimizers
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print results
    print('Optimizer:', optimizer)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)


# In[15]:


import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile model with different optimizers
optimizers = ['sgd', 'adam', 'rmsprop']
for optimizer in optimizers:
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print results
    print('Optimizer:', optimizer)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

# Allow user to input values for the flower attributes
print('\nInput values for the flower attributes:')
sepal_length = float(input('Sepal length (cm): '))
sepal_width = float(input('Sepal width (cm): '))
petal_length = float(input('Petal length (cm): '))
petal_width = float(input('Petal width (cm): '))

# Predict class of flower based on input values
input_values = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_values)
predicted_class = np.argmax(prediction)
class_names = iris.target_names
print('\nPredicted class:', class_names[predicted_class])


# In[16]:


import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(4,), activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Define dictionary of optimizers
optimizers = {
    'sgd': tf.keras.optimizers.SGD(),
    'adam': tf.keras.optimizers.Adam(),
    'rmsprop': tf.keras.optimizers.RMSprop()
}

# Compile model with different optimizers
for optimizer_name, optimizer in optimizers.items():
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Optimizer:', optimizer_name)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

# Estimate memory requirement
size_in_bytes = model.count_params() * 4  # each parameter is a 32-bit float
size_in_mb = size_in_bytes / (1024 * 1024)
print(f'Memory requirement: {size_in_mb:.2f} MB')


# In[ ]:


##Implement Feed-forward Neural Network and train the network with different optimizers and compare the results. 


# In[ ]:

##
##Theory: A Feed Forward Neural Network is an artificial neural network in which the connections
##between nodes does not form a cycle. The opposite of a feed forward neural network is a recurrent
##neural network, in which certain pathways are cycled. The feed forward model is the simplest form of
##neural network as information is only processed in one direction. While the data may pass through
##multiple hidden nodes, it always moves in one direction and never backwards.
##A Feed Forward Neural Network is commonly seen in its simplest form as a single layer perceptron.
##In this model, a series of inputs enter the layer and are multiplied by the weights. Each value is then
##added together to get a sum of the weighted input values. If the sum of the values is above a specific
##threshold, usually set at zero, the value produced is often 1, whereas if the sum falls below the threshold,
##the output value is -1. The single layer perceptron is an important model of feed forward neural
##networks and is often used in classification tasks. Furthermore, single layer perceptron’s can
##incorporate aspects of machine learning.
##
