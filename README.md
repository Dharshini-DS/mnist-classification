# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![dl1](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/2ae212fd-4a8d-4e5d-8047-325c6bd69ea0)

## Neural Network Model

<img width="640" alt="dl2" src="https://github.com/Dharshini-DS/mnist-classification/assets/93427345/a37099df-67bc-40af-856f-547c2ed0a8d9">

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP 5:
Split the data into train and test

### STEP 6:
Build the convolutional neural network model

### STEP 7:
Train the model with the training data

### STEP 8:
Plot the performance plot

### STEP 9:
Evaluate the model with the testing data

### STEP 10:
Fit the model and predict the single input

## PROGRAM
```
Developed By: Dharshini DS
Reg No: 212221230022
```
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="softmax")
])
model.compile(loss="categorical_crossentropy", metrics='accuracy',optimizer="adam")
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))
img = image.load_img('imgfive.png')
img = image.load_img('imgfive.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)     
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![1](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/4befb7bb-a11f-4462-ba69-4e67ce20e2bd)

![2](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/e4697a3f-1d23-46c3-82a1-207341a6b45d)

### Classification Report

![3](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/1f4224fe-9200-48b1-a9b0-b3f4a53da2c6)

### Confusion Matrix

![4](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/ec17638b-d138-4539-affd-1e465a2d6416)

### New Sample Data Prediction

![5](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/8e3fc0c2-6a78-4811-9a84-f438f6cc62e2)

![6](https://github.com/Dharshini-DS/mnist-classification/assets/93427345/02f23be1-8b31-431c-9dc4-c9def39e892a)

## RESULT

Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
