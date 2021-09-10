---
layout: single
title:  "Image Regconition System with KNN"
date:   2021-09-10 09:09:56 -0500
---
Machine Learning topic: Implement an image recognition system for classifying digits using K-Nearest Neighbor.

## Data

In this blog, I'm going to use the images from the openml `mnist_784`data library. The data set contains 70,000 images of hand-writing digits which ranged from 0 to 9. 

Our main tasks include:
* Building a KNN model and fine tuned the hyperparameter - k value
* Predict the KNN model accuracies with different training size and evaluate the trade off between training size and excecution time
* Evaluate the final model with the confusion matrix

## Data Exploratory

* Let's start with importing the libraries for our mini project

```Python
# Importlibraries.
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

# Set the randomizer seed so the results are the same each time running
np.random.seed(0)

# This's for imbeding the plot in colab or jupyter notebook. I'm running on colab for this mini project.
%matplotlib inline
```

* Then import our dataset from openml. We are going to split our data set into training set, mini training set, development set and test sets for fine tuning the hyperparameters, in this blog, it's the k value.

```Python
# Load the digit data from https://www.openml.org/d/554 or from default local location '~/scikit_learn_data/...'
X, Y = fetch_openml(name='mnist_784', return_X_y=True, cache=False)

# Rescale grayscale values to [0,1].
X = X / 255.0

# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering. 
# And our random seed ensure the results are the same each time running
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

print('data shape: ', X.shape)
print('label shape:', Y.shape)

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[61000:], Y[61000:]
dev_data, dev_labels = X[60000:61000], Y[60000:61000]
train_data, train_labels = X[:60000], Y[:60000]
mini_train_data, mini_train_labels = X[:1000], Y[:1000]
```
  - Result: there are 70,000 records in our data set, each image has been flaten to 1D - array data. 
    * data shape: (70000, 784)
    * label shape: (70000, )

* Let's visualize some of the images from our data set. In particular, let's show a 10 x 10 grid that visualizes 10 examples for each digit. We will use matplotlib `imshow()` for this task. We will also need to reshape our X data from a 1D vector feature to 2D matrix for rendering and able to show the images.

```Python
def show_images(num_examples=10, X=train_data, y=train_labels):
  """ Plot 10 examples of each digit in a 10x10 grid from the training set"""

  # Find the index number for the first 10 examples of each digit
  idx = []
  for i in range(10):
      idx.append([index for index, label in enumerate(y) \
                  if int(label) == i][:num_examples])
  idx = [idx[i][j] for i in range(10) for j in range(num_examples)]

  # Plot the examples, each digit is in a row
  plt.figure(figsize=(10,10))
  plt.suptitle("Example of Training Images", fontsize=16)
  for i in range(10*num_examples):
    plt.subplot(10,num_examples,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[idx][i].reshape(28, 28), cmap=plt.cm.binary)
    if (i % 10 == 0):
      plt.ylabel(f"{y[idx][i]:<8}", rotation=0, fontsize=14)
  plt.show()
  
## Show 10 images for each digit, you could choose to show 15, 20, etc.
show_images(10
```
![Digits]("./others/knn_01_images.png")

<img src="https://github.com/latuyetmai/portfolio/blob/master/others/knn_01_images.png">
  
---
Under construction
