---
layout: single
title:  "Image Regconition System with KNN"
date:   2021-09-10 09:09:56 -0500
---
Machine Learning topic: Implement an image recognition system for classifying digits using K-Nearest Neighbor.

## Data

In this post, I'm going to use the images from the openml `mnist_784`data library. The data set contains 70,000 images of hand-writing digits which ranged from 0 to 9. 

Our main tasks include:
* Building a KNN model and fine tuned the hyperparameter - k value
* Predict the KNN model accuracies with different training size and evaluate the trade off between training size and excecution time
* Evaluate the final model with the confusion matrix

## Data Exploratory

* Let's start with importing the libraries for our mini project.

```py
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

* Then import our dataset from `openml`. We are going to split our data set into training set, mini training set, development set and test sets for fine tuning hyperparameters, in this post, it's the k value.

```py
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

```py
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
show_images(10)
```
![](./img/knn_01_images.png)

<img src="{{ site.baseurl }}/others/knn_01_images.png">

## Evaluating different choices of k

* I'm going to produce k-Nearest Neighbors models with different values, (i.e., k = 1, 3, 5, 7, 9) and will evaluate the accuracy of each model. We will use the following:
  - Train the data on mini training set
  - Evaluate performance on the devlopment set
 
* Results:
  - k = 1 giving the best performance with higest accuracy
  - The classification report for k=1 show that number 8 is the most difficult for the 1-Nearest Neighbor model to classify correctly as it has the lowest recall comparing to the other numbers. It also has the lowest F1 score
 
* Reviews: Definitions for Precision, Recall and F1 score
  - Precision = True Positive/ (True Positive + False Positive)
  - Recall = True Positive/ (True Positive + False Positive)
  - F1-score = 2*(Precision * Recall) / (Precision + Recall)

```py
def k_value_evaluation(k_values=1, X_train=mini_train_data, y_train=mini_train_labels,
       X_test=dev_data, y_test=dev_labels):
  """This function take in a list of k_values, and return the accuracy of each
  k_Nearest Neighbors models with k in the list. If 1 is in the k_values list, 
  it will also print out the classification report for k=1 KNN model 
  """
  # Convert y from text to numeric
  y_train = y_train.astype(int)
  y_test = y_test.astype(int)

  # Training and Prediction with different k value
  for k_value in list(k_values):
    # Train model
    knn = KNeighborsClassifier(n_neighbors=k_value, p=2, metric='minkowski')
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Performance evaluation
    accuracy = knn.score(X_test, y_test)
    if k_value == 1:
      class_report_k1 = classification_report(y_test, y_pred)

    # Print out accuracy results  
    if k_value == k_values[0]:
      print("\n Accuracy of each KNN model:\n")
      print("| K Value | Prediction Accuracy |")
    print(f"| {k_value:>7} | {accuracy:>19.3f} |")

  # Print out the classification report for k = 1
  print("\n\n Classification report for k=1:\n\n", class_report_k1)
    
  ### K-value Evaluation ###

k_values = [1, 3, 5, 7, 9]
k_value_evaluation(k_values)
```

* Output:
![](https://github.com/latuyetmai/portfolio/blob/master/others/knn_02_k-values.png)
<img src="{{ site.baseurl }}/others/knn_02_k-values.png">
  
## Examing the importance of training size and excecution time

* In this section, I will evaluate the effect of training size on the 1-Nearest Neighbor models' accuracy and excecution time.
* The time is classified into:
  - Time for training for each model
  - Time for measuring accuracy for each model
* Notes:
  - Use the training set for training the model
  - Evaluate on the development set
  - Use `time.time()` to measure elapsed time of operations 
* Results:
  - The accuracy of the models increase as the traing size increases.
  - However, the evaluating time also increases with running time of O(N) as the training size grows.
  - As expected, the evaluating time is much longer than training time. This is because KNN is a lazy learning algorithm. 
  - In a real world situation, when we have extensive amount of data, we will need to decide which trade off should we take. Do we want to have the best accuracy model but it will require extensive computer resources? Or are we good with a relatively correct model (i.e. with accuracy ~90 - 95%), but it's running faster and less constrained on the resources required? 
  
  ```py
  def time_evaluation(train_sizes, accuracies, train_times, eval_times, 
       X_train=train_data, y_train=train_labels, X_test=dev_data, 
       y_test=dev_labels): 
       #, plot="yes", print_result="yes", get_accuracy="no"):
       
    # Convert y from text to numeric
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # KNN model evaluation with different training sizes 
    for train_size in list(train_sizes):
      # Train model and measure time for training
      knn = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

      start_time = time.time()
      knn.fit(X_train[:train_size], y_train[:train_size])
      end_time = time.time()
      train_time = end_time - start_time
      train_times.append(train_time)

      # Predict
      y_pred = knn.predict(X_test)

      # Performance evaluation
      start_time = time.time()
      accuracy = knn.score(X_test, y_test)
      end_time = time.time()
      eval_time = end_time - start_time
      eval_times.append(eval_time)
      accuracies.append(accuracy)

      # Print out the result:
      # if print_result == "yes":
      if train_size == train_sizes[0]:
        print("\n Accuracy and Excecution Time of KNN model with k=1:\n")
        print("| Training Size | Prediction Accuracy | Training Time (s) | Evaluating Time (s) |")
      print(f"| {train_size:>13} | {accuracy:>19.3f} | {train_time:>17.3f} | {eval_time:>19.3f} |")

    # Plot training time and evaluating time
    # if plot == "yes":
    print("")
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_times, color='blue', label="Training Time")
    plt.plot(train_sizes, eval_times, color='red', label="Calculating Accuracy Time")
    plt.xlabel("Training Size")
    plt.ylabel("Time (s)")
    plt.title("1-Nearest Neighbor, Training and Evaluating Time")
    plt.legend(loc="upper left")
    plt.show()


    ### Training Size and Running Time Evaluation ###

  train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
  accuracies = []
  train_times = []
  eval_times = []
  time_evaluation(train_sizes, accuracies, train_times, eval_times)
  ```

* Output:
![](https://github.com/latuyetmai/portfolio/blob/master/others/knn_03_train-size.png)
<img src="{{ site.baseurl }}/others/knn_03_train-size.png">

##  Evaluating if we should obtain additional data

* In this section, we will use linear regression to predict what happens to the 1-Nearest Neighbor model's accuracy as we increase the training size.

* Four models will be ran with different transformations to our X and Y data, in order to find the linear regression model that could best predict the 1-KNN accuracies. I will only show the results of the first and the last model:
  - The first model is the base, where no transformation will be applied to X & Y
  - In the last model, I will transform X to log10 and Y to probability using logit and logistic function

* Notes:
  - We will print the 1-KNN accuracies predicted values for the training set sizes 60000, 120000 and 1000000.
  - Transformations applied to the output variables (1-KNN accuracies) using logit and logistic function:
    * [logistic](https://en.wikipedia.org/wiki/Logistic_function): $\frac{1}{1 + e^{-x}} = \frac{e^x}{1+e^x}$ which takes numbers in $\[\infty,-\infty\]$ and outputs numbers in $(0, 1)$.
    * [logit](https://en.wikipedia.org/wiki/Logit): $log(\frac{p}{1 - p})$ which takes numbers between $(0, 1)$ and outputs numbers between $\[\infty,-\infty\]$.
    * It also happens that $x = logit(p)$ is the same thing as $logiistic(x)=p$.
    * R2-score will be manually calculated instead of using `.score()` method from sklearn

* Result:
  - After transformation both X and Y values, our regression model shows significant improvement in the predition of 1-KNN accuracies with R2-score increases from 0.418 to 0.99.
  - The last model shows that by increasing our training size to 60000, 120000 or 1000000, the accuracies gained are just slightly improve and we do not gain as much benefits by doing this. In my opinion, the training size at 12,000 records should be good enough for 1-KNN predicting at accuracy ~95%, and we should not obtain more data in this case. 
  
  ```py
  def evaluate_train_size(X_train=train_sizes, y_train=accuracies, 
       X_pred=[60000, 120000, 1000000]):

    ## Model 1, No Transformation
    X_train = np.array(X_train)[:, np.newaxis]
    y_train = np.array(y_train)
    X_pred = np.array(X_pred)[:, np.newaxis]
    # Get Model 1 predicted accuracies, print out $R^2$ score, and plot
    print("Model 1 - No Transformation - Results:")
    y_hat_mod1, y_pred_mod1 = regress_model(X_train, y_train, X_pred)
    print_predicts(X_pred, y_pred_mod1)
    plot_accuracies(X_train, y_train, y_hat_mod1)

    ## Model 2, Transform X to log10 scale:
    X_train_transform = np.log10(X_train)
    X_pred_transform = np.log10(X_pred)
    # Get Model 2 predicted accuracies, print out $R^2$ score, and plot
    print("\n\nModel 2 - Transform X to Log10 - Results:")
    y_hat_mod2, y_pred_mod2 = regress_model(X_train_transform, y_train, 
                                            X_pred_transform)
    print_predicts(X_pred, y_pred_mod2)
    plot_accuracies(X_train, y_train, y_hat_mod2)

    ## Model 3, Transform y to probability:
    y_train_transform = np.log((y_train/(1-y_train)))

    # Get Model 3 predicted accuracies, print out $R^2$ score, and plot
    print("\n\nModel 3 - Transform y to Probability - Results:")
    y_hat_trans_mod3, y_pred_trans_mod3 = regress_model(X_train, y_train_transform, 
                                                        X_pred)
    # Transfrom back the result from probability to real number
    y_hat_mod3 = np.exp(y_hat_trans_mod3)/(np.exp(y_hat_trans_mod3)+1)
    y_pred_mod3 = np.exp(y_pred_trans_mod3)/(np.exp(y_pred_trans_mod3)+1)
    # Print prediction & plot
    print_predicts(X_pred, y_pred_mod3)
    plot_accuracies(X_train, y_train, y_hat_mod3)

    ## Model 4, Tansform X to log10 and Transform y to probability:
    print("\n\nModel 4 - Transform X to Log10, Transform y to Probability - Results:")
    y_hat_trans_mod4, y_pred_trans_mod4 = regress_model(X_train_transform, 
                                                        y_train_transform, 
                                                        X_pred_transform)
    # Transfrom back the result from probability to real number
    y_hat_mod4 = np.exp(y_hat_trans_mod4)/(np.exp(y_hat_trans_mod4)+1)
    y_pred_mod4 = np.exp(y_pred_trans_mod4)/(np.exp(y_pred_trans_mod4)+1)
    # Print prediction & plot
    print_predicts(X_pred, y_pred_mod4)
    plot_accuracies(X_train, y_train, y_hat_mod4)

  def regress_model(X_train, y_train, X_pred):
    """Runing Linear Regression Model, print out R-squared and return predicted values"""

    # Fit linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Calculate $R^2$ of the model
    y_hat = lm.predict(X_train)
    R_squared = 1 - ((y_train - y_hat)**2).sum()/((y_train -y_train.mean())**2).sum()

    # Predict accuracies for training set sizes 60000, 120000, and 1000000.
    y_pred = lm.predict(X_pred)

    # Prind out $R^2$ value
    print(f"R2 score: {R_squared:.3f}")

    return y_hat, y_pred


  def print_predicts(X_pred, y_pred):
    print("Predicted Value:\n\n| Traing Set Size | Predicted Accuracy |")
    for i in range(y_pred.size):
      print(f"| {X_pred[i][0]:>15} | {y_pred[i]:>18.3f} |")
    print("")


  def plot_accuracies(X_train, y_train, y_hat):
    """Plot actual and predicted accuracies from regression model """
    # Plot actual accuracies and predicted accuracies of 1_Nearest Neighbor Model
    plt.figure(figsize=(8,6))
    plt.plot(X_train[:, 0], y_train, color='blue', label="Actual Accuracies")
    plt.plot(X_train[:, 0], y_hat, color='red', label="Predicted Accuracies" )
    plt.xlim(X_train[:,0].min(), X_train[:,0].max() )
    plt.plot()
    plt.legend()
    plt.title("1-Nearest Neighbor Model - Actual & Predicted Accuracies")
    plt.xlabel("Traing Set Size")
    plt.ylabel("Accuracies")
    plt.show()

  ### Evaluate Training Size ###

  evaluate_train_size()
  ```
  
* Output:
![](https://github.com/latuyetmai/portfolio/blob/master/others/knn_04_train-size2.png)
<img src="{{ site.baseurl }}/others/knn_04_train-size2.png">

![](https://github.com/latuyetmai/portfolio/blob/master/others/knn_05_train-size3.png)
<img src="{{ site.baseurl }}/others/knn_05_train-size3.png">

## Evaluation KNN model with the confusion matrix

* In the section, we will display the confusion matrix for the 1-Nearest model and find out which digit does the model most often confuse with another digit.
* We will show the examples of the misclassified digit
* Notes:
  - We will using mini train set and evaluate performance on the development set
* Results:
  - From the confusion matrix, we could conclude that number 4 has the highest misclassified result with number 9 by using the 1-KNN model
 
 ```py
 def conf_matrix_eval(k_values=1, X_train=mini_train_data, y_train=mini_train_labels,
       X_test=dev_data, y_test=dev_labels):
       
    # Convert y from text to numeric
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Train model
    knn = KNeighborsClassifier(n_neighbors=k_values, p=2, metric='minkowski')
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Performance evaluation
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Plot confustion matrix
    plot_conf_matrix(conf_matrix)
    print("\n\n")

    ## Show 10 examples of the most misclassified digits
    # Where the actual number is 4, and predicted value is 9
    idx = [index for index, label in enumerate(y_test) if label == 4 and \
                 y_pred[index] == 9][:10]

    # Plot examples of the misclassified digits:
    plt.figure(figsize=(8, 4))
    plt.suptitle("Example of Misclassified Images, Actual=4, Predict=9", fontsize=16)
    for i in range(len(idx)):
      plt.subplot(2, 5, i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(X_test[idx][i].reshape(28, 28), cmap=plt.cm.binary)
    plt.tight_layout()
    plt.show()


  def plot_conf_matrix(conf_matrix):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.matshow(conf_matrix, cmap='YlGn')
    plt.colorbar(im,cax = fig.add_axes([0.93, 0.12, 0.03, 0.75]))
    for i in range(conf_matrix.shape[0]):
      for j in range(conf_matrix.shape[1]):
        ax.text(j, i, conf_matrix[i,j], va='center', ha='center', color='k')
    ax.set_xlabel("Predicted Number", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("1-Nearest Neighbor - Confusion Matrix", fontsize=16)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.show()

  ### Plotting the Confusion Matrix and Fing the most confusing paired digits ###

  conf_matrix_eval()
 ```

* Output:
![](https://github.com/latuyetmai/portfolio/blob/master/others/knn_06_conf_matrix.png)
<img src="{{ site.baseurl }}/others/knn_06_conf_matrix.png">

![](https://github.com/latuyetmai/portfolio/blob/master/others/knn_07_most_confusing.png)
<img src="{{ site.baseurl }}/others/knn_07_most_confusing.png">

## Summary

We have accomplished the following tasks in this post:
* How to use KNN for images classification.
* How to fine tune the k-number hyperparameter for our model
* Finding what should be the decent training size to use and answer the question should we obtain more data for our model.
* Using confusion matrix to identify which digit does the model most often confuse with which digit. 
