# Classify clothing items Using TensorFlow 2.0

![Clothes](clothes.jpg)

This project classifies lothes using Deep Learning and [Tensorflow](https://www.tensorflow.org) 2.0. 

### Data Reference:

Fashion training set consists of 70,000 images divided into 60,000 training and 10,000 testing samples. Dataset sample consists of 28x28 grayscale image, associated with a label from 10 classes.

The 10 classes are as follows:

0 => T-shirt/top
1 => Trouser
2 => Pullover
3 => Dress
4 => Coat
5 => Sandal
6 => Shirt
7 => Sneaker
8 => Bag
9 => Ankle boot

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.


### Step 1: Import TensorFlow and Python Libraries


```
!pip install tensorflow-gpu==2.0.0.alpha0
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Step 2: Import the dataset

You will need to mount your drive using the following commands:
For more information regarding mounting, please check this out [here](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory).


```
from google.colab import drive
drive.mount('/content/drive')
```

Upload the data file from Kaggle to your Google drive and then access it

The dataframes for both training and testing datasets 
```
fashion_train_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Fashion Dataset/fashion-mnist_train.csv',sep=',')
fashion_test_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Fashion Dataset/fashion-mnist_test.csv', sep = ',')
```

Alternatively, you can use the same dataset made readily available by keras Using the following lines of code:
```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
```

Get more information about your dataset
```
diabetes.info()
diabetes.describe()
diabetes.head(10)
diabetes.tail(10)
```

### Step 3: Visualize the dataset using Seaborn, a python library
See more steps in the colab.

### Step 4: Create testing and training data set and clean the data. 
See steps in the colab.

### Step 5: Train the Model. 
See steps in the colab.

### Step 6: Evaluate the Model. 
See steps in the colab.

### Step 7: Improve the Model
If you are not satisfied with the results, then you can increase the number of independent variables and retrain the same model. See steps in the colab.
