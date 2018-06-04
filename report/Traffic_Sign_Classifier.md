
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle
import csv

training_file = 'data/train.p'
validation_file= 'data/valid.p'
testing_file = 'data/test.p'
signnames_file = 'signnames.csv'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Load labels
sign_labels = []
with open(signnames_file, mode='r') as f:
    csv_contents = csv.reader(f, delimiter=',')
    sign_labels = [row[1] for row in csv_contents][1:]
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_valid.shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import numpy as np
# Visualizations will be shown in the notebook.
%matplotlib inline

def get_examples_per_class():
    u, indices = np.unique(y_train, return_index=True)
    examples =[]
    
    for i in range(1, len(indices) + 1):
        img = X_train[indices[i - 1]]
        label = y_train[indices[i - 1]]
        label_string = sign_labels[label]
        examples.append([img, label, label_string])
    
    return examples

def show_class_examples(examples_per_class):
    #Plot an example for each class
    columns = 5
    rows = np.ceil(len(examples_per_class)/columns)

    fig = plt.figure(figsize = (20,30));
    plt.title("Training classes examples")
    plt.axis('off')
    plt.subplots_adjust(top=2)
    for i in range(0, len(examples_per_class)):
        example = examples_per_class[i]
        img = example[0]
        label = example[1]
        label_string = example[2]
        ax = fig.add_subplot(rows, columns, i + 1)
        plt.title("{0} - {1}".format(label, label_string))
        ax.imshow(img) if img.shape[2] == 3 else ax.imshow(img[:,:,0], cmap='gray')
        ax.axis('off')
    fig.tight_layout()
    
        
def plot_examples_per_class(y, title):
    fig = plt.figure(figsize = (20,5));
    plt.title(title)
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    train_examples_per_class = list(zip(unique_elements, counts_elements))
    width = 1/1.5
    x = unique_elements
    y = counts_elements
    plt.bar(x, y, width, color="blue")
        
examples_per_class = get_examples_per_class()
show_class_examples(examples_per_class)
plot_examples_per_class(y_train, "Training classes: counts")
plot_examples_per_class(y_valid, "Validation classes: counts")
```


![png](output_8_0.png)



![png](output_8_1.png)



![png](output_8_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
# Generate some data
import sys
import random
import math
import scipy
import numpy as np
from skimage.transform import warp, AffineTransform

def rotate_image(image, angle=10):
    return scipy.ndimage.rotate(image, random.randint(-angle,angle), reshape=False)

def shear_image(image):
    shear = random.randint(1,2) / 10
    tform = AffineTransform(shear=shear)
    return warp(image, tform, preserve_range=True).astype(int)

def scale_image(image):
    sx = 1 - random.randint(1,2) / 10
    sy = 1 - random.randint(1,2) / 10
    tform = AffineTransform(scale=(sx, sy))
    return warp(image, tform, preserve_range=True).astype(int)

def translate_image(image):
    tx = random.randint(0,3)
    ty = random.randint(0,3)
    tform = AffineTransform(translation=(tx, ty))
    return warp(image, tform, preserve_range=True).astype(int)

def transform_image(image):
    transformations = {
        0: rotate_image,
        1: shear_image,
        2: scale_image,
        3: translate_image
    }
    return transformations[random.randint(0, len(transformations) - 1)](image)

# level the numbers per class
new_trainX = []
new_trainY = []
target_size = 4000
max_size = target_size

def extend_image_set(samples):
    expanded = []
    expanded.extend(samples)

    sample_size = len(samples)
    new_size = sample_size
    while True:
        for image in samples:
            expanded.append(transform_image(image))
            new_size += 1
            if  new_size >= target_size:
                return expanded

sys.stdout.write("Processing images for class: ")

# Concatenate train and validation sets
X_train = np.concatenate((X_train, X_valid))
y_train = np.concatenate((y_train, y_valid))

# Augment data
for i in range(0,43):
    sys.stdout.write("{0}..".format(i))
    samples = X_train[np.where(y_train == i)]
    sample_size = samples.shape[0]
    if (sample_size < target_size):
        samples = np.array(extend_image_set(samples))
    elif (sample_size > max_size):
        samples = samples[0:max_size]
    new_trainY.append(np.full((samples.shape[0]), i).flatten())
    new_trainX.append(samples)
    
sys.stdout.write('done\n')
sys.stdout.flush()

new_sample_length = sum([len(x) for x in new_trainX])
y_train = np.concatenate(new_trainY)
X_train = np.concatenate(new_trainX)

# Split train and validation sets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X_train, y_train = shuffle(X_train, y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Plot new sets
plot_examples_per_class(y_train, "Training classes: new counts")
plot_examples_per_class(y_valid, "Validation classes: new counts")
```

    Processing images for class: 0..1..2..3..4..5..6..7..8..9..10..11..12..13..14..15..16..17..18..19..20..21..22..23..24..25..26..27..28..29..30..31..32..33..34..35..36..37..38..39..40..41..42..done



![png](output_12_1.png)



![png](output_12_2.png)



```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import numpy as np

def toFloat(X):
    return X.astype(np.float32)

def normalise(X):
    return (X  - 128) / 128

def grayscale(X):
    return np.reshape(np.mean(X, axis=3), (X.shape[0], X.shape[1], X.shape[2], 1))

def preprocess(X):
    return normalise(grayscale(toFloat(X)))

X_train, X_valid, X_test = [preprocess(X) for X in [X_train, X_valid, X_test]]

show_class_examples(get_examples_per_class())
```


![png](output_13_0.png)


### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def model(x, keep_prob):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', name='conv1_0') + conv1_b
    conv1 = tf.nn.relu(conv1, name='conv1_1')
#     conv1 = tf.nn.dropout(conv1, keep_prob) Adding dropout to the convolutional layers is not helpful
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv1_2')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID', name='conv2_0') + conv2_b
    conv2 = tf.nn.relu(conv2, name='conv2_1')
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='conv2_2')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 200.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 200), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(200))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1, name='fc1')
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 200. Output = 120.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(200, 120), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(120))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2, name='fc2')
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # Layer 5: Fully Connected. Input = 200. Output = 84.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(84))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b
    fc3 = tf.nn.relu(fc3, name='fc3')
    fc3 = tf.nn.dropout(fc3, keep_prob)

    # Layer 6: Fully Connected. Input = 84. Output = 43.
    fc4_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc4_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc3, fc4_W) + fc4_b

    return logits, tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc3_W) + tf.nn.l2_loss(fc4_W)
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from sklearn.utils import shuffle

# EPOCHS = 10
# BATCH_SIZE = 128
EPOCHS = 60
BATCH_SIZE = 128
KEEP_PROB = 0.5 # Keep probability for dropout <- todo lower
BETA = 0.0001 # Beta for L2 regularization <- todo lower factor 10

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

# Training pipeline
rate = 0.0005
#rate = 0.001

logits, reg = model(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy) + reg * BETA
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train
accuracy = {'training': [], 'validation':[]}
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    highest_accuracy = 0
    for i in range(EPOCHS):
        print("EPOCH {}".format(i+1))
        print("Training...")
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})
            
        print("Evaluating...".format(i+1))
        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        accuracy["training"].append(training_accuracy);
        accuracy["validation"].append(validation_accuracy);
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
        # Only save the highest accuracy
        if validation_accuracy > highest_accuracy:
            highest_accuracy = validation_accuracy
            
            saver.save(sess, './model')
            print("Model saved")
            
        print()



```

    Training...
    
    EPOCH 1
    Training...
    Evaluating...
    Training Accuracy = 0.619
    Validation Accuracy = 0.618
    Model saved
    
    EPOCH 2
    Training...
    Evaluating...
    Training Accuracy = 0.792
    Validation Accuracy = 0.788
    Model saved
    
    EPOCH 3
    Training...
    Evaluating...
    Training Accuracy = 0.856
    Validation Accuracy = 0.851
    Model saved
    
    EPOCH 4
    Training...
    Evaluating...
    Training Accuracy = 0.897
    Validation Accuracy = 0.891
    Model saved
    
    EPOCH 5
    Training...
    Evaluating...
    Training Accuracy = 0.916
    Validation Accuracy = 0.909
    Model saved
    
    EPOCH 6
    Training...
    Evaluating...
    Training Accuracy = 0.923
    Validation Accuracy = 0.915
    Model saved
    
    EPOCH 7
    Training...
    Evaluating...
    Training Accuracy = 0.937
    Validation Accuracy = 0.931
    Model saved
    
    EPOCH 8
    Training...
    Evaluating...
    Training Accuracy = 0.946
    Validation Accuracy = 0.940
    Model saved
    
    EPOCH 9
    Training...
    Evaluating...
    Training Accuracy = 0.953
    Validation Accuracy = 0.948
    Model saved
    
    EPOCH 10
    Training...
    Evaluating...
    Training Accuracy = 0.957
    Validation Accuracy = 0.950
    Model saved
    
    EPOCH 11
    Training...
    Evaluating...
    Training Accuracy = 0.963
    Validation Accuracy = 0.956
    Model saved
    
    EPOCH 12
    Training...
    Evaluating...
    Training Accuracy = 0.967
    Validation Accuracy = 0.960
    Model saved
    
    EPOCH 13
    Training...
    Evaluating...
    Training Accuracy = 0.975
    Validation Accuracy = 0.969
    Model saved
    
    EPOCH 14
    Training...
    Evaluating...
    Training Accuracy = 0.975
    Validation Accuracy = 0.967
    
    EPOCH 15
    Training...
    Evaluating...
    Training Accuracy = 0.979
    Validation Accuracy = 0.971
    Model saved
    
    EPOCH 16
    Training...
    Evaluating...
    Training Accuracy = 0.980
    Validation Accuracy = 0.973
    Model saved
    
    EPOCH 17
    Training...
    Evaluating...
    Training Accuracy = 0.983
    Validation Accuracy = 0.976
    Model saved
    
    EPOCH 18
    Training...
    Evaluating...
    Training Accuracy = 0.984
    Validation Accuracy = 0.977
    Model saved
    
    EPOCH 19
    Training...
    Evaluating...
    Training Accuracy = 0.985
    Validation Accuracy = 0.977
    Model saved
    
    EPOCH 20
    Training...
    Evaluating...
    Training Accuracy = 0.986
    Validation Accuracy = 0.979
    Model saved
    
    EPOCH 21
    Training...
    Evaluating...
    Training Accuracy = 0.986
    Validation Accuracy = 0.979
    Model saved
    
    EPOCH 22
    Training...
    Evaluating...
    Training Accuracy = 0.988
    Validation Accuracy = 0.981
    Model saved
    
    EPOCH 23
    Training...
    Evaluating...
    Training Accuracy = 0.987
    Validation Accuracy = 0.980
    
    EPOCH 24
    Training...
    Evaluating...
    Training Accuracy = 0.990
    Validation Accuracy = 0.982
    Model saved
    
    EPOCH 25
    Training...
    Evaluating...
    Training Accuracy = 0.987
    Validation Accuracy = 0.980
    
    EPOCH 26
    Training...
    Evaluating...
    Training Accuracy = 0.990
    Validation Accuracy = 0.982
    Model saved
    
    EPOCH 27
    Training...
    Evaluating...
    Training Accuracy = 0.989
    Validation Accuracy = 0.982
    
    EPOCH 28
    Training...
    Evaluating...
    Training Accuracy = 0.991
    Validation Accuracy = 0.983
    Model saved
    
    EPOCH 29
    Training...
    Evaluating...
    Training Accuracy = 0.990
    Validation Accuracy = 0.983
    
    EPOCH 30
    Training...
    Evaluating...
    Training Accuracy = 0.990
    Validation Accuracy = 0.982
    
    EPOCH 31
    Training...
    Evaluating...
    Training Accuracy = 0.991
    Validation Accuracy = 0.983
    Model saved
    
    EPOCH 32
    Training...
    Evaluating...
    Training Accuracy = 0.991
    Validation Accuracy = 0.984
    Model saved
    
    EPOCH 33
    Training...
    Evaluating...
    Training Accuracy = 0.992
    Validation Accuracy = 0.984
    Model saved
    
    EPOCH 34
    Training...
    Evaluating...
    Training Accuracy = 0.992
    Validation Accuracy = 0.985
    Model saved
    
    EPOCH 35
    Training...
    Evaluating...
    Training Accuracy = 0.993
    Validation Accuracy = 0.986
    Model saved
    
    EPOCH 36
    Training...
    Evaluating...
    Training Accuracy = 0.993
    Validation Accuracy = 0.985
    
    EPOCH 37
    Training...
    Evaluating...
    Training Accuracy = 0.992
    Validation Accuracy = 0.985
    
    EPOCH 38
    Training...
    Evaluating...
    Training Accuracy = 0.992
    Validation Accuracy = 0.985
    
    EPOCH 39
    Training...
    Evaluating...
    Training Accuracy = 0.993
    Validation Accuracy = 0.985
    
    EPOCH 40
    Training...
    Evaluating...
    Training Accuracy = 0.994
    Validation Accuracy = 0.987
    Model saved
    
    EPOCH 41
    Training...
    Evaluating...
    Training Accuracy = 0.994
    Validation Accuracy = 0.987
    
    EPOCH 42
    Training...
    Evaluating...
    Training Accuracy = 0.994
    Validation Accuracy = 0.986
    
    EPOCH 43
    Training...
    Evaluating...
    Training Accuracy = 0.993
    Validation Accuracy = 0.985
    
    EPOCH 44
    Training...
    Evaluating...
    Training Accuracy = 0.993
    Validation Accuracy = 0.985
    
    EPOCH 45
    Training...
    Evaluating...
    Training Accuracy = 0.994
    Validation Accuracy = 0.987
    
    EPOCH 46
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    Model saved
    
    EPOCH 47
    Training...
    Evaluating...
    Training Accuracy = 0.994
    Validation Accuracy = 0.986
    
    EPOCH 48
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.988
    Model saved
    
    EPOCH 49
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 50
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 51
    Training...
    Evaluating...
    Training Accuracy = 0.994
    Validation Accuracy = 0.987
    
    EPOCH 52
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 53
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 54
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.988
    
    EPOCH 55
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 56
    Training...
    Evaluating...
    Training Accuracy = 0.996
    Validation Accuracy = 0.988
    
    EPOCH 57
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 58
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.988
    
    EPOCH 59
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    
    EPOCH 60
    Training...
    Evaluating...
    Training Accuracy = 0.995
    Validation Accuracy = 0.987
    



```python
plt.plot([100 * x for x in accuracy["training"]])
plt.plot([100 * x for x in accuracy["validation"]])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
```


![png](output_19_0.png)



```python
with tf.Session() as sess:
    saver.restore(sess, "./model")
    print("Test accuracy: ", evaluate(X_test, y_test))
```

    Test accuracy:  0.944655581797


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
from os import listdir
from os.path import isfile, join
from PIL import Image

def resize(img):
    IMG_SIZE = 32, 32
    width, height = img.size

    if width > height:
        delta = width - height
        left = int(delta/2)
        upper = 0
        right = height + left
        lower = height
    else:
        delta = height - width
        left = 0
        upper = int(delta/2)
        right = width
        lower = width + upper

    img = img.crop((left, upper, right, lower)).convert(mode='RGB')
    img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
    return img

# Load file names
img_path = './german_sign_examples'
files = [img for img in [join(img_path, f) for f in listdir(img_path)] if isfile(img) and img.endswith(".png")]

# Load images and convert to RGB 32x32
adjusted_images = [resize(Image.open(img)) for img in files]

# Pre-process image for cnn
preprocessed_images = preprocess(np.array([np.array(image) for image in adjusted_images]))

```

### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
import re

def plot_prediction_results(predictions):
    predictions = np.argmax(predictions_per_logit, axis=1)
    
    columns = 2
    rows = len(files)

    fig = plt.figure(figsize = (20,30));
    plt.title("Predictions")
    plt.axis('off')
    plt.subplots_adjust(top=2)
    
    for i in range(0, len(predictions) * 2, columns):    
        ax = fig.add_subplot(rows, columns, i + 1)
        plt.title("Input image ({0})".format(re.search('/(\d*)\.', files[i//2]).group(1)))
        ax.imshow(adjusted_images[i // 2])
        ax.axis('off')
        
        example = examples_per_class[predictions[i // 2]]
        img = example[0]
        label = example[1]
        label_string = example[2]
        
        ax = fig.add_subplot(rows, columns, i + 2)
        plt.title("{0} - {1}".format(label, label_string))
        ax.imshow(img)
        ax.axis('off')

predictions = []
predictions_per_logit = None
with tf.Session() as sess:
    saver.restore(sess, "./model")
    predictions_per_logit = sess.run(logits, feed_dict={x: preprocessed_images, keep_prob: 1})
    predictions = np.argmax(predictions_per_logit, axis=1)
    plot_prediction_results(predictions)
```


![png](output_25_0.png)


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

# File names indicate the actual class
actuals = [int(re.search('/(\d*)\.', file).group(1)) for file in files]

print("Accuracy: {0}%".format(len([(a, b) for (a,b) in zip(actuals, predictions) if a == b]) / len(actuals) * 100))
```

    Accuracy: 100.0%


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

K = 5
with tf.Session() as sess:
    saver.restore(sess, "./model")
    predictions_per_logit_softmax = tf.nn.softmax(predictions_per_logit)
    topK = sess.run(tf.nn.top_k(predictions_per_logit_softmax, k=K))
    
    
    fig = plt.figure(figsize = (20,30));
    plt.axis('off')
    plt.subplots_adjust(top=2)
    
    rows = len(files)
    columns = 2
    
    
    for i in range(len(files)):
        classes = []
        percentages = []
        for j in range(K):
            percentages.append(topK[0][i][j] * 100)
            label = topK[1][i][j]
            label_string = sign_labels[label]
            classes.append("{0} ({1})".format(label_string, label))
            
        ax = fig.add_subplot(rows, columns, i * 2 + 1)

        y_pos = np.arange(K)
        
        for k, v in enumerate(percentages):
            ax.text(v + 3 if v < 80 else 80, k, "{:.5f}%".format(v), color='blue', fontweight='bold')

        ax.barh(y_pos, percentages, align='center', color='green', ecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel('Percentage')
#         ax.set_ylabel('Class')
        label = int(re.search('.*/(\d*).*', files[i]).group(1))
        label_string = sign_labels[label]
        ax.set_title("{0} ({1})".format(label_string, label))
        
        
        ax = fig.add_subplot(rows, columns, i * 2 + 2)
        ax.set_title(re.search('.*/(.*)', files[i]).group(1))
        ax.imshow(adjusted_images[i])
        ax.axis('off')
```


![png](output_30_0.png)


## Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="../visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1, title=""):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess, feed_dict={x : image_input, keep_prob:1})
    featuremaps = activation.shape[3]

    columns = 8
    rows = math.ceil(featuremaps / columns)
    fig = plt.figure(plt_num, figsize=(columns * 2, rows * 2))
    plt.title(title, fontdict={'fontweight':'bold'}, loc='left', y=1.08)
    plt.axis('off')
    
    for featuremap in range(featuremaps):
        ax = fig.add_subplot(rows, columns, featuremap + 1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            ax.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
#     fig.tight_layout()
```


```python
with tf.Session() as sess:
    saver.restore(sess, "./model")
    names = ['conv1_0:0', 'conv1_1:0', 'conv1_2:0', 'conv2_0:0', 'conv2_1:0', 'conv2_2:0']

    input_img = preprocessed_images[1:2,:,:,:]
    
    fig = plt.figure(0, figsize=(2, 2))
    plt.title("Input", fontdict={'fontweight':'bold'}, loc='left', y=1.08)
    plt.axis('off')
    plt.imshow(input_img[0,:,:,0], cmap="gray")
    
    for i, name in enumerate(names):
        tensor = sess.graph.get_tensor_by_name(name)
        outputFeatureMap(input_img, tensor, plt_num=i + 1, title=name)
```


![png](output_35_0.png)



![png](output_35_1.png)



![png](output_35_2.png)



![png](output_35_3.png)



![png](output_35_4.png)



![png](output_35_5.png)



![png](output_35_6.png)



```python

```
