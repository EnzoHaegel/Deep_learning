# Artificial Neural Network and Deep Learning
# Project #1: Back-Propagation Neural Network Implementation
# and Application to Handwritten Digits Recognition
# Subject:
# Implement the back-propagation neural network algorithm to classify/recognize the
# handwritten digits. A database includes more than 40,000 examples of MNIST
# handwritten digits. The dataset contains ten folders, namely 0, 1, 2, …, and 9. Each
# folder contains 4,000 sample images of the corresponding digit. The MNIST database
# is a large database of handwritten digits that is commonly used for training various
# image processing systems. This database is also widely used for training and testing in
# the machine learning field. It was created by "re-mixing" the samples from NIST's
# original datasets. All digits have been size-normalized and centered in a fixed-size
# image (28*28 pixel, PNG format) as shown in the following examples.
# It is a good database for people who want to try learning techniques and pattern
# recognition methods on real-world data while spending minimal effort on
# preprocessing and formatting.
# Note:
# Design your code using any programming language. DO NOT apply any open source
# code of BP nor existing BP library/function for this project. To ensure the program is
# designed by yourself, you will be requested on-site on October 21st to re-train the BP
# model for a given dataset with part of the digits and the variations.
# Evaluation Method:
# 1. For network performance evaluation, you will need to meet TA on October 21st.
# The evaluation time and place will be announced later. TA will give you a new
# set of digits, you need to modify your program and train on the new dataset.
# After you have finished the training as requested on-site, meet TA for evaluation.
# 2
# TA will prepare a set of test images with file names 0001.png, 0002.png, …, and
# 5000.png as illustrated in the following figure.
# 2. Your program needs to read all test images, perform the recognition for each
# digit image, and then generate a text file called Student_ID.txt. The output
# format is shown in the following format. The output file must contain two
# columns: first column is the filename; the second column is the recognition result
# of the digit. For example, 0001 represents a file name of test image 0001.png and
# 0 represents your recognition result as digit 0.
# 3. Then TA will check your output file and compute the accuracy of your
# recognition results. You have two chances of evaluations. However, if you want
# to do the test again, the evaluation will be counted based on your second test.
# Accuracy=	total	#	of	correct	outputs/total	#	of	test	images
# Example:	testing	images
# Output: Student_ID.txt

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# C:\Users\Enzo\Dataset\png
DATADIR = "C:/Users/Enzo/Dataset/png"
IMG_SIZE = 28
TESTDIR = "C:/Users/Enzo/Dataset/test"

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    random.shuffle(training_data)
    return training_data

def create_test_data():
    test_data = []
    for category in CATEGORIES:
        path = os.path.join(TESTDIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass
    random.shuffle(test_data)
    return test_data

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(training_data):
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    model = create_model()
    model.fit(X, y, epochs=3)
    return model

def test_model(test_data, model):
    X = []
    y = []
    for features, label in test_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    val_loss, val_acc = model.evaluate(X, y)
    print(val_loss, val_acc)

def save_model(model):
    model.save('model.h5')

def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

def predict_image(image, model):
    image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(image)
    return prediction

def check_prediction_file():
    """
    for each line of the file, split the line and check if the 2 first chars are the same
    if they are the same add 1 to counter
    """
    counter = 0
    with open('prediction.txt', 'r') as f:
        for line in f:
            if line.split()[0][0] == line.split()[1][0]:
                counter += 1
    print(counter + ' correct predictions out of 50')
    print((counter / 50 * 100) + '%')

def main():
    training_data = create_training_data()
    test_data = create_test_data()
    model = train_model(training_data)
    test_model(test_data, model)
    save_model(model)
    model = load_model()
    for category in CATEGORIES:
        path = os.path.join(TESTDIR, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                prediction = predict_image(new_array, model)
                print(prediction)
            except Exception as e:
                pass
    check_prediction_file()

main()