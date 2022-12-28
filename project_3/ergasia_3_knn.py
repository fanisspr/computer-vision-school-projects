'''
Here's what the code is doing:

1. Create a vocabulary of visual words by clustering extracted SIFT descriptors, or load one.
2. Create a bag-of-words representation-index for each image by counting the number of occurrences of each visual word in the image, or load one.
3. Train a KNN classifier using the bag-of-words representations-train index.
4. Evaluate the classifier on the test set.
5. Calculate the classification accuracy.
'''

import os
import cv2 as cv
import numpy as np
import json
from functions import *


root_dir = os.path.relpath(os.path.dirname(__file__))
train_dir = os.path.join(root_dir, 'imagedb_train')
test_dir = os.path.join(root_dir, 'imagedb_test')

train_folders = [os.path.join(train_dir, subdir)
                 for subdir in os.listdir(train_dir)]
test_folders = [os.path.join(test_dir, subdir)
                for subdir in os.listdir(test_dir)]

sift = cv.SIFT_create()

print("Initiating Knn Classifier")

# create or load vocabulary
vocab_filename = os.path.join(root_dir, 'vocabulary.npy')
if not os.path.exists(vocab_filename):
    vocabulary = create_vocabulary(
        extract_train_descriptors(train_folders, sift))
    np.save(vocab_filename, vocabulary)
    print('Vocabulary is created')
else:
    vocabulary = np.load(os.path.join(root_dir, 'vocabulary.npy'))
    print('Vocabulary is loaded')

# Create train Index or load if exists
index_filename = os.path.join(root_dir, 'index.npy')
index_paths_filename = os.path.join(root_dir, 'index_paths.txt')
if not any([os.path.exists(index_filename), os.path.exists(index_paths_filename)]):
    print('Creating test index...')
    bow_descs_trained, img_paths_train = create_index(
        train_folders, vocabulary)
    np.save(index_filename, bow_descs_trained)
    with open(index_paths_filename, mode='w+') as file:
        json.dump(img_paths_train, file)
    print('Train Index created')
else:
    bow_descs_trained = np.load(index_filename)
    with open(index_paths_filename, mode='r') as file:
        img_paths_train = json.load(file)
    print('Train Index is loaded')

# create test Index or load if exists
if not os.path.exists(os.path.join(root_dir, 'index_test.npy')):
    bow_descs_test, img_paths_test = create_index(
        test_folders, vocabulary, crossCheck=False)
    np.save(os.path.join(root_dir, 'index_test.npy'), bow_descs_test)
    with open(os.path.join(root_dir, 'index_paths_test.txt'), mode='w+') as file:
        json.dump(img_paths_test, file)
    print('Test Index created')
else:
    bow_descs_test = np.load(os.path.join(root_dir, 'index_test.npy'))
    with open(os.path.join(root_dir, 'index_paths_test.txt'), mode='r') as file:
        img_paths_test = json.load(file)
    print('Test Index is loaded')

train_labels = labels(img_paths_train)
test_labels = labels(img_paths_test)

Knn_classes = Knn(5, bow_descs_test, bow_descs_trained, train_labels)

Acc = classification_accuracy(Knn_classes, test_labels)
print("Classification accuracy: ", Acc)

print("Knn classifier finished")
