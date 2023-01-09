'''
Here's what the code is doing:

1. Create a vocabulary of visual words by clustering extracted SIFT descriptors, or load one.
2. Create a bag-of-words representation-index for each image by counting the number of occurrences of each visual word in the image, or load one.
3. Train a linear SVM classifier for each class.
'''

import os
import cv2 as cv
import numpy as np
import json
from functions import *


root_dir = os.path.relpath(os.path.dirname(__file__))
index_dir = os.path.join(root_dir, 'index')
models_dir = os.path.join(root_dir, 'models')
train_dir = os.path.join(root_dir, 'imagedb_train')
train_folders = [os.path.join(train_dir, subdir)
                 for subdir in os.listdir(train_dir)]

sift = cv.SIFT_create()

# create or load vocabulary
vocab_filename = os.path.join(index_dir, 'vocabulary.npy')
if not os.path.exists(vocab_filename):
    print('Creating vocabulary...')
    vocabulary = create_vocabulary(
        extract_train_descriptors(train_folders, sift))
    np.save(vocab_filename, vocabulary)
    print('Vocabulary is created')
else:
    vocabulary = np.load(os.path.join(index_dir, 'vocabulary.npy'))
    print('Vocabulary is loaded')

# Create train Index or load if exists
index_filename = os.path.join(index_dir, 'index.npy')
index_paths_filename = os.path.join(index_dir, 'index_paths.txt')
if not all([os.path.exists(index_filename), os.path.exists(index_paths_filename)]):
    print('Creating train index...')
    bow_descs, img_paths = create_index(train_folders, vocabulary, sift)
    np.save(index_filename, bow_descs)
    with open(index_paths_filename, mode='w+') as file:
        json.dump(img_paths, file)
    print('Train Index created')
else:
    bow_descs = np.load(index_filename)
    with open(index_paths_filename, mode='r') as file:
        img_paths = json.load(file)
    print('Train Index is loaded')

classes = os.listdir(train_dir)

# Train SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
for cls in classes:
    print('Training svm_' + cls)
    tlabels = np.array([cls in a for a in img_paths], np.int32)
    svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, tlabels)
    svm.save(os.path.join(models_dir, 'svm_') + cls)

print('Training finished')
