import os
import cv2 as cv
import numpy as np
import json


def extract_local_features(path, detector):
    img = cv.imread(path)
    kp = detector.detect(img)
    desc = detector.compute(img, kp)
    desc = desc[1]
    return desc


def BOWImgDescExtractor(vocabulary, desc, bf):
    matches = bf.match(desc, vocabulary) #Find the nearest visual words from the vocabulary for each keypoint descriptor.
    histogram = np.zeros((1, vocabulary.shape[0]), dtype=int)
    for m in matches:
        histogram[0, m.trainIdx] += 1
    histogram = histogram / cv.norm(histogram, normType=cv.NORM_L2)
    histogram = np.array(histogram)
    return histogram


def extract_train_descriptors(train_folders, detector):
    # Extract Database
    print('Extracting features...')
    train_descs = np.zeros((0, 128))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            desc = extract_local_features(path, detector)
            if desc is None:
                continue
            train_descs = np.concatenate((train_descs, desc), axis=0)
    return train_descs


def create_vocabulary(train_descs):
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv.BOWKMeansTrainer(70, term_crit, 1, cv.KMEANS_PP_CENTERS)
    vocabulary = trainer.cluster(train_descs.astype(np.float32))
    return vocabulary


def create_index(train_folders, vocabulary, detector, crossCheck=True):
    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))

    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:

            path = os.path.join(folder, file)

            desc = extract_local_features(path, detector)
            # bow_desc = descriptor_extractor.compute(img, kp)  #returns the indices of clusters (words) that exist in both the image and the vocab
            bow_desc = BOWImgDescExtractor(vocabulary, desc, cv.BFMatcher.create(cv.NORM_L2SQR, crossCheck=crossCheck))

            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

            img_paths.append(path)
    return bow_descs, img_paths


def Knn(k, bow_descs_test, bow_descs_trained, train_labels):
    class_predictions = []
    for bow_desc in bow_descs_test:
         distances = np.sum((bow_desc - bow_descs_trained) ** 2, axis=1) #Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
         retrieved_ids = np.argsort(distances)
         retrieved_ids = retrieved_ids[:k]

         neighbors_class = [train_labels[id] for id in retrieved_ids.tolist()]
         prediction = max(set(neighbors_class), key= neighbors_class.count)
         class_predictions.append(prediction) # Make a classification prediction with neighbors
    return class_predictions


def classification_accuracy(predictions, test_labels):
    correct = 0
    for i in range(len(test_labels)):
    	if test_labels[i] == predictions[i]:
    		correct += 1
    accuracy = correct / float(len(test_labels)) * 100.0
    print("Classification accuracy: ", accuracy)
    return accuracy


def labels(img_paths):
    labels = [i.split('\\')[2] for i in img_paths]
    return labels