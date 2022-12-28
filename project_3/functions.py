import os
import cv2 as cv
import numpy as np
import json


def extract_local_features(path: str, detector) -> np.ndarray:
    """
    Extract local features (keypoints and their corresponding descriptors) from the given image.

    Parameters:
    path (str): The path to the image.
    detector (cv2.FeatureDetector): The detector used to detect keypoints in the image.

    Returns:
    np.ndarray: The descriptors for the keypoints.
    """
    img = cv.imread(path)
    kp = detector.detect(img)
    desc = detector.compute(img, kp)
    desc = desc[1]
    return desc


def BOWImgDescExtractor(vocabulary: np.ndarray, desc: np.ndarray, bf) -> np.ndarray:
    """
    Extract a bag-of-words descriptor for the given local features using the given vocabulary and descriptor extractor.

    Parameters:
    vocabulary (np.ndarray): The vocabulary used to create the bag-of-words descriptor.
    desc (np.ndarray): The local features (keypoints and their corresponding descriptors).
    bf (cv2.DescriptorMatcher): The descriptor extractor used to create the bag-of-words descriptor.

    Returns:
    np.ndarray: The bag-of-words descriptor for the given local features.
    """
    matches = bf.match(
        desc, vocabulary)  # Find the nearest visual words from the vocabulary for each keypoint descriptor.
    histogram = np.zeros((1, vocabulary.shape[0]), dtype=int)
    for m in matches:
        histogram[0, m.trainIdx] += 1
    histogram = histogram / cv.norm(histogram, normType=cv.NORM_L2)
    histogram = np.array(histogram)
    return histogram


def extract_train_descriptors(train_folders: list, detector) -> np.ndarray:
    """
    Extract descriptors for all images in the given train folders.

    Parameters:
    train_folders (list): A list of paths to the train folders.
    detector (cv2.FeatureDetector): The detector used to detect keypoints in the images.

    Returns:
    np.ndarray: The descriptors for all images in the given train folders.
    """
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


def create_vocabulary(train_descs: np.ndarray) -> np.ndarray:
    """
    Creates a vocabulary of visual words by clustering the local feature descriptors.

    Parameters:
    train_descs (np.ndarray): The local feature descriptors extracted from the training images.

    Returns:
    np.ndarray: The vocabulary of visual words.
    """
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv.BOWKMeansTrainer(70, term_crit, 1, cv.KMEANS_PP_CENTERS)
    vocabulary = trainer.cluster(train_descs.astype(np.float32))
    return vocabulary


def create_index(train_folders: list, vocabulary: np.ndarray, detector, crossCheck: bool = True) -> tuple:
    """
    Creates an index of bag-of-words (BoW) descriptors for the training images.

    Parameters:
    train_folders (list): A list of paths to the folders containing the training images.
    vocabulary (np.ndarray): The vocabulary of visual words.
    detector: The object detector used to extract local feature descriptors from the images.
    crossCheck (bool, optional): Determines if cross-checking is performed in the descriptor matching process. Defaults to True.

    Returns:
    tuple: A tuple containing:
        np.ndarray: The BoW descriptors for the training images.
        list: A list of paths to the training images.
    """
    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))

    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)

            desc = extract_local_features(path, detector)
            bow_desc = BOWImgDescExtractor(vocabulary, desc, cv.BFMatcher.create(
                cv.NORM_L2SQR, crossCheck=crossCheck))

            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)
            img_paths.append(path)
    return bow_descs, img_paths


def Knn(k: int, bow_descs_test: np.ndarray, bow_descs_trained: np.ndarray, train_labels: list) -> list:
    """
    Makes classification predictions for the test images using a k-nearest neighbor (KNN) classifier.

    Parameters:
    k (int): The number of nearest neighbors to consider for the classification decision.
    bow_descs_test (np.ndarray): The BoW descriptors for the test images.
    bow_descs_trained (np.ndarray): The BoW descriptors for the training images.
    train_labels (list): A list of labels corresponding to the training images.

    Returns:
    list: A list of predicted labels for the test images.
"""
    class_predictions = []
    for bow_desc in bow_descs_test:
        distances = np.sum((bow_desc - bow_descs_trained) ** 2, axis=1)
        retrieved_ids = np.argsort(distances)
        retrieved_ids = retrieved_ids[:k]
        neighbors_class = [train_labels[id] for id in retrieved_ids.tolist()]
        prediction = max(set(neighbors_class), key=neighbors_class.count)

        class_predictions.append(prediction)
    return class_predictions


def classification_accuracy(predictions: list, test_labels: list) -> float:
    """
    Calculates the classification accuracy by comparing the predicted labels with the true labels.

    Parameters:
    predictions (list): A list of predicted labels.
    test_labels (list): A list of true labels.

    Returns:
    float: The classification accuracy as a percentage.
    """
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            correct += 1
    accuracy = correct / float(len(test_labels)) * 100.0

    return accuracy


def labels(img_paths: list) -> list:
    """
    Extracts the labels from the image paths.

    Parameters:
    img_paths (list): A list of paths to the images.

    Returns:
    list: A list of labels corresponding to the images.
    """
    labels = [i.split('\\')[2] for i in img_paths]
    return labels
