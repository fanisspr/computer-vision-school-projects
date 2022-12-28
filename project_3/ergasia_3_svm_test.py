import os
import cv2 as cv
import numpy as np
import json
from functions import *


# train_folders = ['imagedb_train/069.fighter-jet', 'imagedb_train/145.motorbikes-101', 'imagedb_train/178.school-bus', 'imagedb_train/224.touring-bike', 'imagedb_train/251.airplanes-101', 'imagedb_train/252.car-side-101']
# test_folders = ['imagedb_test/069.fighter-jet', 'imagedb_test/145.motorbikes-101', 'imagedb_test/178.school-bus', 'imagedb_test/224.touring-bike', 'imagedb_test/251.airplanes-101', 'imagedb_test/252.car-side-101']

root_dir = os.path.relpath(os.path.dirname(__file__))
train_dir = os.path.join(root_dir, 'imagedb_train')
test_dir = os.path.join(root_dir, 'imagedb_test')

train_folders = [os.path.join(train_dir, subdir)
                 for subdir in os.listdir(train_dir)]
test_folders = [os.path.join(test_dir, subdir)
                for subdir in os.listdir(test_dir)]

sift = cv.SIFT_create()

# Load vocabulary
try:
    vocabulary = np.load(os.path.join(root_dir, 'vocabulary.npy'))
    print('Vocabulary is loaded')
except:
    FileNotFoundError

# create test Index or load if exists
index_test_filename = os.path.join(root_dir, 'index_test.npy')
index_paths_test_filename = os.path.join(root_dir, 'index_paths_test.txt')
if not os.path.exists(index_test_filename):
    print('Creating test index...')
    bow_descs_test, img_paths_test = create_index(
        test_folders, vocabulary, sift, crossCheck=False)
    np.save(index_test_filename, bow_descs_test)
    with open(index_paths_test_filename, mode='w+') as file:
        json.dump(img_paths_test, file)
    print('Test Index created')
else:
    bow_descs_test = np.load(index_test_filename)
    with open(index_paths_test_filename, mode='r') as file:
        img_paths_test = json.load(file)
    print('Test Index is loaded')

# make test_labels
test_labels = labels(img_paths_test)

classes = os.listdir(train_dir)

svm = cv.ml.SVM_create()
class_predictions = []
for i, bow_desc in enumerate(bow_descs_test):
    responses = []
    for cls in classes:
        svm_cls = svm.load(os.path.join(root_dir, 'svm_') + cls)
        response = svm_cls.predict(np.reshape(bow_desc.astype(np.float32), (1, -1)),
                                   flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses.append(min(response[1]))
    min_arg = np.argmin(responses)

    class_predictions.append(classes[min_arg])
    print(img_paths_test[i], end=' -> ')
    if min_arg == 0:
        print('It is a fighter jet')
    elif min_arg == 1:
        print('It is a motorbike')
    elif min_arg == 2:
        print('It is a school bus')
    elif min_arg == 3:
        print('It is a touring bike')
    elif min_arg == 4:
        print('It is an airplane')
    elif min_arg == 5:
        print('It is a car-side')

Acc = classification_accuracy(class_predictions, test_labels)
print("Classification accuracy: ", Acc)

print('Svm classification finished')
