import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dependencies import *


Xs = []
Ys = []

NEW_IMAGE_RESIZE_VALUE = 450
CLASSIFIER_CLASSES = {'minor': 0, 'moderate': 1, 'severe': 2}
IMAGE_SHOW_SERIES = 35
RANDOM_TRAINING_STATE = 10
TEST_SIZE_PERCENTAGE = .2

collect_images_from_files(Xs, Ys, CLASSIFIER_CLASSES, NEW_IMAGE_RESIZE_VALUE)

Xs = np.array(Xs)
Ys = np.array(Ys)
Xs_reshape = Xs.reshape(len(Xs), -1)

xTrain, xTest, yTrain, yTest = train_test_split(
    Xs_reshape, 
    Ys, 
    random_state=RANDOM_TRAINING_STATE,
    test_size=TEST_SIZE_PERCENTAGE
)

xTrain, xTest = scale_data_tests(xTrain, xTest)
classifier = SVC(C=1, kernel='rbf')
classifier.fit(xTrain, yTrain)

classifier.score(xTest, yTest)

run_live_images_testing(classifier, IMAGE_SHOW_SERIES, NEW_IMAGE_RESIZE_VALUE)
