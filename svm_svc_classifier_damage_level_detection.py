import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dependencies import *

x_s = []
y_s = []

NEW_IMAGE_RESIZE_VALUE = 450
CLASSIFIER_CLASSES = {'minor': 0, 'moderate': 1, 'severe': 2}
IMAGE_SHOW_SERIES = 35
RANDOM_TRAINING_STATE = 10
TEST_SIZE_PERCENTAGE = .2

collect_images_from_files(x_s, y_s, CLASSIFIER_CLASSES, NEW_IMAGE_RESIZE_VALUE)

x_s = np.array(x_s)
y_s = np.array(y_s)

# Sklearn accepts only bidiomensional data, so we have to convert it

xs_reshape = x_s.reshape(len(x_s), -1)
xs_reshape.shape

x_train, x_test, y_train, y_test = train_test_split(
    xs_reshape, 
    y_s, 
    random_state=RANDOM_TRAINING_STATE,
    test_size=TEST_SIZE_PERCENTAGE
)

x_train, x_test = scale_data_tests(x_train, x_test)
classifier = SVC(C=1, kernel='rbf')
classifier.fit(x_train, y_train)
print(classifier.score(x_test, y_test))