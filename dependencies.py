import matplotlib.pyplot as plt
import cv2
import os

STANDARD_DATA_DIRECTORY_STRUCTURE = 'dataset/validation/'
STANDARD_RESIZE_VALUE = 300
classifier_classes = {'minor': 0, 'moderate': 1, 'severe': 2}


def collect_images_from_files(X, Y, classifier_classes, resize_value=STANDARD_RESIZE_VALUE):
    for cls_class in classifier_classes:
        path = STANDARD_DATA_DIRECTORY_STRUCTURE + cls_class
        for file in os.listdir(path):
            img = cv2.imread(path + '/' + file, 0)
            img = cv2.resize(img, (resize_value, resize_value))
            X.append(img)
            Y.append(classifier_classes[cls_class])


def test_based_on_images(classifier, file_testing_class, size=9, resize_value=STANDARD_RESIZE_VALUE):
    values_classifier = {0: 'minor', 1: 'moderate', 2: 'severe'}

    plt.figure(figsize=(12, 8))
    subplot_index = 1

    for i in os.listdir(STANDARD_DATA_DIRECTORY_STRUCTURE + file_testing_class)[:size]:
        plt.subplot(3, 3, subplot_index)

        img = cv2.imread(STANDARD_DATA_DIRECTORY_STRUCTURE + file_testing_class + i, 0)
        img_resized = cv2.resize(img, (resize_value, resize_value))
        img_resized = img_resized.reshape(1, -1) / 255
        path = classifier.predict(img_resized)
        plt.title(values_classifier[path[0]])
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        if subplot_index == 9:
            subplot_index = 1
        else:
            subplot_index += 1


def scale_data_tests(x_train, x_test):
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, x_test


def run_live_images_testing(classifier, image_show_series, new_image_resize_value):
    test_based_on_images(classifier, 'minor/', image_show_series, new_image_resize_value)
    test_based_on_images(classifier, 'moderate/', image_show_series, new_image_resize_value)
    test_based_on_images(classifier, 'severe/', image_show_series, new_image_resize_value)
