import cv2
import os
import numpy as np


def high_pass_filter(img):
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    kernel_size = 3
    img_blurred = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    max_intensity_value = 255
    img_stroke = (img_gray.astype(int) - img_blurred.astype(int)) + max_intensity_value
    return img_stroke


def canny_edge_detector(img, img_to_grayscale, img_to_blur, blur_size, lower_threshold, upper_threshold):
    if img_to_blur:
        img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    if img_to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sketch = cv2.Canny(img, lower_threshold, upper_threshold)
    sketch = cv2.bitwise_not(sketch)
    return sketch


def loop_canny_edge_detector(data_path):
    img_list = os.listdir(data_path)
    img_count = 0
    for img_file in img_list:
        file_path = '{}/{}'.format(data_path, img_file)
        try:
            img = cv2.imread(file_path)
        except:
            print('problematic file:', file_path)
            continue
        if img is None:
            print('problematic file:', file_path)
            continue
        else:
            sketch = canny_edge_detector(img, 1, 1, 3, 50, 250)
            cv2.imwrite(file_path, sketch)
        img_count += 1
        if img_count % 100 == 0:
            print('{} images processed!'.format(img_count), end='\r')

    print('finished processing {} images !'.format(img_count))
