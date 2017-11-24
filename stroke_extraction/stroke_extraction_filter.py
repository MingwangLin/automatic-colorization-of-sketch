import cv2
import os
import numpy as np
from util.helper import *
from keras.models import load_model

mod = load_model('data/mod.h5')


def img_to_sketch_with_hed(raw_path, new_img_size, new_path):
    img = cv2.imread(raw_path)
    img = img.transpose((2, 0, 1))
    light_map = np.zeros(img.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(img[channel])
    light_map = normalize_pic(light_map)
    light_map = light_map[None]
    light_map = light_map.transpose((1, 2, 3, 0))
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = np.amax(line_mat, 2)
    adjust_and_save_img(line_mat, new_img_size, path=new_path)
    return

def img_to_sketch_with_hed_with_loop(data_path):
    img_list = os.listdir(data_path)
    img_count = 0
    time_start = time.time()
    for img_file in img_list[:100]:
        file_path = '{}/{}'.format(data_path, img_file)
        try:
            img = cv2.imread(file_path)
            # resize img
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        except:
            print('problematic file:', file_path)
            continue
        if img is None:
            print('problematic file:', file_path)
            continue
        else:
            img = img.transpose((2, 0, 1))
            light_map = np.zeros(img.shape, dtype=np.float)
            for channel in range(3):
                light_map[channel] = get_light_map_single(img[channel])
            light_map = normalize_pic(light_map)
            light_map = light_map[None]
            light_map = light_map.transpose((1, 2, 3, 0))
            line_mat = mod.predict(light_map, batch_size=1)
            line_mat = line_mat.transpose((3, 1, 2, 0))[0]
            line_mat = np.amax(line_mat, 2)
            old_file_location = 'full'
            new_file_location = 'hed'
            new_path = file_path.replace(old_file_location, new_file_location)
            path = new_path, new_path
            print('newpath', new_path)
            adjust_and_save_img(line_mat, 512, path)
            img_count += 1
        if img_count % 1000 == 0:
            time_end = time.time()
            print('{} images processed! time cost{}'.format(img_count, time_end - time_start))
            time_start = time_end
    print('finished processing {} images !'.format(img_count))
    return

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

