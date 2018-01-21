import cv2
import os
from util.util import mkdir
from .stroke_extraction_filter import high_pass_filter
import argparse
from util.util import mkdir
from stroke_extraction_filter import high_pass_filter, canny_edge_detector

parser = argparse.ArgumentParser('stroke_extraction')
parser.add_argument('--data_path', dest='data_path', help='input image directory ', type=str, default='')
parser.add_argument('--new_path', dest='new_path', help='new directory for image sketch', type=str, default='')
parser.add_argument('--new_img_size', dest='new_img_size', help='new szie for  image sketch', type=int, default=286)

args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))
data_path = args.data_path


def stroke_extraction_with_loop(data_path):
    img_list = os.listdir(data_path)
    mkdir(args.new_path)
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
            img_stroke = high_pass_filter(img)
            old_file_location = 'full'
            new_file_location = 'sketch'
            new_path = file_path.replace(old_file_location, new_file_location)
            cv2.imwrite(new_path, img_stroke)
        img_count += 1
        if img_count % 1000 == 0:
            print('{} images processed!'.format(img_count), end='\r')

            print('finished processing {} images !'.format(img_count))

stroke_extraction_with_loop(data_path)

img_list = os.listdir(args.data_path)

mkdir(args.new_path)
name_list = os.listdir(args.new_path)

img_count = 0
for img_name in img_list:
    if img_name in name_list:
        pass
    else:
        img_path = os.path.join(args.data_path, img_name)
        try:
            img = cv2.imread(img_path)
        except:
            print('problematic file:', img_path)
            continue
        if img is None:
            print('problematic file:', img_path)
            continue
        else:
            img_stroke = high_pass_filter(img)
            # img_stroke = canny_edge_detector(img, 1, 1, 7, 20, 60)
            # img_stroke = cv2.resize(img_stroke, (args.new_img_size, args.new_img_size), interpolation=cv2.INTER_AREA)
            new_path = os.path.join(args.new_path, img_name)
            cv2.imwrite(new_path, img_stroke)
        img_count += 1
        if img_count % 1000 == 0:
            print('{} images processed!'.format(img_count), end='\r')

print('finished processing {} images !'.format(img_count))
