import os
import string
import random
import argparse
from ..util.util import string_generator

parser = argparse.ArgumentParser('rename image')
parser.add_argument('--dir_path', dest='dir_root', help='input directory for image ', type=str, default='')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

dir_root = args.dir_path

for dir_name in os.listdir(dir_root):
    dir_path = os.path.join(dir_root, dir_name)
    for file_name in os.listdir(dir_path):
        img_extension = file_name.split('.')[-1]
        oldpath = os.path.join(dir_path, file_name)
        new_name = string_generator(length=16)
        new_path = os.path.join(dir_path, new_name + '.' + img_extension)
        os.rename(oldpath, new_path)


def rename_file_pairs(dir_path_a, dir_path_b):
    for file_name in os.listdir(dir_path_a):
        img_extension = file_name.split('.')[-1]
        new_name = string_generator(length=16)

        oldpath_a = os.path.join(dir_path_a, file_name)
        new_path_a = os.path.join(dir_path_a, new_name + '.' + img_extension)
        os.rename(oldpath_a, new_path_a)

        oldpath_b = os.path.join(dir_path_b, file_name)
        new_path_b = os.path.join(dir_path_b, new_name + '.' + img_extension)
        os.rename(oldpath_b, new_path_b)
