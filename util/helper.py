import cv2
import time
import string
import random
from scipy import ndimage
import numpy as np
from datetime import datetime


def get_normal_map(img):
    img = img.astype(np.float)
    img = img / 255.0
    img = - img + 1
    img[img < 0] = 0
    img[img > 1] = 1
    return img


def get_gray_map(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    highPass = gray.astype(np.float)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    highPass = highPass[None]
    return highPass.transpose((1, 2, 0))


def get_light_map(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    # highPass = highPass / 10000.0
    highPass = highPass[None]
    return highPass.transpose((1, 2, 0))


def get_light_map_single(img):
    gray = img
    gray = gray[None]
    gray = gray.transpose((1, 2, 0))
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    # print('blur', blur.shape)
    gray = gray.reshape((gray.shape[0], gray.shape[1]))
    highPass = gray.astype(int) - blur.astype(int)
    highPass = highPass.astype(np.float)
    highPass = highPass / 64.0
    # print('highPass', highPass.shape, highPass)
    return highPass


def normalize_pic(img):
    if np.max(img) != 0:
        img = img / (np.max(img))
        img = img
    return img


def adjust_and_save_img(img, new_img_size, path):
    mat = img.astype(np.float)
    threshold = 0.0
    mat[mat < threshold] = 0
    mat = - mat + 1
    mat = (mat * 255.0)
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imwrite(path[0], mat)
    img = cv2.resize(mat, (new_img_size, new_img_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path[1], img)
    return


def get_light_map_drawer(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    highPass = highPass[None]
    return highPass.transpose((1, 2, 0))


def get_light_map_drawer2(img):
    ret = img.copy()
    ret = ret.astype(np.float)
    ret[:, :, 0] = get_light_map_drawer3(img[:, :, 0])
    ret[:, :, 1] = get_light_map_drawer3(img[:, :, 1])
    ret[:, :, 2] = get_light_map_drawer3(img[:, :, 2])
    ret = np.amax(ret, 2)
    return ret


def get_light_map_drawer3(img):
    gray = img
    blur = cv2.blur(gray, ksize=(5, 5))
    highPass = gray.astype(int) - blur.astype(int) + 255
    highPass[highPass < 0] = 0
    highPass[highPass > 255] = 255
    highPass = highPass.astype(np.float)
    highPass = highPass / 255.0
    highPass = 1 - highPass
    # highPass = highPass.astype(np.uint8)
    return highPass


def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)


def to_pencil_sketch(img):
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    print('gray', img_gray)
    print('blur', img_blur)
    high_pass = dodgeV2(img_gray, img_blur)

    print('highpass', high_pass.shape, high_pass[125:150])

    return high_pass


def high_pass(img):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    print('gray', gray)
    print('blur', blur)
    highPass = (gray.astype(int) - blur.astype(int)) + 255

    print('highpass', highPass.shape, highPass)
    # highPass = 255 - blur.astype(int)
    # highPass[highPass < 0] = 0
    # highPass[highPass > 255] = 255

    # #
    # # highPass = highPass.astype(np.float)
    # highPass = highPass / 255.0
    # highPass = (1 - highPass)*255
    # highPass = highPass.astype(np.uint8)
    # highPass = cv2.bitwise_not(highPass)
    print('highpass', highPass.shape, highPass)

    return highPass


def high_pass_sketchkeras(img):
    mat_color = get_light_map(img)
    print('mat_color_divide', mat_color.shape, mat_color)
    mat_color = normalize_pic(mat_color)
    print('mat_color_norm', mat_color.shape, mat_color)
    # mat_color = resize_img_512(mat_color)
    mat = mat_color.astype(np.float)
    print('mat_color_float', mat.shape, mat)
    # threshold = 0.1
    # mat[mat < threshold] = 0
    mat = (1 + mat / 128) * 255.0
    print('mat_color_multi', mat.shape, mat)
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    print('mat_color', mat_color.shape)
    return mat


def superlize_pic(img):
    img = img * 2.33333
    img[img > 1] = 1
    return img


def mask_pic(img, mask):
    mask_mat = mask
    mask_mat = mask_mat.astype(np.float)
    mask_mat = cv2.GaussianBlur(mask_mat, (0, 0), 1)
    mask_mat = mask_mat / np.max(mask_mat)
    mask_mat = mask_mat * 255
    mask_mat[mask_mat < 255] = 0
    mask_mat = mask_mat.astype(np.uint8)
    mask_mat = cv2.GaussianBlur(mask_mat, (0, 0), 3)
    mask_mat = get_gray_map(mask_mat)
    mask_mat = normalize_pic(mask_mat)
    mask_mat = resize_img_512(mask_mat)
    super_from = np.multiply(img, mask_mat)
    return super_from


def resize_img_512(img):
    zeros = np.zeros((512, 512, img.shape[2]), dtype=np.float)
    zeros[:img.shape[0], :img.shape[1]] = img
    return zeros


def resize_img_512_3d(img):
    zeros = np.zeros((1, 3, 512, 512), dtype=np.float)
    zeros[0, 0: img.shape[0], 0: img.shape[1], 0: img.shape[2]] = img
    return zeros.transpose((1, 2, 3, 0))


def broadcast_img_to_3d(img):
    zeros = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float)
    zeros[0, :, :, :] = img
    return zeros.transpose((1, 2, 3, 0))


def show_active_img_and_save(name, img, path):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def denoise_mat(img, i):
    return ndimage.median_filter(img, i)


def show_active_img_and_save_denoise(name, img, path):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def show_active_img_and_save_denoise_filter(name, img, path):
    mat = img.astype(np.float)
    threshold = 0.18
    mat[mat < threshold] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def show_active_img_and_save_denoise_filter2(name, img, path):
    mat = img.astype(np.float)
    threshold = 0.1
    mat[mat < threshold] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    cv2.imshow(name, mat)
    cv2.imwrite(path, mat)
    return


def show_active_img(name, img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    cv2.imshow(name, mat)
    return


def get_active_img(img):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


def get_active_img_fil(img):
    mat = img.astype(np.float)
    mat[mat < 0.18] = 0
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    return mat


def show_double_active_img(name, img):
    mat = img.astype(np.float)
    mat = mat * 128.0
    mat = mat + 127.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    cv2.imshow(name, mat.astype(np.uint8))
    return


def debug_pic_helper():
    for index in range(1130):
        gray_path = 'data\\gray\\' + str(index) + '.jpg'
        color_path = 'data\\color\\' + str(index) + '.jpg'

        mat_color = cv2.imread(color_path)
        mat_color = get_light_map(mat_color)
        mat_color = normalize_pic(mat_color)
        mat_color = resize_img_512(mat_color)
        show_double_active_img('mat_color', mat_color)

        mat_gray = cv2.imread(gray_path)
        mat_gray = get_gray_map(mat_gray)
        mat_gray = normalize_pic(mat_gray)
        mat_gray = resize_img_512(mat_gray)
        show_active_img('mat_gray', mat_gray)

        cv2.waitKey(1000)


def log(*args):
    # t = time.time()
    # tt = time.strftime(r'%Y/%m/%d %H:%M:%S', time.localtime(t))
    # current_milli_time = t * 1000
    tt = datetime.now().strftime("%H:%M:%S.%f")
    print(tt, *args)
    return
    # 2016/6/22 21:40:10.000


def string_generator(length):
    chars = string.ascii_lowercase + string.digits
    # chars = string.digits
    return ''.join(random.SystemRandom().choice(chars) for _ in range(length))
