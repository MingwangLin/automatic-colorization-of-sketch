import os
import time
from data.data_loader import CreateDataLoader
from data.future_vision import to_pil_image
from util.util import mkdir
from stroke_extraction.stroke_extraction_filter import single_img_to_sketch_with_hed
from options.train_options import TrainOptions
from PIL import Image

opt = TrainOptions().parse()
dataloader = CreateDataLoader(opt).load_data()
dataset_size = len(dataloader)


def resize_and_extract_sketch_with_multiprocess():
    img_count = 0
    time_start = time.time()
    # data_path = '/home/lin/Pictures/ndsketch286/'
    # name_list = os.listdir(data_path)
    # path_list = []
    # for img_name in name_list:
    #     raw_path = '/home/lin/Pictures/nd/nd'
    #     img_path = os.path.join(raw_path, img_name)
    #     path_list += [img_path]
    for i, data in enumerate(dataloader):
        # print('data', data)
        data_tensor = data['A']
        img_num = data_tensor.size()[0]
        for j in range(img_num):
            pil_img = to_pil_image(data_tensor[j, ...])
            path = data['A_paths'][j]
            # print(path)
            # print(path_list[:5])
            # if path in path_list:
            #     # print('----------------------------------------')
            #     pass
            # else:

            # delete middle dir name
            index_end = path.rfind('/')
            index_start = path.rfind('/', 0, index_end)
            path = path[:index_start] + path[index_end:]
            #
            folder_name_index_end = path.rfind('/')
            folder_name_index_start = path.rfind('/', 0, folder_name_index_end)
            old_folder_name = path[folder_name_index_start + 1:folder_name_index_end]

            # save 515*512 image
            small_img_folder_name = old_folder_name + '512'
            small_img_path = path.replace(old_folder_name, small_img_folder_name)
            pil_img.save(small_img_path, quality=95)

            # save 256*256 image
            smaller_img_folder_name = old_folder_name + '286'
            smaller_img_path = path.replace(old_folder_name, smaller_img_folder_name)
            smaller_img_size = 286
            pil_img_smaller = pil_img.resize((smaller_img_size, smaller_img_size), Image.ANTIALIAS)
            pil_img_smaller.save(smaller_img_path, quality=95)

            # extract sketch and save
            small_sketch_folder_name = old_folder_name + 'sketch512'
            small_sketch_path = path.replace(old_folder_name, small_sketch_folder_name)

            smaller_sketch_folder_name = old_folder_name + 'sketch286'
            smaller_sketch_path = path.replace(old_folder_name, smaller_sketch_folder_name)

            single_img_to_sketch_with_hed(raw_path=small_img_path, new_img_size=286,
                                          new_path=(small_sketch_path, smaller_sketch_path))
            img_count += 1
            if img_count % 10000 == 0:
                time_end = time.time()
                print('{} images processed! time cost{}'.format(img_count, time_end - time_start))
                time_start = time_end
    return




resize_and_extract_sketch_with_multiprocess()

def img_to_sketch_with_hed(data_path):
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


# img_to_sketch_with_hed(data_path='/home/lin/Downloads/boorucp/full')


def resize_img(data_path, img_size_resized):
    img_list = os.listdir(data_path)
    img_count = 0
    print('img_list', len(img_list))
    time_start = time.time()
    for img_file in img_list:
        file_path = '{}/{}'.format(data_path, img_file)
        try:
            img = cv2.imread(file_path)
            h, w, c = img.shape
        except:
            print('problematic file:', file_path)
            continue
        if h == img_size_resized and w == img_size_resized:
            continue
        else:
            img = cv2.resize(img, (img_size_resized, img_size_resized), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_path, img)
        img_count += 1
        if img_count % 10000 == 0:
            time_end = time.time()
            print('{} images processed! time cost{}'.format(img_count, time_end - time_start))
            time_start = time_end
    print('finished processing {} images !'.format(img_count))


def scale_img(data_path, scale_size):
    img_list = os.listdir(data_path)
    img_count = 0
    print('img_list', len(img_list))
    for img_file in img_list:
        file_path = '{}/{}'.format(data_path, img_file)
        try:
            img = cv2.imread(file_path)
            width = float(img.shape[1])
            height = float(img.shape[0])
        except:
            print('problematic file:', file_path)
            continue
        if img is None:
            print('problematic file:', file_path)
            continue
        else:
            if width > height:
                img = cv2.resize(img, (scale_size, int(scale_size / width * height)), interpolation=cv2.INTER_AREA)
            else:
                img = cv2.resize(img, (int(scale_size / height * width), scale_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file_path, img)
        img_count += 1
        if img_count % 100 == 0:
            print('{} images processed!'.format(img_count), end='\r')
    print('finished processing {} images !'.format(img_count))


def batch_img_to_sketch_with_hed(data_path):
    img_list = os.listdir(data_path)
    img_count = 0
    time_start = time.time()
    list_len = len(img_list)
    batch_size = opt.batchSize
    batch_num = int(list_len / batch_size)
    batch_count = 0
    while batch_count < batch_num - 1:
        iter_count = 0
        log(1)
        for img_file in img_list[batch_count * batch_size:(batch_count + 1) * batch_size]:
            file_path = '{}/{}'.format(data_path, img_file)
            try:
                img = cv2.imread(file_path)
                h, w, c = img.shape
            except:
                print('problematic file:', file_path)
                continue
            img = img.transpose((2, 0, 1))
            light_map = np.zeros(img.shape, dtype=np.float)
            for channel in range(3):
                light_map[channel] = get_light_map_single(img[channel])
            light_map = normalize_pic(light_map)
            light_map = light_map[None]
            light_map = light_map.transpose((1, 2, 3, 0))
            if iter_count == 0:
                batch_light_map = light_map
            else:
                batch_light_map = np.concatenate((batch_light_map, light_map), axis=3)
            iter_count += 1
            # print (light_map.shape)
        line_mat = mod.predict(batch_light_map, batch_size=batch_light_map.shape[-1])
        for i in line_mat:
            line_mat = line_mat.transpose((3, 1, 2, 0))[i]
            line_mat = np.amax(line_mat, 2)
            file_path = data_path[:18] + '/nicodata_copy/' + img_list[batch_count * batch_size + i]
            # print(file_path)
            adjust_and_save_img(line_mat, file_path)
            img_count += 1
            if img_count % 100 == 0:
                time_end = time.time()
                print('{} images processed! time cost{}'.format(img_count, time_end - time_start))
                time_start = time_end
        batch_count += 1
        log(2)
    print('finished processing {} images !'.format(img_count))
