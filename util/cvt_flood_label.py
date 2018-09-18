# Yongqing Liang
# root # lyq.me
# Created at: 2018-09-16

import os
import numpy as np
import cv2

flood_root = '/Ship01/Dataset/flood/'
dst_folder = 'realworld/'

dst_path = ''
labels_path = ''
imgs_path = ''

dst_width = 854
dst_height = 480

def get_image_list(images_folder):

    tmp_list = os.listdir(images_folder)

    image_list = []
    for i in range(len(tmp_list)):
        
        image_folder = os.path.join(images_folder, tmp_list[i])
        if os.path.isdir(image_folder):
            image_list.append(tmp_list[i])

    image_list.sort(key=len)

    return image_list


def create_dst_folders():

    global dst_path, labels_path, imgs_path
    dst_path = os.path.join(flood_root, dst_folder)
    labels_path = os.path.join(dst_path, 'labels/')
    imgs_path = os.path.join(dst_path, 'imgs/')

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    if not os.path.exists(labels_path):
        os.mkdir(labels_path)

    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)


def cvt_object_label(img, ori_label_color):

    blue_mask = np.ones([dst_height, dst_width, 1], dtype=np.uint8) * ori_label_color[0]
    green_mask = np.ones([dst_height, dst_width, 1], dtype=np.uint8) * ori_label_color[1]
    red_mask = np.ones([dst_height, dst_width, 1], dtype=np.uint8) * ori_label_color[2]
    color_mask = cv2.merge((blue_mask, green_mask, red_mask))

    # Set label_color to 0
    img = cv2.bitwise_xor(img, color_mask)

    # Set label_color to 255
    img = cv2.bitwise_not(img)

    # Set other pixel to 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)

    return img
    

def cvt_control(video_folder='Canal Rain Event 6.12 1 fps Labeled Images revised/',
                sub_folder='ControlledExperiments/',
                ori_label_color=(0, 255, 255),
                dst_folder=None):

    if dst_folder is None:
        dst_folder = video_folder

    # Get image list
    images_path = os.path.join(flood_root, sub_folder, video_folder)
    image_list = get_image_list(images_path)

    print("Convert", images_path)
    print("to", dst_folder)
    print(len(image_list), "frames.")

    # Create dst folders
    dst_labels_path = os.path.join(labels_path, dst_folder)
    if not os.path.exists(dst_labels_path):
        os.mkdir(dst_labels_path)

    dst_imgs_path = os.path.join(imgs_path, dst_folder)
    if not os.path.exists(dst_imgs_path):
        os.mkdir(dst_imgs_path)
    
    # Loop each image
    for i in range(len(image_list)):

        image_folder_path = os.path.join(images_path, image_list[i])

        # Write image
        ori_image_path = os.path.join(image_folder_path, 'img.png')
        dst_image_path = os.path.join(dst_imgs_path, str(i) + '.png')

        ori_image = cv2.imread(ori_image_path)
        dst_image = cv2.resize(ori_image, (dst_width, dst_height))
        cv2.imwrite(dst_image_path, dst_image)

        # Write label
        ori_label_path = os.path.join(image_folder_path, 'label_modified_vis.png')
        dst_label_path = os.path.join(dst_labels_path, str(i) + '.png')

        ori_label = cv2.imread(ori_label_path)
        dst_label = cv2.resize(ori_label, (dst_width, dst_height))
        dst_label = cvt_object_label(dst_label, ori_label_color)
        cv2.imwrite(dst_label_path, dst_label)


if __name__ == '__main__':

    create_dst_folders()

    cvt_control(video_folder='buffalo_integrated_dataset_v3_label_unified/',
                sub_folder='FloodMeasurement/',
                dst_folder='buffalo0/',
                ori_label_color=(0, 0, 200))