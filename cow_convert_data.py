import os
import shutil
import xml.etree.ElementTree as ET

import yaml
from tqdm import tqdm
from utils.general import download, Path

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path = "./datasets/cow"  # dataset root dir
train = "train/" # train images (relative to 'path') 128 images
val = "test/"  # val images (relative to 'path') 128 images

# Classes
nc = 1  # number of classes
names = ['cow']  # class names

def convert_label(path, lb_path, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    #print(lb_path)
    in_file = open(path + f'/train/{image_id}.xml')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        #print(cls)
        if cls in names and not int(obj.find('difficult').text) == 1:
            #print('yay')
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            #print(bb)
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


imgs_path = path + '/' + train + 'images'
lbs_path = path + '/' + train + 'labels'
try:
    os.mkdir(imgs_path)
except OSError as error:
    pass

try:
    os.mkdir(lbs_path)
except OSError as error:
    pass

lbs_path = path + '/' + train + 'labels'
image_ids = open(path + '/cow_images_train.txt').read().strip().splitlines()
for id in tqdm(image_ids, desc=f'cow'):
    #print(id.split())
    id = id.split()[0]
    l_num = len(id)-4
    lbl = str(id)[0:l_num]
    lb_path = lbs_path + '/' + lbl + '.txt' # new label path
    shutil.move(path + '/' + train + str(id), imgs_path + '/' + str(id))
    #print(path + '/' + train + str(id))
    #print(lbs_path + '/' + str(id))
    convert_label(path, lb_path, lbl)  # convert labels to YOLO format