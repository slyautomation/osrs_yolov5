import shutil
import xml.etree.ElementTree as ET
import os
import pathlib




def convert_label(path, lb_path, image_id):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    #print(lb_path)
    #print(path)
    in_file = open(path + f'{image_id}')
    out_file = open(lb_path, 'w')
    try:
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
    except:
        out_file.close()
        print('no objects annotted!')
        os.remove(lb_path)
        replace = '\\labels'
        lb_path = lb_path.replace(replace, '')
        print(lb_path)
        end = len(lb_path)
        jpg_path = lb_path[:end-3] + "jpg"
        try:
            os.remove(jpg_path)
        except:
            print('file not found, carry on')



# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path = "\\datasets\\osrs\\"  # dataset root dir

directory = pathlib.Path(__file__).parent.resolve()
print(str(directory) + path)
# Classes
nc = 1  # number of classes
names = ['Al Kharid warrior']  # class names


imgs_path = str(directory) + path + 'images'
lbs_path = str(directory) + path + 'labels'

print('img_path:',imgs_path)
print('lbs_path:',lbs_path)
try:
    os.mkdir(imgs_path)
except OSError as error:
    pass

try:
    os.mkdir(lbs_path)
except OSError as error:
    pass

lbs_path = path + 'labels\\'

files = os.listdir(str(directory) + path)
#print(str(files))
os.chdir(str(directory) + path)
for filename in os.listdir(str(directory) + path):
    if filename.endswith('xml'):
        print(filename)
        lfile = len(filename) - 4
        lbl = filename
        #print(lbl)
        lb_path = lbs_path + filename[0: lfile] + '.txt'  # new label path
        convert_label(str(directory) + path, str(directory) + lb_path, lbl)  # convert labels to YOLO format
    if filename.endswith('jpg'):
        shutil.move(str(directory) + path + '/' + filename, imgs_path + '/' + filename)

