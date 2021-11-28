import multiprocessing
import os
import random
import sys
import time
from multiprocessing import Process
from pathlib import Path

import cv2
import pyautogui
import pytesseract
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageGrab

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device

# set start time to current time
start_time = time.time()

shoot_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Set monitor size to capture
monitor = (40,  0, 800, 800)

width = 1920  # 800
height = 1080  # 640

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


class YOLO():

    def __init__(
            self,
            weights: str,  # model.pt path(s)
            source: str,  # file/dir/URL/glob, 0 for webcam
            imgsz: int or str,  # inference size (pixels)
            conf_thres: int,  # confidence threshold
            iou_thres: int,  # NMS IOU threshold
            max_det: int,  # maximum detections per image
            device: str,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img: bool,  # show results
            classes: None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms: bool,  # class-agnostic NMS
            augment: bool,  # augmented inference
            visualize: bool,  # visualize features
            line_thickness: int,  # bounding box thickness (pixels)
            hide_labels: bool,  # hide labels
            hide_conf: bool,  # hide confidences
            half: bool,  # use FP16 half-precision inference
    ):
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        self.names, self.model, self.stride, self.imgsz = self.generate(self.weights, self.device, self.imgsz)

    def generate(self, weights, device, imgsz):
        #print(weights)
        #print(device)
        #print(imgsz)
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # print(imgsz)

        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        cudnn.benchmark = True  # set True to speed up constant image size inference
        return names, model, stride, imgsz

    def detect_image(self, source, names, model, imgsz, stride):
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

        # Run inference
        # model(torch.zeros(1, 3, *[1280,1280]).to(device).type_as(next(model.parameters())))

        #t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Inference
            #t1 = time_sync()
            pred = model(img, augment=self.augment, visualize=self.visualize)[0]

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            #t2 = time_sync()

            # Process predictions
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                s += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                           line_width=self.line_thickness)
                    return im0, xyxy, label, conf
                # Print time (inference + NMS)
                #print(f'{s}Done. ({t2 - t1:.3f}s)')



def image_to_text(preprocess, image):
    global text
    # construct the argument parse and parse the arguments
    image = cv2.imread(image)
    image = cv2.bitwise_not(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    if preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    if preprocess == 'adaptive':
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    f = open("action.txt", "w")
    f.write(text)
    f.close()
    # print(text)
    # show the output images
    # cv2.imshow("Image", image)
    # cv2.imshow("Output", gray)
    # cv2.waitKey(0)


def resizeImage():
    png = GRABMSS_screen_text()
    im = Image.open(png)  # uses PIL library to open image in memory
    left = 10
    top = 51
    right = 140
    bottom = 69
    im = im.crop((left, top, right, bottom))  # defines crop points
    im.save('textshot.png')  # saves new cropped image
    width, height = im.size
    new_size = (width * 4, height * 4)
    im1 = im.resize(new_size)
    im1.save('textshot.png')


def click_attack(x, y):
    b = random.uniform(0.05, 0.09)
    x = random.uniform(x - 2.5, x + 2.5)
    y = random.uniform(y - 2.5, y + 2.5)
    c = random.uniform(0.03, 0.06)
    pyautogui.moveTo(x, y, duration=b)
    time.sleep(c)
    pyautogui.click(x, y, 1, button='left')

def GRABMSS_screen():
    im = ImageGrab.grab(bbox=monitor) # left , top , right, bottom
    im.save('fullscreen.png', 'png')
    return 'fullscreen.png'

def GRABMSS_screen_text():
    im = ImageGrab.grab(bbox=monitor) # left , top , right, bottom
    im.save('textscreen.png', 'png')
    return 'textscreen.png'

def READ_Text():
    while True:
        global fished
        # Grab screen image
        resizeImage()
        image_to_text('thresh', 'textshot.png')
        # Put image from pipe


def SHOWMSS_screen():
    global fps, start_time, coords, classes, percent
    im0 = None
    xyxy = None
    label = None
    conf = None
    shoot_time = time.time()
    yolo = YOLO(weights,
                source,
                imgsz,
                conf_thres,
                iou_thres,
                max_det,
                device,
                view_img,
                classes,
                agnostic_nms,
                augment,
                visualize,
                line_thickness,
                hide_labels,
                hide_conf,
                half)
    while True:
        img = GRABMSS_screen()
        if yolo.detect_image(img, yolo.names, yolo.model, yolo.imgsz, yolo.stride) == None:
            print('nothing')
            img = cv2.imread(img)
            cv2.imshow("YOLO v5", img)
        else:
            im0, xyxy, label, conf = yolo.detect_image(img, yolo.names, yolo.model, yolo.imgsz, yolo.stride)
            f = open("action.txt", "r")
            # print(f.readline().strip())
            object = str(f.readline().strip())
            #print(object)
            #print('im0:', im0)
            print('label:', label)
            #print('conf:', conf)
            cv2.imshow("YOLO v5", im0)
            attack = time.time() - shoot_time
            c = random.uniform(5, 8)
            if float(conf) > 0.8 and attack > c:
                print(object)
                if object != "Cow" and object != "Cou" and object != "Cow calf":
                    mid_x = (xyxy[0] + xyxy[2])/2 + 40
                    mid_y = (xyxy[1] + xyxy[3])/2
                    pyautogui.moveTo(mid_x,mid_y)
                    click_attack(mid_x, mid_y)
                    shoot_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
        fps += 1
        TIME = time.time() - start_time
        if (TIME) >= display_time:
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'): break


weights = 'best.pt'  # model.pt path(s)
source = 'data/images'  # file/dir/URL/glob, 0 for webcam
imgsz = [640, 640]  # inference size (pixels)
conf_thres = 0.7  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
max_det = 10  # maximum detections per image
device = '0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img = True  # show results
classes = None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False  # class-agnostic NMS
augment = False  # augmented inference
visualize = False  # visualize features
line_thickness = 1  # bounding box thickness (pixels)
hide_labels = False  # hide labels
hide_conf = False  # hide confidences
half = False  # use FP16 half-precision inference

if __name__ == "__main__":
    x = random.randrange(625, 635)
    y = random.randrange(345, 355)
    pyautogui.click(x, y, button='right')
    j = 0
    # creating new processes
    p2 = multiprocessing.Process(target=SHOWMSS_screen)
    p3 = Process(target=READ_Text)

    # starting our processes
    p2.start()
    p3.start()
    p3.join()
