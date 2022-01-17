

# Yolov5 Object Detection In OSRS using Python code, Detecting Cows - Botting

name: GeForce GTX 1060 6GB (average fps 11 on monitor display using screenshots)

https://user-images.githubusercontent.com/81003470/148143834-97b237c1-205c-4e95-b2c7-fb16e9938184.mp4

For a video with commentary: https://youtu.be/rqk0kq4Vu3M

Full Video Tutorial: TBA

# Quick Start

### installing pycharm

pycharm = https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC

Step walkthrough Installation of pycharm: https://github.com/slyautomation/osrs_yolov5/wiki/How-to-Install-Pycharm

### Check your cuda version

type in terminal: ```nvidia-smi```

![image](https://user-images.githubusercontent.com/81003470/147712277-5b1fae1d-33b2-4ff0-a4de-19ef762e1b14.png)

my version that i can use is up to: 11.5 but for simplicity i can use previous versions namely 10.0 and 10.2

Check if your gpu will work: https://developer.nvidia.com/cuda-gpus and use the cuda for your model and the latest cudnn for the cuda version.

full list of cuda versions: https://developer.nvidia.com/cuda-toolkit-archive

cuda 10.0 = https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10

cuda 10.2 = https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe

### Install Cudnn

cuDNN = https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-10

Step walkthrough Installation of cuDNN: https://github.com/slyautomation/osrs_yolov5/wiki/How-to-install-CuDNN

### Download and Open LabelImg

labelImg = https://tzutalin.github.io/labelImg/

Step walkthrough download and using LabelImg: https://github.com/slyautomation/osrs_yolov5/wiki/Downloading-and-using-LabelImg

### Using this Repo

Add this repo to a project in Pycharm

Step walkthrough adding project with Pycharm: https://github.com/slyautomation/osrs_yolov5/wiki/How-to-add-Project-with-Pycharm

### Ensure Project is within venv (virtual environment)

Step walkthrough activating venv: https://github.com/slyautomation/osrs_yolov5/wiki/how-to-ensure-venv-(virtual-environment)-is-active

## Install Module Requirements

in the terminal type:

```pip install -r requirements.txt```

![image](https://user-images.githubusercontent.com/81003470/147746531-aa622ccb-d6a0-4310-85b7-4775f8b0732a.png)

## Check cuda version is compatiable with torch and torchvision

goto website and check version https://download.pytorch.org/whl/torch_stable.html

To take advantage of the gpu and cuda refer to the list for your cuda version search for cu<version of cuda no spaces or fullstops> e.g cu102 for cuda 10.2.
  
use the latest versions found, i at this point in time found: torch 1.9.0 and torchvision 0.10.0 (these 2 module versions so far i have had no issues other versions i get errors when running <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/detect.py">detect.py</a>)

  ![image](https://user-images.githubusercontent.com/81003470/147751626-8be13bfb-e97d-4642-81db-20955f2a41ad.png)
  
  in the terminal type the torch version + your cuda version (except for torchaudio no cuda version required):
  
  ```pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html```
  
  ![image](https://user-images.githubusercontent.com/81003470/147749033-c5de2a74-5365-444c-93c1-f5d9f75512c4.png)

  ```pip install torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html```
  
  ![image](https://user-images.githubusercontent.com/81003470/147749284-9411be6f-f000-4bf9-a167-b0d214b977f5.png)

  ```pip install torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```
  
- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Make sure when installing torchvision it doesn't try to install another version due to incompatability, try to either find a later version of torch or use a downgraded version of torchvision. there could be issues if another torch version is installed but the cuda version doesn't align with your gpu.`

## Test pytorch and cuda work
  
in the project run <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/main.py">main.py</a>, the output should result in the device used as cuda, and the tensor calculations should run without errors:
  
  ![image](https://user-images.githubusercontent.com/81003470/147753127-c97b0ce4-e9c6-49d4-a817-f9a71928e240.png)

  This will also download the yolov5 weight files:
  
  ![image](https://user-images.githubusercontent.com/81003470/147753307-5c3df94e-206b-4bac-8f2d-8a5e7301c010.png)

# Custom training setup with YAML
  
<p><a href="https://www.kaggle.com/ultralytics/coco128" rel="nofollow">COCO128</a> is an example small tutorial dataset composed of the first 128 images in <a href="http://cocodataset.org/#home" rel="nofollow">COCO</a> train2017. These same 128 images are used for both training and validation to verify our training pipeline is capable of overfitting. <a href="https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml">data/coco128.yaml</a>, shown below, is the dataset config file that defines 1) the dataset root directory <code>path</code> and relative paths to <code>train</code> / <code>val</code> / <code>test</code> image directories (or *.txt files with image paths), 2) the number of classes <code>nc</code> and 3) a list of class <code>names</code>:</p>
  
```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]  # class names
  ```
Copying the method above i have done the same with <a href="https://github.com/slyautomation/osrs_yolov5/blob/master/data/osrs.yaml">data/osrs.yaml</a> 
  
```
# parent
# ├── yolov5
# └── datasets
#     └── osrs ← downloads here
#       └── cow ← add each class
  #     └── xxx ← add each class


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./datasets/osrs  # dataset root dir
train: images/ # train images (relative to 'path') 128 images
val: images/  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['cow']  # class names

```
To start using <a href="https://github.com/slyautomation/osrs_yolov5/blob/master/data/osrs.yaml">data/osrs.yaml</a> run <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/extract_osrs_zip.py">extract_osrs_zip.py</a>, this will unzip the cow.zip.001, cow.zip.002 and cow.zip.003 files
and will create a folder in datasets osrs ready to train the osrs cow model.

# Training

Epochs. Start with 300 epochs. If this overfits early then you can reduce epochs. If overfitting does not occur after 300 epochs, train longer, i.e. 600, 1200 etc epochs.

Image size. COCO trains at native resolution of --img 640, though due to the high amount of small objects in the dataset it can benefit from training at higher resolutions such as --img 1280. If there are many small objects then custom datasets will benefit from training at native or higher resolution. Best inference results are obtained at the same --img as the training was run at, i.e. if you train at --img 1280 you should also test and detect at --img 1280.

Batch size. Use the largest --batch-size that your hardware allows for. Small batch sizes produce poor batchnorm statistics and should be avoided.
In the terminal type:

For more information and tips on datasets, model selection and training settings refer to: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results

The setting i found useful with a GeForce GTX 1060 6GB gpu are as follows. In the terminal type:

```python train.py --data osrs.yaml --weights yolov5s.pt --batch-size 2 --epoch 200``` 
  
![image](https://user-images.githubusercontent.com/81003470/147907954-cd20c621-d848-49e3-83a5-45032ba768ba.png)
  
This will run <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/train.py">train.py</a> with the parameters mentioned above.

![image](https://user-images.githubusercontent.com/81003470/147908948-10a1de98-4eb5-449e-8e24-92e421139b49.png)

## Training Finished

Once finished the resulting model best.pt and last.pt will be saved in the folder runs/train/exp<number>
  
![image](https://user-images.githubusercontent.com/81003470/147910872-6700f739-232e-42f4-a210-479dd7c12734.png)

# Detecting

This is where the detecting of objects take place, based on the parameters given, the code will run the default or custom weights and identify objects (inference) in 
images, videos, directories, streams, etc.
  
## Test Dectections

Run a test to ensure all is installed correctly, in the terminal type:

```python detect.py --source data/images/bus.jpg --weights yolov5s.pt --img 640```

![image](https://user-images.githubusercontent.com/81003470/148015379-5c099720-af00-425a-92b0-0d9e05545cd7.png)

This will run the default yolov5s weight file on the bus image and store the results in runs/detect/exp

These are the labels (the first integer is the class index and the rest are coordinates and bounding areas of the object)

```
5 0.502469 0.466204 0.995062 0.547222 # bus
0 0.917284 0.59213 0.162963 0.450926 # person
0 0.17284 0.603241 0.222222 0.469444 # person
0 0.35 0.588889 0.146914 0.424074 # person
```
 
Here is the resulting image with bounding boxes identifying the bus and people:
  
![bus](https://user-images.githubusercontent.com/81003470/148015666-65439829-1856-435f-a8d0-eea7b9baade0.jpg)

## Test Custom Detections (osrs cows model)

Move the trained model located in runs/train/exp<number> to the parent folder (overwrite the previous best.pt):

![image](https://user-images.githubusercontent.com/81003470/148020954-d42a32b0-b741-4791-8b69-300af762966d.png)
  
Let's see the results for osrs cow detection, to test in the terminal type:
  
``` python detect.py --source data/images/cow_osrs_test.png --weights best.pt --img 640```

The labels results are:
```
  0 0.946552 0.362295 0.062069 0.0754098
0 0.398276 0.460656 0.106897 0.140984
0 0.426724 0.572131 0.105172 0.140984
0 0.352586 0.67377 0.122414 0.167213
0 0.310345 0.898361 0.117241 0.190164
0 0.151724 0.411475 0.062069 0.180328
0 0.705172 0.37541 0.0689655 0.127869
0 0.812931 0.319672 0.087931 0.127869
  ```
And here's the image result:
  
![image](https://user-images.githubusercontent.com/81003470/148021301-f7c58ad1-bc2e-43af-a82f-71a929d5a0cc.png)

## Test Custom Detections (osrs cows model) on Monitor Display (screenshots with Pillow ImageGrab)
 
For a single screenshot, in the terminal type:

```python detect.py --source stream.jpg --weights best.pt --img 640 --use-screen ```

For a constant stream of the monitor display, in the terminal run:
  
```python detect_screenshots.py``` or right click on the detect_screenshots_only.py script and select run:
  
![image](https://user-images.githubusercontent.com/81003470/148022895-e34e65d6-0b6b-4d64-b9dc-a7b7ab38e148.png)

This will run <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/detect_screenshots.py">detect_screenshots.py</a> with the default parameters listed below and can be changed to suit your needs.
  
```
def main_auto():
    run(weights='best.pt',  # model.pt path(s)
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=10,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        Run_Duration_hours=6,  # how long to run code for in hours
        Enable_clicks=False
        )
 ```

For the users that prefer using object orientated programming scripts refer to <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/detect_oob_screenshots">detect_oob_screenshots.py</a>
  
## Test Custom Detections (osrs cows model) on Data Capture
  
To increase the fps (frames per second) and get a better detection rate, i use a hdmi data capture device. This takes a data stream of your monitor displays and sends the data like a webcam, which results in a significant increase in fps compared to taking screenshots of the screen:

In the terminal type:
  
```python detect.py --source 0 --weights best.pt --img 640```
  
See below examples on amazon under $20:
  
-<a target="_blank" href="https://www.amazon.com.au/gp/product/B097YC56QH/ref=as_li_tl?ie=UTF8&camp=247&creative=1211&creativeASIN=B097YC56QH&linkCode=as2&tag=slyautomation-22&linkId=6401772c54cb0d307a4955953cb207ab">MSY Upgraded Version HDMI to USB 3.0 Video Capture Card Recorder Streaming Max HD 1080P with Cable for Game Video Live</a>

![image](https://user-images.githubusercontent.com/81003470/148019089-7c226fd6-382a-4da6-ab37-e8a66a3e838a.png)
  
## Retrain using Detections
  
By default detect.py will take labels and fullscreen images while the detection is running. These will be saved under runs/detect/exp<number>, labels in labels folder and images in crops folder.

![image](https://user-images.githubusercontent.com/81003470/148140979-4b7d1bd1-7e5f-4c58-8102-3e860f1b132b.png)

To save labels and images with detect_screenshots.py set save_text and save_crop to True.
  
![image](https://user-images.githubusercontent.com/81003470/148141081-8bfd281d-94fb-460c-9b6f-c34494e43553.png)

Move the txt files (labels) to datasets/osrs/labels or datasets/[name of dataset]/labels.
  
Move the image files (crops) to datasets/osrs/images or datasets/[name of dataset]/images.

![image](https://user-images.githubusercontent.com/81003470/148141478-95fcd5a0-f68c-4f88-b85f-5a17a97234b9.png)

As mentioned above follow the steps for <a target="_blank" href="https://github.com/slyautomation/osrs_yolov5/blob/main/README.md#training">training</a>.
### Troubleshooting
  
Runtimeerror on Train.py: make sure there is enough hard drive storage space, the models will need approx 20 gbs of space to run smoothly.
  
RuntimeError on Detect.py: Couldn't load custom C++ ops. This can happen if your PyTorch and torchvision versions are incompatible, or if you had errors while compiling torchvision from source. For further information on the compatible versions, check https://github.com/pytorch/vision#installation for the compatibility matrix. Please check your PyTorch version with torch.__version__ and your torchvision version with torchvision.__version__ and verify if they are compatible and if not please reinstall torchvision so that it matches your PyTorch install.

If the above error occurs install a different version of pytorch and install the compatiable torchvision module. TBA i'll add a list of compatiable versions with this project with my gpu and cuda version.
