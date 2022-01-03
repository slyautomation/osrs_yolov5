

# Yolov5 Object Detection In OSRS using Python code, Detecting Cows - Botting

name: GeForce GTX 1060 6GB

# Quick Start

### installing pycharm


pycharm = https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC

Step walkthrough Installation of pycharm: https://github.com/slyautomation/osrs_yolov5/wiki/How-to-Install-Pycharm

### Check your cuda version

type in terminal: nvidia-smi

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


### Using this Repo

Add this repo to a project in Pycharm

Step walkthrough adding project with Pycharm: https://github.com/slyautomation/osrs_yolov5/wiki/How-to-add-Project-with-Pycharm

### Ensure Project is within venv (virtual environment)

Step walkthrough activating venv: https://github.com/slyautomation/osrs_yolov5/wiki/how-to-ensure-venv-(virtual-environment)-is-active

## Install Module Requirements

in the terminal type:

pip install -r requirements.txt

![image](https://user-images.githubusercontent.com/81003470/147746531-aa622ccb-d6a0-4310-85b7-4775f8b0732a.png)

## Check cuda version is compatiable with torch and torchvision

goto website and check version https://download.pytorch.org/whl/torch_stable.html

To take advantage of the gpu and cuda refer to the list for your cuda version search for cu<version of cuda no spaces or fullstops> e.g cu102 for cuda 10.2.
  
use the latest versions found, i at this point in time found: torch 1.10.1 and torchvision 0.11.2

  ![image](https://user-images.githubusercontent.com/81003470/147751626-8be13bfb-e97d-4642-81db-20955f2a41ad.png)
  
  in the terminal type the torch version + your cuda version (except for torchaudio no cuda version required):
  
  pip install torch==1.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
  
  ![image](https://user-images.githubusercontent.com/81003470/147749033-c5de2a74-5365-444c-93c1-f5d9f75512c4.png)

  pip install torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/torch_stable.html
  
  ![image](https://user-images.githubusercontent.com/81003470/147749284-9411be6f-f000-4bf9-a167-b0d214b977f5.png)

- ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) `Make sure when installing torchvision it doesn't try to install another version due to incompatability, try to either find a later version of torch or use a downgraded version of torchvision. there could be issues if another torch version is installed but the cuda version doesn't align with your gpu.`

## Test pytorch and cuda work
  
in the project run main.py, the output should result in the device used as cuda, and the tensor calculations should run without errors:
  
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

### Training

Epochs. Start with 300 epochs. If this overfits early then you can reduce epochs. If overfitting does not occur after 300 epochs, train longer, i.e. 600, 1200 etc epochs.

Image size. COCO trains at native resolution of --img 640, though due to the high amount of small objects in the dataset it can benefit from training at higher resolutions such as --img 1280. If there are many small objects then custom datasets will benefit from training at native or higher resolution. Best inference results are obtained at the same --img as the training was run at, i.e. if you train at --img 1280 you should also test and detect at --img 1280.

Batch size. Use the largest --batch-size that your hardware allows for. Small batch sizes produce poor batchnorm statistics and should be avoided.
In the terminal type:

For more information and tips on datasets, model selection and training settings refer to: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results

The setting i found useful with a GeForce GTX 1060 6GB gpu are as follows. In the terminal type:

python train.py --data osrs.yaml --weights yolov5s.pt --batch-size 2 --epoch 200 
  
![image](https://user-images.githubusercontent.com/81003470/147907954-cd20c621-d848-49e3-83a5-45032ba768ba.png)
  
This will run <a href="https://github.com/slyautomation/osrs_yolov5/blob/main/train.py">train.py</a> with the parameters mentioned above.

![image](https://user-images.githubusercontent.com/81003470/147908948-10a1de98-4eb5-449e-8e24-92e421139b49.png)

### Training Finished

Once finished the resulting model best.pt and last.pt will be saved in the folder runs/train/exp<number>
  
![image](https://user-images.githubusercontent.com/81003470/147910872-6700f739-232e-42f4-a210-479dd7c12734.png)

### Troubleshooting
  
Runtimeerror on Train.py: make sure there is enough hard drive storage space, the models will need approx 20 gbs of space to run smoothly.
  
