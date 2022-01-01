

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

# Custom training
  
<a href="https://www.kaggle.com/ultralytics/coco128" rel="nofollow">COCO128</a><p><a href="https://www.kaggle.com/ultralytics/coco128" rel="nofollow">COCO128</a> is an example small tutorial dataset composed of the first 128 images in <a href="http://cocodataset.org/#home" rel="nofollow">COCO</a> train2017. These same 128 images are used for both training and validation to verify our training pipeline is capable of overfitting. <a href="https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml">data/coco128.yaml</a>, shown below, is the dataset config file that defines 1) the dataset root directory <code>path</code> and relative paths to <code>train</code> / <code>val</code> / <code>test</code> image directories (or *.txt files with image paths), 2) the number of classes <code>nc</code> and 3) a list of class <code>names</code>:</p>
  
# Images and XML files for object detection
example unzip files: cows.z01 , cows.z02 , cows.z03

add image and xml files to folder OID//Dataset//train//name of class

***** IMAGES MUST BE IN JPG FORMAT (use png_jpg to convert png files to jpg files) *******
