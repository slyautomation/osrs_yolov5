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
in the terminal type:

cd venv

cd scripts

activate

There should be (venv) in front of your project's location in the terminal:

![image](https://user-images.githubusercontent.com/81003470/147746002-21504567-4224-4d44-9b96-b0334574c4a9.png)


# Images and XML files for object detection
example unzip files: cows.z01 , cows.z02 , cows.z03

add image and xml files to folder OID//Dataset//train//name of class

***** IMAGES MUST BE IN JPG FORMAT (use png_jpg to convert png files to jpg files) *******
