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

cudnn = https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-10

![image](https://user-images.githubusercontent.com/81003470/147714728-a81e015e-4422-41ba-a8ec-b16b7ac397ae.png)

![image](https://user-images.githubusercontent.com/81003470/147715159-b9de83e0-9fd2-4853-93c6-e0ab84ab661d.png)

![image](https://user-images.githubusercontent.com/81003470/147715257-6448fccc-51a7-4518-a546-15fa7e03c250.png)

![image](https://user-images.githubusercontent.com/81003470/147715323-62f7558d-d23f-4ac3-8fd5-b572e3bd366d.png)




labelImg = https://tzutalin.github.io/labelImg/


# Images and XML files for object detection
example unzip files: cows.z01 , cows.z02 , cows.z03

add image and xml files to folder OID//Dataset//train//name of class

***** IMAGES MUST BE IN JPG FORMAT (use png_jpg to convert png files to jpg files) *******
