# Yolov5 Object Detection In OSRS using Python code, Detecting Cows - Botting

name: GeForce GTX 1060 6GB

# Quick Start

### installing pycharm

pycharm = https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=windows&code=PCC

once downloaded: click on the file to start ithe installisation:

![image](https://user-images.githubusercontent.com/81003470/147712917-e0f87af5-9f67-4fff-a490-4e0958a56871.png)

the installisation program will start click on the next buttons, if applicable uninstall any previous version and 'Uninstall Silently to ensure no settings are lost by checking the boxes:

![image](https://user-images.githubusercontent.com/81003470/147713017-9786df65-773c-4852-8b0c-f9929938fffd.png)

Make sure to check the boxes for updating the PATH Variable and create associations, anything else is optional:

![image](https://user-images.githubusercontent.com/81003470/147713061-8eed0194-0ed9-4323-9695-bb621980a432.png)

### Check your cuda version

type in terminal: nvidia-smi

![image](https://user-images.githubusercontent.com/81003470/147712277-5b1fae1d-33b2-4ff0-a4de-19ef762e1b14.png)

my version that i can use is up to: 11.5 but for simplicity i can use previous versions namely 10.0 and 10.2

Check if your gpu will work: https://developer.nvidia.com/cuda-gpus and use the cuda for your model and the latest cudnn for the cuda version.

full list of cuda versions: https://developer.nvidia.com/cuda-toolkit-archive

cuda 10.0 = https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10

cuda 10.2 = https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe

cudnn = https://developer.nvidia.com/rdp/cudnn-archive#a-collapse765-10








labelImg = https://tzutalin.github.io/labelImg/


# Images and XML files for object detection
example unzip files: cows.z01 , cows.z02 , cows.z03

add image and xml files to folder OID//Dataset//train//name of class

***** IMAGES MUST BE IN JPG FORMAT (use png_jpg to convert png files to jpg files) *******
