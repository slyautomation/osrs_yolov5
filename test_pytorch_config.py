import os

from utils.downloads import attempt_download
import torch
import torch.onnx

def download_weights():
    for x in ['s', 'm', 'l', 'x']:
        attempt_download(f'yolov5{x}.pt')

# run download_weights
# install cuda 10.2
# change coco128.yaml path: ../datasets/coco128  # dataset root dir to path: ./datasets/coco128  # dataset root dir
# comment out train.py sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
# pip install torch===1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchvision===0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchaudio===0.9.0

# $ python train.py --data coco128.yaml --cfg yolov5s.yaml --batch-size 2

# $ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
#                                          yolov5m                                40
#                                          yolov5l                                24
#                                          yolov5x                                16

def test_cpu():
    print(torch.__version__)
    my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
    print(my_tensor)
    torch.cuda.is_available()
def test_gpu():
    print(torch.__version__)

    print(torch.cuda.is_available())

    print(torch.cuda.current_device())

    print(torch.cuda.device(0))

    print(torch.cuda.device_count())

    print(torch.cuda.get_device_name(0))



# pip install torch===1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchvision===0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchaudio===0.9.0

# download_weights()
# test_cpu()
# test_gpu()

import zipfile

zipPath = './datasets/'

with zipfile.ZipFile(zipPath + 'osrs.zip*', 'r') as zip_ref:
    zip_ref.extractall('./datasets/osrs/')
