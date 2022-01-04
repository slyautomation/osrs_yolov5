import pathlib

from utils.downloads import attempt_download
import torch
import torch.onnx
import torchvision.io
def download_weights():
    for x in ['s', 'm', 'l', 'x']:
        attempt_download(f'yolov5{x}.pt')

directory = pathlib.Path(__file__).parent.resolve()
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print('testing pytorch with cpu first....')
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
print('result is:', my_tensor)

#Additional Info when using cuda
if device.type == 'cuda':
    print('pytorch version used:', torch.__version__)
    print('pytorch and cuda is working:', torch.cuda.get_device_name(0))
    print('pytorch and cuda is active:', print(torch.cuda.is_available()))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda")
    print('testing pytorch with cuda...result is:', my_tensor)
    ok = torchvision.io.read_image(str(directory) + "/data/images/cow_osrs_test.png")
    print(ok)
download_weights()
