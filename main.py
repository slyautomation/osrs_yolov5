from utils.downloads import attempt_download
import torch
import torch.onnx

def download_weights():
    for x in ['s', 'm', 'l', 'x']:
        attempt_download(f'yolov5{x}.pt')

# run download_weights
# pip install torch===1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchvision===0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torchaudio===0.9.0

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


test_cpu()
test_gpu()
download_weights()