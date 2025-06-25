import torch

from torchvision import transforms
from PIL import Image
from pathlib import Path

def is_gpu_available() -> str:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda found!')
        return device
    else:
        device = torch.device('cpu')
        print('no cuda found, will use cpu')
        return device


def get_data_path(foldername: str = "", filename: str = "") -> str:
    project_root = Path(__file__).resolve()
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise FileNotFoundError("data 폴더를 찾을 수 없습니다.")
        project_root = project_root.parent

    path = project_root / "data"
    if foldername:
        path = path / foldername
    if filename:
        path = path / filename
        
    return str(path.resolve())


def load_image(image_path, resize=(256, 256)):
    """이미지를 로드하고 텐서로 변환합니다."""
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)