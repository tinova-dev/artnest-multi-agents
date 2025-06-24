import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import torch
import lpips
from PIL import Image
from torchvision import transforms
import os

from utils.utils import get_data_path


def load_image(image_path, resize=(256, 256)):
    """이미지를 로드하고 텐서로 변환합니다."""
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def compute_lpips_similarity(image_path_1: str, image_path_2: str) -> float:
    """
    LPIPS 유사도 계산 함수

    Args:
        image_path_1 (str): 첫 번째 이미지 경로
        image_path_2 (str): 두 번째 이미지 경로

    Returns:
        float: LPIPS distance 값 (0 ~ 1 사이, 낮을수록 유사함)
    """
    
    image_path_1 = get_data_path(image_path_1)
    image_path_2 = get_data_path(image_path_2)
    if not os.path.exists(image_path_1) or not os.path.exists(image_path_2):
        raise FileNotFoundError("입력된 이미지 경로가 존재하지 않습니다.")

    model = lpips.LPIPS(net='alex')
    img1 = load_image(image_path_1)
    img2 = load_image(image_path_2)

    with torch.no_grad():
        distance = model(img1, img2).item()
    return distance


# 예시 실행용
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("사용법: python compute_lpips_similarity.py <image1> <image2>")
        exit(1)

    img1, img2 = sys.argv[1], sys.argv[2]
    score = compute_lpips_similarity(img1, img2)
    print(f"LPIPS Distance: {score:.4f}")
