import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import argparse
import torchvision
from typing import List

from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM


from utils.utils import get_data_path

class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]
        
resnet = torchvision.models.resnet50(pretrained=True)
resnet.eval()
model = ResnetFeatureExtractor(resnet)

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)
    
class DifferenceFromConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return 1 - cos(model_output, self.features)


def add_title_to_image(image: np.ndarray, title: str, font_size: int = 30, title_height: int = 40) -> np.ndarray:
    """
    텍스트 크기와 무관하게 고정된 높이로 제목 공간 확보

    Args:
        image (np.ndarray): 원본 이미지
        title (str): 제목 텍스트
        font_size (int): 폰트 크기
        title_height (int): 고정된 제목 영역 높이

    Returns:
        np.ndarray: 제목이 포함된 새로운 이미지
    """
    pil_img = Image.fromarray(image)
    img_w, img_h = pil_img.size

    font = ImageFont.load_default(font_size)

    # 새 이미지 생성: 위에 title_height만큼 공간 추가
    new_img = Image.new("RGB", (img_w, img_h + title_height), (255, 255, 255))
    new_img.paste(pil_img, (0, title_height))

    # 텍스트 추가 (위쪽 가운데 정렬)
    draw = ImageDraw.Draw(new_img)
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    draw.text(((img_w - text_w) / 2, (title_height - text_h) / 2), title, fill=(0, 0, 0), font=font)

    return np.array(new_img)


def get_image_with_path(path: str):
    img_bgr = cv2.imread(path)  # 이미지 로드 (BGR)
    if img_bgr is None:
        raise ValueError(f"이미지를 찾을 수 없습니다: {path}")
    
    img_bgr = cv2.resize(img_bgr, (512, 512))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # RGB로 변환

    rgb_img_float = np.float32(img_rgb) / 255.0

    input_tensor = preprocess_image(
        rgb_img_float,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    return img_rgb, rgb_img_float, input_tensor


def generate_gradcam_heatmap(
    input_tensor: torch.Tensor,
    input_float_image: np.ndarray,
    targets: List,
) -> np.ndarray:
    """
    Grad-CAM을 이용해 주어진 이미지에서 개념 벡터(concept_vector)와의 유사도 기반으로
    주목한 영역을 heatmap으로 반환

    Args:
        model: Grad-CAM에 사용할 모델
        input_tensor (torch.Tensor): 입력 이미지 텐서 (1, 3, H, W)
        input_float_image (np.ndarray): 0~1 사이로 정규화된 이미지 (H, W, 3), RGB
        concept_vector (torch.Tensor): 비교할 개념 벡터 (1D)
        target_layers (List): Grad-CAM에서 사용할 레이어 리스트

    Returns:
        np.ndarray: RGB heatmap 이미지 (uint8)
    """
    
    target_layers = [resnet.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    heatmap = show_cam_on_image(input_float_image, grayscale_cam, use_rgb=True)
    return heatmap


def compute_pixel_attribution(
    image_path_a: str,
    image_path_b: str,
) -> str:
    """
    이미지 A가 이미지 B와 비교했을 때 임베딩 차이에 얼마나 영향을 주는지
    픽셀 단위로 표시한 heatmap 이미지 생성

    Args:
        image_path_a (str): 변경 이미지 경로 (표절 의심 이미지)
        image_path_b (str): 기준 이미지 경로 (원작)
        save_path (str): 저장할 heatmap 경로

    Returns:
        str: 저장된 heatmap 경로
    """

    # Load and preprocess images
    image_path_a = get_data_path(image_path_a)
    image_path_b = get_data_path(image_path_b)
    original_img, _, original_tensor = get_image_with_path(image_path_a)
    suspected_img, suspected_img_float, suspected_tensor  = get_image_with_path(image_path_b)
    
    original_concept_features = model(original_tensor)[0, :]
    
    #
    original_targets = [SimilarityToConceptTarget(original_concept_features)]
    
    original_cam_image = generate_gradcam_heatmap(
        input_tensor=suspected_tensor,
        input_float_image=suspected_img_float,
        targets=original_targets
    )   
    
    # 
    not_original_targets = [DifferenceFromConceptTarget(original_concept_features)]
    
    not_original_cam_image = generate_gradcam_heatmap(
        input_tensor=suspected_tensor,
        input_float_image=suspected_img_float,
        targets=not_original_targets
    )   
    
    original_img_labeled = add_title_to_image(original_img, "Original Image")
    suspected_img_labeled = add_title_to_image(suspected_img, "Suspected Image")
    original_cam_image_labeled = add_title_to_image(original_cam_image, "Which area is suspected")
    not_original_cam_image_labeled = add_title_to_image(not_original_cam_image, "GradCAM (Not Original)")
    
    result_image = np.hstack((original_img_labeled, suspected_img_labeled, original_cam_image_labeled))
    result_pil_image = Image.fromarray(result_image)
    
    save_path = get_data_path("gradcam.png")
    result_pil_image.save(save_path)

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pixel-level attribution visualization using CLIP embeddings."
    )
    parser.add_argument("image_b", type=str, help="Path to reference image (image A)")
    parser.add_argument("image_a", type=str, help="Path to suspect image (image B)")

    args = parser.parse_args()

    output = compute_pixel_attribution(args.image_a, args.image_b)
    print(f"✅ Heatmap saved to: {output}")
