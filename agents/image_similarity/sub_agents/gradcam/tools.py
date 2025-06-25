import torch
from PIL import Image
import numpy as np
import cv2
import argparse
import torchvision

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
    original_img, original_img_float, original_tensor = get_image_with_path(image_path_a)
    suspected_img, suspected_img_float, suspected_tensor  = get_image_with_path(image_path_b)
    
    original_concept_features = model(original_tensor)[0, :]
    suspected_concept_features = model(suspected_tensor)[0, :]
    
    #
    target_layers = [resnet.layer4[-1]]
    original_targets = [SimilarityToConceptTarget(original_concept_features)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        car_grayscale_cam = cam(input_tensor=suspected_tensor,
                            targets=original_targets)[0, :]
    original_cam_image = show_cam_on_image(suspected_img_float, car_grayscale_cam, use_rgb=True)
    Image.fromarray(original_cam_image)

    # 
    # not_original_targets = [DifferenceFromConceptTarget(original_concept_features)]
    
    combined_image = np.hstack((original_img, suspected_img, original_cam_image))
    combined_pil_image = Image.fromarray(combined_image)
    
    save_path = get_data_path("gradcam.png")
    combined_pil_image.save(save_path)

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
