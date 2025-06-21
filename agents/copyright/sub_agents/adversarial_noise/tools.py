import numpy as np
import os
import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import resnet18

from PIL import Image
from typing import Optional
from google.adk.tools import ToolContext

from copyright.utils.utils import get_data_path


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """이미지 불러오기 및 전처리"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str):
    """Tensor를 이미지로 저장"""
    image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    image.save(path)


def fgsm_attack(model, image, label, epsilon: float):
    """FGSM 공격 함수"""
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, data_grad


async def run_fgsm_attack(
    image_path: str,
    label_index: int,
    epsilon: float = 0.1,
    output_dir: str = "./output",
    tool_context: Optional[ToolContext] = None,
):
    """
    FGSM 공격 실행 Tool (Google ADK Tool 호환)

    Args:
        image_path: 원본 이미지 경로
        label_index: 정답 레이블 인덱스 (예: 3)
        epsilon: 공격 강도 (0 ~ 1)
        output_dir: 출력 이미지 저장 경로
        tool_context: ADK 세션 상태 객체
    """


    model = resnet18(pretrained=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    image_path = get_data_path(image_path)

    # 이미지 로드
    image = load_image(image_path).to(device)
    label = torch.tensor([label_index]).to(device)

    # 공격 수행
    perturbed_image, grad = fgsm_attack(model, image, label, epsilon)

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    original_path = os.path.join(output_dir, "original.png")
    perturbed_path = os.path.join(output_dir, "perturbed.png")
    diff_path = os.path.join(output_dir, "perturbation.png")

    save_image(image, original_path)
    save_image(perturbed_image, perturbed_path)

    # Perturbation visualization (gradient 값 시각화)
    perturbation = grad.squeeze(0).detach().cpu().numpy()
    perturbation = np.abs(perturbation).mean(axis=0)
    perturbation_img = Image.fromarray(np.uint8(perturbation * 255))
    perturbation_img = perturbation_img.convert("L")
    perturbation_img.save(diff_path)

    result = {
        "original_image_path": original_path,
        "perturbed_image_path": perturbed_path,
        "perturbation_image_path": diff_path,
        "epsilon": epsilon,
    }

    # ADK 상태에 저장
    if tool_context:
        tool_context.state["fgsm_result"] = result

    return result
