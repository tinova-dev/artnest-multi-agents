import numpy as np
import os
import random
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


def evaluate(model, image, label):
    """모델의 예측이 label과 일치하는지 확인"""
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)
    return pred.item() == label.item()


def one_pixel_attack(model, image, label, max_trials=100):
    """
    One Pixel Attack 직접 구현 (Foolbox 없이)
    이미지의 단 하나의 픽셀을 변경하여 분류 오류를 유도

    Returns:
        attacked_image: 공격된 이미지
        success: 공격 성공 여부
    """
    attacked_image = image.clone().detach()
    _, _, H, W = attacked_image.shape

    original_pred_correct = evaluate(model, attacked_image, label)

    for _ in range(max_trials):
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)
        c = random.randint(0, 2)

        original_value = attacked_image[0, c, y, x].item()

        # 무작위 픽셀값 설정
        new_value = random.uniform(0, 1)
        attacked_image[0, c, y, x] = new_value

        if not evaluate(model, attacked_image, label):
            return attacked_image, True  # 성공

        # 복원하고 다음 시도
        attacked_image[0, c, y, x] = original_value

    return image.clone(), False  # 실패


async def run_one_pixel_attack(
    image_path: str,
    label_index: int,
    output_dir: str = "./output",
    tool_context: Optional[ToolContext] = None,
):
    """
    One Pixel Attack 실행 Tool (Foolbox 없이 구현)

    Args:
        image_path: 공격할 이미지 경로
        label_index: 원본 이미지의 정답 레이블 인덱스
        output_dir: 결과 저장 디렉토리
        tool_context: ADK 세션 상태
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(pretrained=True).eval().to(device)

    image_path = get_data_path(image_path)
    image = load_image(image_path).to(device)
    label = torch.tensor([label_index]).to(device)

    attacked_image, is_success = one_pixel_attack(model, image, label)

    os.makedirs(output_dir, exist_ok=True)
    original_path = os.path.join(output_dir, "original.png")
    attacked_path = os.path.join(output_dir, "attacked.png")

    save_image(image, original_path)
    save_image(attacked_image, attacked_path)

    result = {
        "original_image_path": original_path,
        "attacked_image_path": attacked_path,
        "is_adversarial": is_success,
    }

    if tool_context:
        tool_context.state["one_pixel_result"] = result

    return result


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
