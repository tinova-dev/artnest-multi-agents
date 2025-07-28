import torch
from PIL import Image
import open_clip
import torch.nn.functional as F

from utils.utils import get_data_path


# 모델 및 전처리기 초기화
def load_openclip_model(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device='mps' if torch.backends.mps.is_available() else 'cpu'):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    print('Available Device: ', device)
        
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer, device


# 이미지 임베딩 추출 함수
def get_image_embedding(image_path: str) -> torch.Tensor:
    model, preprocess, _, device = load_openclip_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # 정규화

    return image_features.squeeze()


# 유사도 계산 함수 (Cosine Similarity)
def compute_clip_similarity(img1_path: str, img2_path: str, threshold: float = 0.9):
    img1_path = get_data_path(img1_path)
    img2_path = get_data_path(img2_path)
    
    emb1 = get_image_embedding(img1_path)
    emb2 = get_image_embedding(img2_path)

    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    is_similar = similarity >= threshold

    print(f"✅ Cosine Similarity: {similarity:.4f}")
    print(f"🧐 유사한 이미지인가요? {'Yes ✅' if is_similar else 'No ❌'}")
    return similarity, is_similar


# ✅ 4. 실행 예시
if __name__ == "__main__":
    img_path1 = "tiger.png"
    img_path2 = "fake-tiger.png"
    
    compute_clip_similarity(img_path1, img_path2, threshold=0.9)
