import torch
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
      

def get_data_path(filename: str = "") -> str:
    """
    프로젝트 루트 기준으로 data 폴더 내 파일의 절대경로 반환.
    예: get_data_path("a1/a2/image.png")
    """
    project_root = Path(__file__).resolve()
    while not (project_root / "data").exists():
        if project_root.parent == project_root:
            raise FileNotFoundError("data 폴더를 찾을 수 없습니다.")
        project_root = project_root.parent

    return str((project_root / "data" / filename).resolve())
