# 🎨 ArtNest - AI 기반 저작권 보호 시스템

"Protecting Creativity with AI" <br>
작품 표절로 어려움을 겪는 창작자들을 위한 멀티 에이전트 기반 저작권 보호 시스템

본 프로젝트는 [2025 Bolt Hackathon 제출 프로젝트](https://worldslargesthackathon.devpost.com/) 입니다. <br>
해당 레포는 ArtNest의 멀티 에이전트 시스템이 구현되어 있습니다. [프론트엔드 레포 바로가기](https://github.com/tinovadev/artnest-frontend)

## 프로젝트 소개

### 프로젝트 배경
실제 미술 전공자인 동생의 어려움을 바탕으로, ArtNest는 AI 기반 기술을 활용해 디지털 예술 작품의 무단 학습 방지, 표절 탐지, 저작권 추적성 확보를 목표로 개발


## 주요 기능

### 저작권 보호
- Invisible Watermarking: AI 학습을 방해하는 워터마크 삽입
- Adversarial Noise: 작품의 feature space를 교란해 무단 학습 저지

### 유사 이미지 탐지
- CLIP: 텍스트-이미지 기반 스타일 및 구도 유사도 분석
- DISTS, LPIPS: 구조적/지각적 유사도 정량화
- Grad-CAM: 모델의 주목 영역 시각화

### 저작권 추적 및 증명
- 이미지에 삽입된 워터마크를 통해 출처 추적 가능
- 유사 이미지 발견 시 원작자와 AI 학습 여부 검증 지원

## 기술 스택

| 구분          | 사용 기술                                     |
| ----------- | ----------------------------------------- |
| Frontend       | Next.js, TailwindCSS, ShadcnUI, Bolt.new               |
| Multi Agent System  | Google ADK, CrewAI                   |
| AI 모델       | CLIP, DISTS, LPIPS, Grad-CAM              |
| DevOps      | Google Cloud (Cloud Run, GCS, PostgreSQL) |


## 기여자

- 손민하: 멀티 에이전트 개발 및 백엔드, 프론트 개발
- 박경서: Algorand 블록체인 연결 및 백엔드, 프론트 개발
- 배가원: UI/UX 디자인 및 Bolt를 이용한 프로토타입 화면 구현
