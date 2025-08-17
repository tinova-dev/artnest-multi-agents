# 🎨 ArtNest - AI 기반 저작권 보호 시스템

"Protecting Creativity with AI" <br>
작품 표절로 어려움을 겪는 창작자들을 위한 멀티 에이전트 기반 저작권 보호 시스템

본 프로젝트는 [2025 Bolt Hackathon 참여 프로젝트](https://worldslargesthackathon.devpost.com/) 입니다. <br>
해당 레포는 ArtNest의 멀티 에이전트 시스템이 구현되어 있습니다. -> [프론트엔드 레포 바로가기](https://github.com/tinovadev/artnest-frontend)

## 프로젝트 소개

### 프로젝트 배경
실제 미술 전공자인 동생의 어려움을 바탕으로, ArtNest는 AI 기반 기술을 활용해 디지털 예술 작품의 무단 학습 방지, 표절 탐지, 저작권 추적성 확보를 목표로 개발

### 프로젝트 개요
- 기간: 2025년 5월 - 6월
- 팀 구성
  - 손민하: 멀티 에이전트 개발 (100%) 및 프론트, 백엔드 개발 (50%)
  - 박경서: Algorand 블록체인 연결 (100%) 및 프론트, 백엔드 개발 (50%)
  - 배가원: UI/UX 디자인
- 프로젝트 제출 페이지 및 데모 영상: https://devpost.com/software/project-1qlz5jvgpocy


## 주요 기능

### 웹 스크롤링 기반 이미지 탐색
- 사용자가 버튼을 클릭하면 Puppeteer를 이용해 웹 페이지를 자동 스크롤
- 온라인에서 사용자의 작품과 유사한 이미지를 수집 및 후보군으로 확보
- 이후 멀티 에이전트 시스템이 수집된 이미지를 분석하여 표절 가능성 평가

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

## ERD

## 기술 스택

| 구분               | 사용 기술                                 |
| ------------------ | ----------------------------------------- |
| Frontend, Backend  | Next.js, TailwindCSS, ShadcnUI, Bolt.new  |
| Multi Agent System | Google ADK, CrewAI                        |
| AI 모델            | CLIP, DISTS, LPIPS, Grad-CAM              |
| DevOps             | Google Cloud (Cloud Run, GCS, PostgreSQL) |

### 기술 스택 선정 이유

Bolt.new 해커톤 제약
- 해커톤 규칙상 대부분의 개발을 Bolt.new 환경에서 진행해야 했음
- 따라서 웹은 프론트엔드, 백엔드 개발이 전부 가능한 **NextJS** 프레임워크로 개발

[Google Cloud 해커톤](https://googlecloudmultiagents.devpost.com/?ref_feature=challenge&ref_medium=discover&_gl=1*6cd8jv*_gcl_au*MTYwNDI5NTM4NC4xNzU1NDI5NDc2*_ga*MTk2MzcxNTkwNi4xNzU1NDI5NDc2*_ga_0YHJK3Y10M*czE3NTU0Mjk0NzUkbzEkZzEkdDE3NTU0Mjk1MzckajYwJGwwJGgw) 병행 시도

- Bolt.new 는 JavaSciprt만 지원했기에 Python 기반의 멀티 에이전트 시스템은 별도 개발이 필요한 상황
- 따라서 Google Cloud 해커톤에 참여하여 $100 불의 크레딧을 활용해 **Gemini, Cloud Run, Google Cloud Storage (GCS), PostgreSQL** 을 이용해 멀티 에이전트 시스템을 구축 및 배포
- 다만 시간 부족으로 Google Cloud 해커톤은 최종 출품까지 이어지지 못함

## 회고 및 배운점

### 웹 크롤링의 한계

- Puppeteer를 활용해 자동 스크롤링 기반 이미지 수집을 구현했으나, 구글에서 반복 요청으로 인해 IP가 차단되는 문제가 발생
- Headless 모드, User-Agent 변경, 요청 간격 조정 등 여러 우회 방법을 시도했으나 해커톤 기간 내에는 안정적인 해결에 한계가 있었음
- 단순 웹 스크래핑 방식의 한계를 확인했고, 서비스 수준에서는 공식 API 활용을 최우선으로 하고, 프록시 로테이션, 요청 지연(backoff)등의 다른 전략이 필요함을 학습

### 저작권 방지 마킹 취약성

- 이미지에 adversarial noise를 삽입하여 AI 모델이 작품을 학습하려는 것을 방지하려고 했지만, GPT 모델이 마킹을 무시하고 원본 이미지를 그대로 설명
- 이를 통해 단순한 노이즈 삽입은 AI 모델 학습 방어에 효과적이지 않음을 확인
- 더 정교한 adversarial perturbation, Opt-out 메타데이터 등 최신 연구 기반 기법을 병행해야 한다는 인사이트를 얻음