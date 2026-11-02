# CNN 작물 이미지 분류

## 프로젝트 개요
- CNN 기반 작물 이미지 분류
- 파이프라인 전 과정 직접 구현
- 설계 근거 기반 실험 진행

## 데이터
- 출처: AI Hub — '노지 작물 질병 진단 이미지'
- 클래스: 고추 / 배추 / 오이 (3종, 정상 이미지만 사용)
- 구성: train / val / test

## 폴더 구조
```
cropclassifier/
├── data/raw/          # 원본 이미지
├── notebooks/         # 실험 노트북
├── results/figures/   # 저장된 그래프
└── models/            # 저장된 모델
```


## 실험 구성
- 01_EDA: 데이터 분석 → 전처리 방향 결정
- 02_data_pipeline: Dataset / DataLoader 직접 구현
- 03_model: CNN 구현 + 학습 루프 작성
- 04_experiments: RGB vs Gray / Augmentation / 모델 깊이 비교
- 05_results: Confusion Matrix / Grad-CAM / 최종 분석

## 실행 환경
- Python 3.12
- PyTorch 2.6 (CUDA 12.4)
- GPU: NVIDIA RTX 3080