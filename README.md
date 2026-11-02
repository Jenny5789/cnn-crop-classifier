# CNN 작물 이미지 분류

PyTorch CNN으로 작물 3종(고추 / 배추 / 오이)을 분류한 프로젝트.  
데이터 준비부터 학습, 모델 깊이 비교 실험까지 파이프라인 전 과정을 직접 구현했다.

---

## 실험 결과

### 모델 깊이 비교 (Conv 블록 수)

| 모델 | Conv 블록 | 파라미터 | Best Val Acc |
|------|-----------|----------|--------------|
| ShallowCNN | 3블록 | 102K | 0.9467 |
| **CurrentCNN** | **4블록** | **422K** | **0.9717** |
| DeepCNN | 5블록 | 1,702K | 0.9600 |

**선택 근거**: DeepCNN은 파라미터가 4배(422K → 1,702K)임에도 성능 차이 1.17%p에 그침.  
작물 3클래스 분류 태스크에서 4블록이 적정 복잡도.

### CurrentCNN 학습 특성
- 초반(~15 에폭): 검증 손실 변동 폭 큼
- 15 에폭 이후: 안정적으로 수렴
- Early Stopping으로 최적 체크포인트 포착

---

## 데이터

| 항목 | 내용 |
|------|------|
| 출처 | AI Hub — 노지 작물 질병 진단 이미지 |
| 클래스 | 고추 / 배추 / 오이 (정상 이미지) |
| 클래스당 샘플 수 | 2,000장 (언더샘플링) |
| 전체 | 6,000장 |
| 분할 | Train 70% / Val 20% / Test 10% |
| 정규화 | 데이터셋 직접 계산 (mean, std) |

---

## 노트북 구성

| 노트북 | 내용 |
|--------|------|
| `01.EDA` | 클래스 분포, 샘플 이미지, RGB/Grayscale 채널 분석 |
| `02.data_pipeline` | CustomDataset, DataLoader, 정규화 통계 계산 |
| `03.model` | CropCNN 구현, 학습 루프, Confusion Matrix |
| `04.experiments` | 모델 깊이 비교 (3 / 4 / 5 Conv 블록) |

---

## 기술 스택

| 분류 | 내용 |
|------|------|
| 언어 | Python 3.12 |
| 프레임워크 | PyTorch 2.6, torchvision |
| 주요 라이브러리 | scikit-learn, Matplotlib, Pillow, NumPy, Pandas, tqdm, torchinfo |
| 환경 | Docker (Linux 컨테이너), Jupyter Notebook |
| GPU | NVIDIA RTX 3080 (CUDA 11.8) |
| 코드 작성 보조 | Claude (Anthropic) |
