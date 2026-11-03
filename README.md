# CNN 작물 이미지 분류

PyTorch로 CNN을 직접 설계해 작물 3종(고추 / 배추 / 오이)을 분류한 프로젝트.

---

## 프로젝트 소개

- AI Hub 노지 작물 질병 진단 이미지 중 정상 이미지를 사용했고, 클래스당 2,000장으로 언더샘플링한 6,000장으로 3종 분류 모델을 구현했다.
- EDA, 데이터 파이프라인, CNN 설계 및 학습, Conv 블록 수(3 / 4 / 5) 비교 실험을 노트북 4개로 나누어 진행했다.

---

## 적용 이론 및 구현 내용

**핵심 역량: 데이터 탐색부터 커스텀 Dataset, CNN 설계, 학습 루프, 깊이 비교 실험까지 PyTorch로 직접 구현**

| 역량 | 구현 내용 |
|------|-----------|
| EDA | 클래스별 이미지 수, 해상도 분포(클래스당 200장 샘플), RGB 채널 분포, 깨진 이미지 검사를 수행했고 그 결과로 언더샘플링 / 224×224 리사이즈 / RGB 유지를 결정했다. |
| 데이터 분할 | `train_test_split`과 `stratify`로 train 80% / val 10% / test 10% (4,800 / 600 / 600장)을 분할했고 `SEED=42`로 재현성을 확보했다. |
| 정규화 | train 이미지 1,500장을 샘플링해 RGB 채널별 mean / std를 계산했고 train만 사용해 val / test 정보 유입을 막았다. |
| 커스텀 Dataset | `Dataset`을 상속해 `__len__` / `__getitem__`을 구현했고 transform을 외부에서 주입해 train과 val/test 전처리를 분리했다. |
| 데이터 증강 | train에만 `RandomHorizontalFlip`을 적용했고 val / test는 증강 없이 평가했다. |
| CNN 설계 | `Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d` 블록 4개(채널 32 / 64 / 128 / 256), `AdaptiveAvgPool2d`, `Dropout(0.5)`를 적용한 `CropCNN`(422,659 파라미터)을 구현했다. |
| 학습 루프 | `CrossEntropyLoss`, `Adam(lr=1e-3)`, 최대 50 에폭으로 train / val 루프를 직접 작성했고 val_loss 기준 Early Stopping(patience=5)과 val_acc 기준 best 모델 저장을 적용했다. |
| 평가 | best 모델로 test set의 Confusion Matrix와 클래스별 정확도를 산출했다. |
| 비교 실험 | Conv 블록 3 / 4 / 5개 모델(`ShallowCNN` / `CurrentCNN` / `DeepCNN`)을 같은 조건으로 학습해 파라미터 수와 Best Val Acc를 비교했다. |
| 실행 환경 | Docker로 Jupyter Notebook 실행 환경을 구성해 GPU(RTX 3080) 학습 환경을 재현할 수 있게 했다. |

---

## 구현 결과 및 실행

### 결과 요약

**모델 깊이 비교 (동일 조건, `04. experiments`)**

| 모델 | Conv 블록 | 파라미터 | Best Val Acc | 중단 에폭 |
|------|-----------|----------|--------------|-----------|
| ShallowCNN | 3 | 102,147 | 0.9467 | 15 |
| **CurrentCNN** | **4** | **422,659** | **0.9717** | 29 |
| DeepCNN | 5 | 1,702,659 | 0.9600 | 10 |

- 5블록은 파라미터가 4블록의 약 4배인데 Best Val Acc 차이는 1.17%p였고, 4블록을 기준 모델로 사용했다.

**기준 모델 Test 평가 (`03.model`)**

| 항목 | 결과 |
|------|------|
| 학습 | 28 에폭에서 Early Stopping, Best Val Acc 0.9733 |
| Test Accuracy | 0.9717 (600장) |
| 클래스별 정확도 | cabbage 0.9950 / cucumber 0.9800 / pepper 0.9400 |

### 스크린샷

<table>
  <tr>
    <td align="center">
      <img src="results/figures/05_training_curves.png" width="480"><br>
      <sub>학습 곡선: epoch별 train / val Loss, Accuracy (28 에폭)</sub>
    </td>
    <td align="center">
      <img src="results/figures/06_confusion_matrix.png" width="480"><br>
      <sub>Test Confusion Matrix와 클래스별 정확도</sub>
    </td>
  </tr>
</table>

<details>
<summary>모델 깊이 비교 곡선</summary>

<img src="results/figures/07_depth_comparison.png" width="800"><br>
<sub>ShallowCNN / CurrentCNN / DeepCNN의 Val Loss, Val Accuracy 비교</sub>

</details>

<details>
<summary>클래스 분포</summary>

<img src="results/figures/01_class_distribution.png" width="600"><br>
<sub>언더샘플링 전 클래스별 이미지 수: pepper 2,705 / cabbage 2,008 / cucumber 2,227</sub>

</details>

<details>
<summary>샘플 이미지</summary>

<img src="results/figures/02_sample_images.png" width="800"><br>
<sub>클래스별 샘플 5장 (224×224)</sub>

</details>

<details>
<summary>RGB / Grayscale 비교, 채널 분포</summary>

<img src="results/figures/03_rgb_vs_gray.png" width="500"><br>
<sub>RGB vs Grayscale 비교</sub>

<img src="results/figures/04_rgb_channel_dist.png" width="800"><br>
<sub>클래스별 RGB 채널 픽셀 분포</sub>

</details>

### 사용 기술

| 분류 | 내용 |
|------|------|
| 언어 | Python, Jupyter Notebook |
| 표준 라이브러리 | os, random, pathlib, collections |
| 외부 라이브러리 | PyTorch, torchvision, torchinfo, scikit-learn, NumPy, Pandas, Matplotlib, seaborn, Pillow, tqdm |
| 환경 | Docker, NVIDIA RTX 3080 (CUDA) |
| 코드 작성 보조 | Claude (Anthropic) |

### 개선 방향

- 검증 손실이 학습 후반(17, 20, 22, 26 에폭)에도 반복적으로 튀었고, 학습률 스케줄러 등으로 변동을 줄이는 실험 여지가 있다.
- 깊이 비교는 동일 시드로 1회씩 실행한 결과이고 val 600장 기준 1.17%p는 약 7장 차이여서, 시드를 바꿔 반복하면 비교의 신뢰도를 높일 수 있다.
- DeepCNN은 10 에폭에서 Early Stopping이 걸려 일찍 중단되었고, patience 조정 후 재비교하면 5블록 성능을 더 공정하게 볼 수 있다.
- Test 오분류는 pepper가 cucumber로 예측된 12장이 가장 많았고, 해당 이미지를 확인해 원인을 분석하는 작업이 남아 있다.

### 실행 방법

1. `Dockerfile`로 이미지를 빌드해 Jupyter Notebook 환경을 실행한다.
2. AI Hub 노지 작물 질병 진단 데이터에서 고추 / 배추 / 오이 원천 이미지를 받아 `data/raw/Training`, `data/raw/Validation` 아래에 배치한다. (`data/`는 `.gitignore` 대상)
3. 노트북을 `01` → `02` → `03` → `04` 순서로 실행한다. (`03`, `04`는 데이터 경로가 `/data_local/raw`로 지정되어 있어 환경에 맞게 `RAW_DIR`을 수정)
4. `03.model` 실행 시 best 모델이 `models/best_model.pth`에 저장된다. (`models/`는 `.gitignore` 대상)
5. 실행 안 될 경우 PyTorch 버전을 고정한다.

### 파일 구성

```
cnn-crop-classifier
 ┣ notebooks
 ┃ ┣ 01.EDA.ipynb              # 클래스 분포, 해상도, 채널 분석
 ┃ ┣ 02.data_pipeline.ipynb    # 분할, 정규화 통계, Dataset, DataLoader
 ┃ ┣ 03.model.ipynb            # CropCNN, 학습 루프, Test 평가
 ┃ ┗ 04. experiments.ipynb     # Conv 블록 3 / 4 / 5 비교 실험
 ┣ results
 ┃ ┗ figures                   # 노트북에서 저장한 그래프 7장
 ┣ Dockerfile                  # Jupyter Notebook 실행 환경
 ┣ requirements.txt            # 패키지 목록
 ┣ .gitignore                  # data, models 제외
 ┗ README.md
```
