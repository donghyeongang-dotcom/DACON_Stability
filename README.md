# 🏗️ 듀얼 스트림(Dual-Stream) 기반 구조물 붕괴 예측 모델

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

본 레포지토리는 DACON '구조물 붕괴 예측 AI 경진대회'의 추론 및 학습 파이프라인 코드를 담고 있습니다. 
구조물의 정면(Front)과 상단(Top) 두 가지 시점의 이미지를 분석하여 붕괴 여부(Stable/Unstable)를 이진 분류하는 딥러닝 모델입니다.

# 💡 Architecture: Dual-Stream ResNet50 (Pseudo-Siamese)


구조물의 붕괴를 예측하기 위해서는 시점에 따라 다른 물리적 특성을 이해해야 합니다.
* 정면(Front): 중력 방향의 하중 지지 및 구조적 비대칭성
* 상단(Top): 무게 중심의 쏠림 및 전체적인 균형

이를 모델에 반영하기 위해 두 이미지의 가중치를 공유하지 않는(`share_weights=False`) Dual-Stream 구조를 채택했습니다. 
두 개의 독립적인 ResNet50 특성 추출기(Backbone)가 각각의 시점에서 2048차원의 Feature를 추출하고, 이를 병합(Concat)하여 최종 분류기(Classifier)를 통과시킵니다.

# 🚀 Key Engineering Points

1. Robust Training Pipeline
* 과적합 방지: RandomAffine, Perspective, ColorJitter 등 강도 높은 데이터 증강(Augmentation)과 `Dropout(0.5)` 적용
* 학습 안정화: `ReduceLROnPlateau` 스케줄러를 통한 동적 학습률 조정 및 `Early Stopping` 도입

2. Memory & Speed Optimization
* AMP (Automatic Mixed Precision): `torch.cuda.amp.autocast()` 및 `GradScaler`를 활용하여 GPU 메모리 사용량을 절반으로 줄이고 학습 속도 최적화

3. Safe Inference
* 데이터로더의 미세한 셔플링이나 멀티프로세싱으로 인해 제출 파일의 `id` 순서가 꼬이는 치명적인 버그를 방지하기 위해, 원본 `sample_submission.csv`를 기준으로 결과를 매핑하는 `pd.merge(how='left')` 방어 로직 구현

## 📁 Repository Structure
```text
├── dataset.py         # 커스텀 Dataset 클래스 및 Data Augmentation 적용
├── model.py           # Dual-View Predictor 모델 아키텍처 정의
├── train.py           # 모델 학습 루프, 검증 및 Best Model 저장 로직
├── inference.py       # 학습된 모델 기반 추론 및 최종 제출용 CSV 생성
└── README.md          # 프로젝트 설명서

