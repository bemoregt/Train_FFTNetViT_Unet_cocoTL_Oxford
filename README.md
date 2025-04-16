# FFTNetViT-UNet을 이용한 이미지 세그멘테이션 (COCO 전이학습)

이 프로젝트는 FFTNetViT(주파수 도메인 어텐션을 활용한 Vision Transformer)와 UNet 디코더를 결합하여 Oxford-IIIT Pet 데이터셋에서 반려동물 세그멘테이션을 수행하는 모델을 구현합니다. COCO 데이터셋으로 사전학습된 가중치를 활용하여 전이학습을 수행합니다.

## 주요 특징

- **FFTNetViT 인코더**: 이미지의 주파수 도메인에서 어텐션을 수행하는 Vision Transformer 기반 인코더
- **U-Net 디코더**: 고해상도 세그멘테이션 마스크 생성을 위한 디코더 구조
- **다중 손실 함수**: BCE, Dice, Boundary 손실의 조합으로 세그멘테이션 정확도 향상
- **COCO 전이학습**: COCO 데이터셋으로 사전학습된 가중치를 활용한 전이학습
- **적응형 주파수 필터링**: 각 어텐션 헤드마다 다른 주파수 대역을 강조하는 적응형 필터링

## 모델 구조

### FFTNetViT

FFTNetViT는 일반적인 Vision Transformer의 확장으로, 다음과 같은 특징이 있습니다:

1. **주파수 도메인 어텐션**: Self-attention을 주파수 도메인에서 수행하여 다양한 주파수 성분에 대한 어텐션 메커니즘을 적용
2. **적응형 스펙트럴 필터링**: 각 어텐션 헤드마다 다른 주파수 대역을 학습하여 강조/억제
3. **멀티헤드 구조**: 여러 어텐션 헤드를 통해 서로 다른 주파수 성분에 집중

### UNet 디코더

세그멘테이션을 위한 UNet 스타일 디코더는 다음 특징을 가집니다:

1. **점진적 업샘플링**: 저해상도 특징 맵에서 고해상도 예측으로 점진적으로 확장
2. **멀티스케일 특징 추출**: 다양한 수준의 특징을 활용하여 세부 정보와 문맥 정보를 결합
3. **스킵 커넥션**: 인코더와 디코더 사이의 연결을 통해 세부 정보 보존

## 데이터셋

Oxford-IIIT Pet 데이터셋:
- 37개 품종의 반려동물(고양이/개) 약 7,400장의 이미지
- 각 이미지에 해당하는 세그멘테이션 마스크(trimap)
- 배경, 경계, 전경으로 구분된 라벨링

## 손실 함수

이 프로젝트는 세 가지 손실 함수의 조합을 사용합니다:

1. **BCE Loss**: 픽셀 단위의 이진 분류를 위한 기본 손실 함수
2. **Dice Loss**: 클래스 불균형에 강인하며 영역 기반 유사도를 측정
3. **Boundary Loss**: 객체의 경계에 더 많은 가중치를 부여하여 경계 정확도 향상

## 설치 및 환경 설정

필요한 패키지:

```bash
pip install torch torchvision tqdm matplotlib numpy requests Pillow einops scipy
```

## 사용 방법

### 학습

```bash
python main.py --epochs 50 --batch-size 16 --lr 1e-4 --img-size 384 --pretrained
```

옵션:
- `--epochs`: 훈련 에폭 수
- `--batch-size`: 배치 크기
- `--lr`: 학습률
- `--img-size`: 입력 이미지 크기
- `--pretrained`: COCO 사전학습 가중치 사용 여부
- `--device`: 학습에 사용할 장치 (cuda/mps/cpu)
- `--save-dir`: 모델 저장 디렉토리
- `--results-dir`: 결과 저장 디렉토리

### 추론/테스트

```bash
# 추론 코드는 별도로 구현 필요
```

## 결과

학습 후 다음과 같은 결과 파일이 생성됩니다:

1. `models/fftnet_unet_pet_coco_pretrained_best.pth`: 최고 성능 모델 가중치
2. `results/training_curves.png`: 학습 및 검증 손실, IoU 그래프
3. `results/final_predictions.png`: 최종 예측 시각화
4. `results/fftnet_unet_pet_coco_pretrained_metrics.csv`: 에폭별 성능 지표

## 파일 구조

- `main.py`: 메인 학습 스크립트
- `fftnet_vit.py`: FFTNetViT 모델 구현
- `fftnet_modules.py`: 기본 모듈 및 어텐션 메커니즘 구현
- `train_fftnet_unet.py`: FFTNetViT-UNet 모델 구현
- `dataset.py`: 데이터셋 및 전처리 클래스
- `loss.py`: 손실 함수 구현
- `utils.py`: 유틸리티 함수 (훈련, 검증, 시각화 등)

## 참고 사항

- 학습에는 GPU 또는 Apple Silicon Mac의 MPS 가속이 권장됩니다.
- Oxford-IIIT Pet 데이터셋은 자동으로 다운로드 및 준비됩니다.
- 최적의 결과를 위해 이미지 크기와 배치 크기를 하드웨어에 맞게 조정하세요.

## 라이센스

MIT License