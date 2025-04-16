import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# IoU 계산 함수
def calculate_iou(pred, target):
    """
    세그멘테이션 마스크의 IoU(Intersection over Union) 계산
    Args:
        pred: 예측 마스크 (배치 형태)
        target: 실제 마스크 (배치 형태)
    Returns:
        배치의 각 이미지에 대한 IoU 값
    """
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3)) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# 예측 시각화 함수
def visualize_predictions(model, val_loader, device, num_samples=4, save_path="results/segmentation_predictions.png"):
    """
    모델의 세그멘테이션 예측 결과를 시각화
    Args:
        model: 학습된 모델
        val_loader: 검증 데이터 로더
        device: 사용할 장치 (CPU/GPU)
        num_samples: 시각화할 샘플 수
        save_path: 결과 이미지 저장 경로
    """
    model.eval()
    images, masks, preds = [], [], []
    
    with torch.no_grad():
        for img, mask in val_loader:
            if len(images) >= num_samples:
                break
            
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred = torch.sigmoid(pred) > 0.5
            
            # 시각화를 위해 CPU로 이동 및 변환
            images.extend(img.cpu())
            masks.extend(mask.cpu())
            preds.extend(pred.cpu())
            
            if len(images) >= num_samples:
                break
    
    # 결과 시각화
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # 이미지 표시 (정규화 해제)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # 원본 이미지
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")
        
        # 실제 마스크
        axes[i, 1].imshow(masks[i].squeeze().numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        # 예측 마스크
        axes[i, 2].imshow(preds[i].squeeze().numpy(), cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 학습 및 검증 지표 시각화 함수
def plot_training_curves(train_losses, val_losses, val_ious, save_path="results/training_curves.png"):
    """
    학습 및 검증 지표 그래프 생성
    Args:
        train_losses: 에폭별 학습 손실 리스트
        val_losses: 에폭별 검증 손실 리스트
        val_ious: 에폭별 검증 IoU 리스트
        save_path: 그래프 이미지 저장 경로
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # IoU 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_ious, 'g-', label='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 훈련 함수
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 에폭의 훈련을 수행
    Args:
        model: 학습할 모델
        train_loader: 훈련 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 학습에 사용할 장치 (CPU/GPU)
    Returns:
        에폭 평균 손실
    """
    model.train()
    running_loss = 0.0
    
    train_loader_tqdm = tqdm(train_loader, desc="Training")
    for images, masks in train_loader_tqdm:
        images, masks = images.to(device), masks.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item() * images.size(0)
        
        # tqdm 진행 상태 업데이트
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    # 에폭 평균 손실 계산
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return epoch_loss

# 검증 함수
def validate(model, val_loader, criterion, device):
    """
    검증 데이터셋에 대한 모델 성능 평가
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 사용할 장치 (CPU/GPU)
    Returns:
        평균 검증 손실, 평균 IoU
    """
    model.eval()
    running_loss = 0.0
    ious = []
    
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        for images, masks in val_loader_tqdm:
            images, masks = images.to(device), masks.to(device)
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # IoU 계산
            batch_iou = calculate_iou(outputs, masks)
            ious.extend(batch_iou.cpu().numpy())
            
            # 통계 업데이트
            running_loss += loss.item() * images.size(0)
            
            # tqdm 진행 상태 업데이트
            val_loader_tqdm.set_postfix(loss=loss.item())
    
    # 에폭 평균 손실 및 IoU 계산
    epoch_loss = running_loss / len(val_loader.dataset)
    mean_iou = np.mean(ious)
    
    return epoch_loss, mean_iou