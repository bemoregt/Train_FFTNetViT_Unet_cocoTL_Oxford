import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from fftnet_modules import MultiHeadSpectralAttention, TransformerEncoderBlock
from fftnet_vit import FFTNetViT
from train_fftnet_unet import FFTNetViTUNet, download_pet_dataset, UNetUpBlock
from dataset import PetSegmentationDataset, SegmentationTransforms
from loss import CombinedLoss
from utils import train_epoch, validate, visualize_predictions, plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description='FFTNetViT-UNet 세그멘테이션 모델 학습')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에폭 수')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--img-size', type=int, default=384, help='입력 이미지 크기')
    parser.add_argument('--pretrained', action='store_true', help='사전학습된 가중치 사용')
    parser.add_argument('--device', type=str, default='', help='사용할 장치 (cuda/mps/cpu)')
    parser.add_argument('--save-dir', type=str, default='models', help='모델 저장 디렉토리')
    parser.add_argument('--results-dir', type=str, default='results', help='결과 저장 디렉토리')
    
    return parser.parse_args()

def main():
    # 인자 파싱
    args = parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # 장치 설정
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Oxford-IIIT Pet Dataset 다운로드 및 준비
    dataset_path = download_pet_dataset()
    
    # 데이터셋 생성
    transform = SegmentationTransforms(img_size=args.img_size)
    
    train_dataset = PetSegmentationDataset(
        root_dir=dataset_path,
        image_set="trainval",
        transform=transform
    )
    
    # 데이터셋을 훈련과 검증으로 분할 (80:20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 모델 초기화 (사전학습된 가중치 활용)
    model = FFTNetViTUNet(
        img_size=args.img_size,
        patch_size=16,
        in_chans=3,
        num_classes=1,
        embed_dim=384,
        depth=16,
        mlp_ratio=4.0,
        dropout=0.1,
        num_heads=16,
        adaptive_spectral=True,
        pretrained=args.pretrained  # 사전학습된 가중치 사용
    )
    
    # 모델을 선택한 장치로 이동
    model = model.to(device)
    
    # 손실 함수 정의
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.5, boundary_weight=0.2)
    
    # 전이학습을 위한 파라미터 그룹 설정
    # 인코더(사전학습된 부분)는 학습률을 낮게, 디코더(새로 추가된 부분)는 높게 설정
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.up_blocks.parameters()) + list(model.final_conv.parameters())
    
    # 옵티마이저 정의 (층별 학습률 설정)
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},  # 인코더(사전학습된 부분)는 낮은 학습률
        {'params': decoder_params, 'lr': args.lr}         # 디코더(새로 추가된 부분)는 높은 학습률
    ], weight_decay=1e-4)
    
    # 학습률 스케줄러 정의
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 및 검증 손실, IoU를 저장할 리스트
    train_losses = []
    val_losses = []
    val_ious = []
    
    # 최고 검증 IoU와 해당 에폭 저장
    best_val_iou = 0.0
    best_epoch = 0
    
    # 학습 루프
    print("Starting training for FFTNet-UNet on Oxford-IIIT Pet Dataset with COCO pretrained weights...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 학습
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 검증
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # 학습률 업데이트
        scheduler.step(val_loss)
        
        # 결과 출력
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # 최고 성능 모델 저장
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'{args.save_dir}/fftnet_unet_pet_coco_pretrained_best.pth')
            print(f"New best model saved with IoU: {val_iou:.4f}")
        
        # 현재 에폭 모델 저장 (주기적으로 저장하기 위해 10 에폭마다)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{args.save_dir}/fftnet_unet_pet_coco_pretrained_epoch_{epoch+1}.pth')
            # 현재 예측 결과 시각화
            visualize_predictions(model, val_loader, device, save_path=f'{args.results_dir}/pred_epoch_{epoch+1}.png')
            
        # 학습 지표를 파일에 저장
        with open(f'{args.results_dir}/fftnet_unet_pet_coco_pretrained_metrics.csv', 'a') as f:
            if epoch == 0:
                f.write('Epoch,Train Loss,Val Loss,Val IoU\n')
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f},{val_iou:.4f}\n")

    # 최종 모델 저장
    torch.save(model.state_dict(), f'{args.save_dir}/fftnet_unet_pet_coco_pretrained_final.pth')
    print(f"Training completed. Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")

    # 최종 예측 결과 시각화
    visualize_predictions(model, val_loader, device, num_samples=8, 
                         save_path=f'{args.results_dir}/final_predictions.png')

    # 학습 결과 시각화
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        val_ious=val_ious,
        save_path=f'{args.results_dir}/training_curves.png'
    )

if __name__ == '__main__':
    main()