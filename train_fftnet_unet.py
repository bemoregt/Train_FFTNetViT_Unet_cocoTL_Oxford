import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import zipfile
import shutil
from PIL import Image
import random
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

# FFTNet 관련 모듈 임포트
from fftnet_modules import MultiHeadSpectralAttention, TransformerEncoderBlock
from fftnet_vit import FFTNetViT

# 디렉토리 생성
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 학습 파라미터 설정
num_epochs = 50
batch_size = 16
learning_rate = 1e-4
weight_decay = 1e-4
image_size = 384  # 세그멘테이션을 위해 더 큰 이미지 크기

# 장치 설정
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# 경계 인식 손실 함수 정의
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        # 예측값을 이진화
        pred = torch.sigmoid(pred)
        
        # Laplacian 커널을 사용하여 경계 감지
        laplacian_kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        # 타겟 마스크의 경계 추출
        target_boundary = F.conv2d(target.float(), laplacian_kernel, padding=1).abs()
        target_boundary = (target_boundary > 0.1).float()
        
        # 경계 영역 내의 가중치 지도 생성 (경계에 가까울수록 높은 가중치)
        dist_maps = self._compute_distance_weight_maps(target_boundary)
        
        # 경계 손실 계산
        weighted_bce = F.binary_cross_entropy(pred, target.float(), weight=dist_maps, reduction='mean')
        
        return weighted_bce
    
    def _compute_distance_weight_maps(self, boundary):
        # 경계에서의 거리를 계산하여 가중치 맵 생성
        # GPU에서 실행 가능한 방식으로 근사 구현
        weight_maps = torch.ones_like(boundary) * self.theta0
        
        # 경계 주변에 더 높은 가중치 부여
        kernel_size = 7  # 가중치 영역 크기
        dilated_boundary = boundary.clone()
        
        for i in range(1, kernel_size // 2 + 1):
            # 팽창 연산으로 경계 확장
            kernel = torch.ones(1, 1, 2*i+1, 2*i+1).to(boundary.device)
            dilated = F.conv2d(dilated_boundary, kernel, padding=i) > 0
            dilated = dilated.float()
            
            # 원래 경계와의 거리에 따라 가중치 감소
            weight = self.theta * max(0, 1 - i / (kernel_size // 2))
            weight_maps = torch.where(dilated > 0, weight_maps + weight, weight_maps)
        
        return weight_maps

# FFTNetViT 인코더와 UNet 디코더를 결합한 세그멘테이션 모델 정의
class FFTNetViTUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, num_classes=1,
                 embed_dim=512, depth=8, mlp_ratio=4.0, dropout=0.1, num_heads=8, 
                 adaptive_spectral=True, pretrained=False):
        super(FFTNetViTUNet, self).__init__()
        
        # FFTNetViT 인코더
        self.encoder = FFTNetViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_heads=num_heads,
            adaptive_spectral=adaptive_spectral,
            return_intermediate=True  # 중간 특징 맵을 반환하도록 수정 필요
        )
        
        # 패치 수 계산
        self.num_patches = (img_size // patch_size) ** 2
        self.latent_size = img_size // patch_size
        
        # 디코더 레이어
        self.decoder_channels = [embed_dim, 256, 128, 64, 32]
        
        # 디코더 업샘플링 레이어
        self.up_blocks = nn.ModuleList()
        for i in range(len(self.decoder_channels) - 1):
            self.up_blocks.append(
                UNetUpBlock(
                    in_channels=self.decoder_channels[i],
                    out_channels=self.decoder_channels[i+1],
                    scale_factor=2
                )
            )
        
        # 최종 출력 레이어
        self.final_conv = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)
        
        # 사전학습된 Mask R-CNN 모델에서 백본 가중치를 로드하는 옵션
        if pretrained:
            self.load_pretrained_weights()
            
    def load_pretrained_weights(self):
        print("Loading pretrained weights from COCO dataset...")
        # COCO 데이터셋으로 사전학습된 Mask R-CNN 모델 로드
        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        # 백본 부분은 ResNet50임
        pretrained_backbone = pretrained_model.backbone.body
        
        # 여기서는 간단한 예시로 첫 번째 컨볼루션 레이어와 첫 번째 트랜스포머 블록만 가중치 전이
        # 실제로는 구조가 다르기 때문에 더 정교한 매핑이 필요할 수 있음
        with torch.no_grad():
            # 패치 임베딩 레이어에 컨볼루션 가중치 전이
            if hasattr(self.encoder, 'patch_embed') and hasattr(self.encoder.patch_embed, 'proj'):
                # ResNet의 첫 번째 컨볼루션 레이어 가중치를 사용
                # 채널 수가 다를 수 있으므로 적절히 조정
                pretrained_conv = pretrained_backbone.conv1.weight
                
                # 패치 임베딩 프로젝션의 가중치 초기화
                # 채널 수와 커널 크기가 다를 수 있으므로 평균을 취하거나 보간하여 사용
                if self.encoder.patch_embed.proj.weight.shape[-1] == pretrained_conv.shape[-1]:
                    # 커널 크기가 같은 경우, 채널 수만 조정
                    self.encoder.patch_embed.proj.weight.data[:, :3, :, :] = pretrained_conv
                else:
                    # 커널 크기가 다른 경우, 보간 필요
                    # 간단한 예시로 새 가중치 초기화 시 표준편차를 조정
                    std = pretrained_conv.std().item()
                    self.encoder.patch_embed.proj.weight.data.normal_(mean=0.0, std=std)
                
        print("Pretrained weights loaded successfully.")
        
    def forward(self, x):
        # 인코더 통과 (중간 특징 맵 획득)
        features = self.encoder(x)
        
        # 마지막 특징 맵 추출
        # 1차원 출력(B, num_patches + 1, embed_dim)을 2D로 재구성
        # 첫 번째 토큰(CLS 토큰)은 제외
        batch_size = x.shape[0]
        latent = features[-1][:, 1:, :]  # CLS 토큰 제외
        latent = latent.reshape(batch_size, self.latent_size, self.latent_size, -1)
        latent = latent.permute(0, 3, 1, 2)  # B, C, H, W 형태로 변환
        
        # 디코더 통과
        x = latent
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # 최종 출력 레이어 통과
        out = self.final_conv(x)
        
        return out

# UNet 스타일 업샘플링 블록
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

# COCO 데이터셋 다운로드 및 준비 함수 (예시)
def download_coco_dataset():
    """
    이 함수는 COCO 데이터셋을 다운로드하여 준비합니다.
    실제 구현에서는 COCO API를 사용하는 것이 좋습니다.
    여기서는 간단한 예시만 제공합니다.
    """
    print("COCO dataset will be downloaded for pretraining.")
    # 실제 구현에서는 COCO API를 사용하여 데이터셋을 다운로드하고 준비합니다.
    # 이 예시에서는 Oxford-IIIT Pet 데이터셋을 그대로 사용합니다.
    return download_pet_dataset()

# Oxford-IIIT Pet Dataset 다운로드 및 준비
def download_pet_dataset():
    dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    dataset_path = "data/oxford-iiit-pet"
    os.makedirs(dataset_path, exist_ok=True)
    
    # 이미지 다운로드 및 압축 해제
    images_tar_path = os.path.join(dataset_path, "images.tar.gz")
    if not os.path.exists(images_tar_path):
        print("Downloading Oxford-IIIT Pet Dataset images...")
        response = requests.get(dataset_url, stream=True)
        with open(images_tar_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print("Extracting images...")
        import tarfile
        with tarfile.open(images_tar_path, "r:gz") as tar:
            tar.extractall(dataset_path)
    
    # 어노테이션 다운로드 및 압축 해제
    annotations_tar_path = os.path.join(dataset_path, "annotations.tar.gz")
    if not os.path.exists(annotations_tar_path):
        print("Downloading Oxford-IIIT Pet Dataset annotations...")
        response = requests.get(annotations_url, stream=True)
        with open(annotations_tar_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print("Extracting annotations...")
        import tarfile
        with tarfile.open(annotations_tar_path, "r:gz") as tar:
            tar.extractall(dataset_path)
    
    print("Oxford-IIIT Pet Dataset ready.")
    return dataset_path