import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice 계수 손실 함수
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = targets.size(0)
        
        # 차원 통합
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Dice 계수 계산
        intersection = (probs * targets).sum(1)
        union = probs.sum(1) + targets.sum(1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

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

# 복합 손실 함수
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.5, boundary_weight=0.2):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, outputs, targets):
        bce_loss = self.bce_loss(outputs, targets.float())
        dc_loss = self.dice_loss(outputs, targets.float())
        bd_loss = self.boundary_loss(outputs, targets)
        
        # 손실 함수 가중치 조정
        return self.bce_weight * bce_loss + self.dice_weight * dc_loss + self.boundary_weight * bd_loss