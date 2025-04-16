import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class MultiHeadSpectralAttention(nn.Module):
    """
    Multi-Head Spectral Attention 모듈
    입력 토큰을 주파수 도메인으로 변환하여 어텐션 수행
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., adaptive=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 적응형 주파수 필터링을 위한 파라미터
        self.adaptive = adaptive
        if adaptive:
            self.freq_params = nn.Parameter(torch.ones(num_heads, 1) * 0.5)
        
    def forward(self, x):
        B, N, C = x.shape  # 배치 크기, 토큰 수, 채널 수
        
        # QKV 계산
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]
        
        # 주파수 도메인으로 변환 (FFT)
        q_freq = torch.fft.fft(q, dim=2)
        k_freq = torch.fft.fft(k, dim=2)
        
        # 적응형 주파수 필터링
        if self.adaptive:
            # 주파수 마스킹(강조할 저주파 영역 비율 설정)
            mask_size = int(N * torch.sigmoid(self.freq_params))
            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
            head_idx = torch.arange(self.num_heads, device=x.device).view(1, self.num_heads, 1, 1)
            
            # 각 헤드별로 다른 주파수 마스크 적용
            for h in range(self.num_heads):
                cutoff = mask_size[h]
                high_indices = torch.arange(cutoff, N, device=x.device)
                # 고주파 성분을 감쇠시킴 (저주파 통과 필터 효과)
                attenuation = torch.exp(-torch.arange(N-cutoff, device=x.device) / (N/4))
                q_freq[:, h, cutoff:, :] *= attenuation.view(1, -1, 1)
                k_freq[:, h, cutoff:, :] *= attenuation.view(1, -1, 1)
        
        # 역변환 (역 FFT)
        q_filtered = torch.fft.ifft(q_freq, dim=2).real
        k_filtered = torch.fft.ifft(k_freq, dim=2).real
        
        # 어텐션 계산
        attn = (q_filtered @ k_filtered.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 값 가중치 계산 및 출력
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out

class Mlp(nn.Module):
    """
    MLP 모듈
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """
    트랜스포머 인코더 블록
    Multi-Head Spectral Attention + MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, adaptive_spectral=True):
        super().__init__()
        
        # 정규화 레이어
        self.norm1 = norm_layer(dim)
        
        # 스펙트럴 어텐션 레이어
        self.attn = MultiHeadSpectralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, adaptive=adaptive_spectral
        )
        
        # 두 번째 정규화
        self.norm2 = norm_layer(dim)
        
        # MLP 레이어
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop
        )
        
    def forward(self, x):
        # 어텐션 레이어
        x = x + self.attn(self.norm1(x))
        
        # MLP 레이어
        x = x + self.mlp(self.norm2(x))
        
        return x

class PatchEmbed(nn.Module):
    """
    이미지를 패치로 분할하고 임베딩하는 레이어
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # 컨볼루션을 사용하여 패치 임베딩
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 패치 임베딩 계산
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, P, E
        
        return x