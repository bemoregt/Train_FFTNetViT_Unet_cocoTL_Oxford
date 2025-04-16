import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from fftnet_modules import TransformerEncoderBlock, PatchEmbed

class FFTNetViT(nn.Module):
    """
    FFTNetViT 모델 구현
    주파수 도메인에서 작동하는 스펙트럴 어텐션이 적용된 Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 adaptive_spectral=True, return_intermediate=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.return_intermediate = return_intermediate
        
        # 패치 임베딩 레이어
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 클래스 토큰 및 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 트랜스포머 블록들
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, act_layer=nn.GELU, 
                norm_layer=norm_layer, adaptive_spectral=adaptive_spectral
            )
            for i in range(depth)
        ])
        
        # 최종 정규화 및 분류 헤드
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 가중치 초기화
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        # 패치 임베딩
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 위치 인코딩 추가
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 중간 특징맵을 저장할 리스트
        intermediate_features = []
        
        # 트랜스포머 블록 통과
        for block in self.blocks:
            x = block(x)
            if self.return_intermediate:
                intermediate_features.append(x)
        
        # 최종 정규화
        x = self.norm(x)
        
        if self.return_intermediate:
            return intermediate_features
        return x[:, 0]  # 클래스 토큰만 반환
        
    def forward(self, x):
        x = self.forward_features(x)
        if not self.return_intermediate:
            x = self.head(x)
        return x

def fftnet_vit_base(img_size=224, patch_size=16, in_chans=3, num_classes=1000, **kwargs):
    """
    FFTNetViT-Base 모델 (ViT-Base와 유사한 아키텍처)
    """
    model = FFTNetViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs
    )
    return model

def fftnet_vit_large(img_size=224, patch_size=16, in_chans=3, num_classes=1000, **kwargs):
    """
    FFTNetViT-Large 모델 (ViT-Large와 유사한 아키텍처)
    """
    model = FFTNetViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs
    )
    return model

def fftnet_vit_small(img_size=224, patch_size=16, in_chans=3, num_classes=1000, **kwargs):
    """
    FFTNetViT-Small 모델 (더 작은 리소스 요구)
    """
    model = FFTNetViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs
    )
    return model