# 세그멘테이션 데이터셋 클래스
import os
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class PetSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set="trainval", transform=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transform = transform
        
        # 이미지 및 마스크 파일 목록 로드
        with open(os.path.join(root_dir, "annotations", f"{image_set}.txt"), "r") as f:
            self.file_list = [line.strip().split()[0] for line in f.readlines()]
        
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
        
        # 이미지 및 마스크 로드
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # 데이터 변환 적용
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # 마스크를 3개 클래스(배경, 경계, 전경)에서 1개 채널(전경/배경)으로 변환
        mask = (mask > 1).float()  # 1은 배경, 2는 경계, 3은 전경
        
        return image, mask

# 데이터 증강 및 전처리
class SegmentationTransforms:
    def __init__(self, img_size=256):
        self.img_size = img_size
    
    def __call__(self, image, mask):
        # 리사이징
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # 무작위 수평 뒤집기
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 무작위 회전
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # 색상 지터링 (Color Jittering)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)
        
        # 가우시안 블러
        if random.random() > 0.7:
            from PIL import ImageFilter
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        
        # 무작위 확대/축소
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = int(self.img_size * scale)
            image = image.resize((new_size, new_size), Image.BILINEAR)
            mask = mask.resize((new_size, new_size), Image.NEAREST)
            
            # 중앙 크롭으로 원래 크기로 복원
            if new_size > self.img_size:
                left = (new_size - self.img_size) // 2
                top = (new_size - self.img_size) // 2
                right = left + self.img_size
                bottom = top + self.img_size
                image = image.crop((left, top, right, bottom))
                mask = mask.crop((left, top, right, bottom))
            else:
                # 패딩으로 원래 크기로 복원
                padded_image = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
                padded_mask = Image.new("L", (self.img_size, self.img_size), 0)
                paste_x = (self.img_size - new_size) // 2
                paste_y = (self.img_size - new_size) // 2
                padded_image.paste(image, (paste_x, paste_y))
                padded_mask.paste(mask, (paste_x, paste_y))
                image = padded_image
                mask = padded_mask
        
        # 텐서로 변환
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long().unsqueeze(0)
        
        # 정규화
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask