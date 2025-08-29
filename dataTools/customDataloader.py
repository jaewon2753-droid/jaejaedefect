import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.badPixelGenerator import generate_bad_pixels
import os
import torch  # ✅ [해결] 이 줄을 추가하여 오류를 해결합니다.

class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        self.transformHRGT = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.transformRI = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # 1) GT 로드
        try:
            gt_image_pil = Image.open(self.image_list[i]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {self.image_list[i]}: {e}")
            return self.__getitem__((i + 1) % len(self.image_list))

        # 2) 불량 화소 생성 및 마스크 준비
        gt_image_np = np.array(gt_image_pil)
        
        # 원본과 복사본을 만들어 마스크 생성
        input_image_np = generate_bad_pixels(gt_image_np.copy())
        
        # 픽셀값이 다른 부분을 찾아 마스크 생성 (True/False)
        mask = (gt_image_np != input_image_np).any(axis=-1)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) # (1, H, W) 형태로 변환

        input_image_pil = Image.fromarray(input_image_np)

        # 3) 변환 적용
        input_tensor = self.transformRI(input_image_pil)
        gt_tensor = self.transformHRGT(gt_image_pil)

        # 4) 마스크 텐서를 추가로 반환
        return input_tensor, gt_tensor, mask_tensor