# dataTools/customDataloader.py

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
import torch

class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        # 변환기는 텐서 변환만 수행하고, 정규화는 각 채널 분리 후 진행합니다.
        self.transform_to_tensor = transforms.ToTensor()

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
        input_image_np = generate_bad_pixels(gt_image_np.copy())
        mask = (gt_image_np != input_image_np).any(axis=-1)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        input_image_pil = Image.fromarray(input_image_np)

        # 3) 텐서로 변환
        input_tensor_3ch = self.transform_to_tensor(input_image_pil)
        gt_tensor_3ch = self.transform_to_tensor(gt_image_pil)

        # ========================================================== #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ---------------------------------------------------------- #
        # 4) R, G, B 채널을 분리하고 각각 정규화를 적용합니다.
        # ========================================================== #
        def normalize_channel(channel_tensor):
            # 각 채널(0~1)을 -1~1 범위로 정규화
            return (channel_tensor * 2) - 1

        # 입력 이미지 채널 분리
        r_plane = normalize_channel(input_tensor_3ch[0, :, :].unsqueeze(0))
        g_plane = normalize_channel(input_tensor_3ch[1, :, :].unsqueeze(0))
        b_plane = normalize_channel(input_tensor_3ch[2, :, :].unsqueeze(0))
        
        # 정규화된 3채널 텐서도 반환
        input_tensor_normalized = torch.cat([r_plane, g_plane, b_plane], dim=0)
        gt_tensor_normalized = normalize_channel(gt_tensor_3ch)

        # 5) 모델 학습에 필요한 모든 텐서를 반환합니다.
        return input_tensor_normalized, r_plane, g_plane, b_plane, gt_tensor_normalized, mask_tensor
