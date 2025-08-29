# utilities/inferenceUtils.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
from PIL import ImageFile
from dataTools.badPixelGenerator import generate_bad_pixels 

ImageFile.LOAD_TRUNCATED_IMAGES = True

# (AddGaussianNoise 클래스는 변경 없이 그대로 둡니다)
class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel

    def __call__(self, tensor):
        if self.noiseLevel == 0:
            return tensor
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)


class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, validation=None, inferenceMode=2):
        self.gridSize = gridSize
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.modelName = modelName
        self.validation = validation
        self.unNormalize = UnNormalize()
        self.inferenceMode = inferenceMode
        
        # ========================================================== #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 1 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ---------------------------------------------------------- #
        # 불량 화소가 추가된 입력 이미지를 저장할 경로를 미리 정의합니다.
        # ========================================================== #
        self.defectiveInputPath = os.path.join(self.outputRootDir, "defective_inputs")
        os.makedirs(self.defectiveInputPath, exist_ok=True)


    def inputForInference(self, imagePath, noiseLevel):
        source_image_pil = Image.open(imagePath).convert("RGB")

        if self.inferenceMode == 1:
            print("Mode 1: Treating input as already defective.")
            img = source_image_pil
            
        elif self.inferenceMode == 2:
            print("Mode 2: Applying bad pixels to clean input.")
            source_image_np = np.array(source_image_pil)
            corrupted_image_np = generate_bad_pixels(source_image_np)
            img = Image.fromarray(corrupted_image_np)

            # ========================================================== #
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 2 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
            # ---------------------------------------------------------- #
            # 모델에 입력되기 직전의, 불량 화소가 추가된 이미지를 파일로 저장합니다.
            # ========================================================== #
            datasetName = os.path.basename(os.path.dirname(imagePath))
            save_dir = os.path.join(self.defectiveInputPath, self.modelName, datasetName)
            os.makedirs(save_dir, exist_ok=True)
            
            defective_filename = f"{extractFileName(imagePath, True)}_defective_input.png"
            defective_save_path = os.path.join(save_dir, defective_filename)
            img.save(defective_save_path)
            
        else:
            raise ValueError(f"Invalid inferenceMode: {self.inferenceMode}")

        # 텐서 변환 및 정규화
        transform = transforms.Compose([
            transforms.ToTensor(),
            # 참고: MultiStream 모델은 dataloader에서 채널별로 정규화를 진행했으므로,
            # BJDD.py에서 추론 시에도 동일하게 채널 분리 후 정규화를 적용해야 합니다.
            # 여기서는 우선 텐서로만 변환합니다.
        ])
        
        # 정규화는 BJDD.py에서 채널 분리 후 직접 수행합니다.
        img_tensor = transform(img)
        normalized_tensor = (img_tensor * 2) - 1 # 0~1 범위를 -1~1 범위로 변경

        return normalized_tensor.unsqueeze(0)


    def saveModelOutput(self, modelOutput, inputImagePath, noiseLevel, step = None, ext = ".png"):
        datasetName = os.path.basename(os.path.dirname(inputImagePath))
        
        # 저장 경로를 결과(output) 폴더로 명확히 합니다.
        save_dir = os.path.join(self.outputRootDir, self.modelName, datasetName)
        os.makedirs(save_dir, exist_ok=True)

        if step:
            imageSavingPath = os.path.join(save_dir, f"{extractFileName(inputImagePath, True)}_corrected_sigma_{noiseLevel}_{self.modelName}_{step}{ext}")
        else:
            imageSavingPath = os.path.join(save_dir, f"{extractFileName(inputImagePath, True)}_corrected_sigma_{noiseLevel}_{self.modelName}{ext}")
        
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)

    # (testingSetProcessor 함수는 변경 없음)
    def testingSetProcessor(self):
        testSets = glob.glob(os.path.join(self.inputRootDir, '*/'))
        if not testSets: 
             testSets = [self.inputRootDir]
             
        if self.validation:
            testSets = testSets[:1]

        testImageList = []
        for t in testSets:
            testSetName = os.path.basename(os.path.normpath(t))
            # 결과(output) 폴더와 불량화소 입력(defective_input) 폴더를 모두 생성합니다.
            createDir(os.path.join(self.outputRootDir, self.modelName, testSetName))
            createDir(os.path.join(self.defectiveInputPath, self.modelName, testSetName))
            imgInTargetDir = imageList(t, False)
            testImageList.extend(imgInTargetDir)
            
        return testImageList
