import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from torchsummary import summary
from ptflops import get_model_complexity_info

from utilities.torchUtils import *
from dataTools.customDataloader import *
from utilities.inferenceUtils import *
from utilities.aestheticUtils import *

from loss.colorLoss import ColorLoss
from loss.percetualLoss import regularizedFeatureLoss
from loss.pytorch_msssim import MSSSIM

from modelDefinition.unet_transformer_gen import MultiStreamUNetTransformer
from modelDefinition.attentionDis import attentiomDiscriminator

from torchvision.utils import save_image


class BJDD:
    def __init__(self, config):
        self.gtPath         = config['gtPath']
        self.targetPath     = config['targetPath']
        self.checkpointPath = config['checkpointPath']
        self.logPath        = config['logPath']
        self.testImagesPath = config['testImagePath']
        self.resultDir      = config['resultDir']
        self.modelName      = config['modelName']
        self.dataSamples    = config['dataSamples']
        self.batchSize      = int(config['batchSize'])
        self.imageH         = int(config['imageH'])
        self.imageW         = int(config['imageW'])
        self.inputC         = int(config['inputC'])
        self.outputC        = int(config['outputC'])
        self.totalEpoch     = int(config['epoch'])
        self.interval       = int(config['interval'])
        self.learningRate   = float(config['learningRate'])
        self.adamBeta1      = float(config['adamBeta1'])
        self.adamBeta2      = float(config['adamBeta2'])
        self.barLen         = int(config['barLen'])

        self.currentEpoch = 0
        self.startSteps  = 0
        self.totalSteps  = 0
        self.unNorm = UnNormalize()
        self.noiseSet = [0, 5, 10]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.generator     = MultiStreamUNetTransformer(n_classes=self.outputC).to(self.device)
        self.discriminator = attentiomDiscriminator().to(self.device)

        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.scheduleLR = None

    def customTrainLoader(self, overFitTest=False):
        targetImageList = imageList(self.targetPath)
        print("Trining Samples (GT):", self.targetPath, len(targetImageList))

        if overFitTest:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:int(self.dataSamples)]

        datasetReadder = customDatasetReader(image_list=targetImageList, imagePathGT=self.gtPath, height=self.imageH, width=self.imageW)
        self.trainLoader = torch.utils.data.DataLoader(dataset=datasetReadder, batch_size=self.batchSize, shuffle=True, num_workers=2)
        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples=None):
        if dataSamples:
            self.dataSamples = dataSamples

        featureLoss        = regularizedFeatureLoss().to(self.device)
        colorLoss          = ColorLoss().to(self.device)
        adversarialLoss    = nn.BCEWithLogitsLoss().to(self.device)
        ssimLoss    = MSSSIM(window_size=11, size_average=True, channel=self.outputC).to(self.device)
        lambda_ssim = 0.2

        trainingImageLoader = self.customTrainLoader(overFitTest=overFitTest)
        if resumeTraning:
            try:
                self.modelLoad()
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                customPrint(Fore.RED + "Starting training from scratch.", textWidth=self.barLen)

        customPrint('Training is about to begin using:' + Fore.YELLOW + f'[{self.device}]'.upper(), textWidth=self.barLen)
        self.totalSteps = int(len(trainingImageLoader) * self.totalEpoch)
        startTime = time.time()
        bar = ProgressBar(self.totalSteps, max_width=int(self.barLen / 2))
        currentStep = self.startSteps

        while currentStep < self.totalSteps:
            for i, (inputImages, r_plane, g_plane, b_plane, gtImages, mask_tensors) in enumerate(trainingImageLoader):
                currentStep += 1
                if currentStep > self.totalSteps:
                    break

                input_real = inputImages.to(self.device)
                r_plane_real = r_plane.to(self.device)
                g_plane_real = g_plane.to(self.device)
                b_plane_real = b_plane.to(self.device)
                gt_real    = gtImages.to(self.device)
                mask       = mask_tensors.to(self.device) # 마스크도 GPU로 이동

                B = input_real.shape[0]
                target_real_label = (torch.rand(B, 1) * 0.3 + 0.7).to(self.device)
                target_fake_label = (torch.rand(B, 1) * 0.3).to(self.device)
                target_ones_label = torch.ones(B, 1).to(self.device)

                # ====== 1) Update Discriminator ======
                self.optimizerD.zero_grad()
                generated_fake = self.generator(r_plane_real, g_plane_real, b_plane_real)
                lossD_real = adversarialLoss(self.discriminator(gt_real), target_real_label)
                lossD_fake = adversarialLoss(self.discriminator(generated_fake.detach()), target_fake_label)
                lossD = lossD_real + lossD_fake
                lossD.backward()
                self.optimizerD.step()

                # ====== 2) Update Generator ======
                self.optimizerG.zero_grad()
                generated_fake_for_g = self.generator(r_plane_real, g_plane_real, b_plane_real)

                weight = torch.ones_like(mask) * 2.0
                weight[mask == 1] = 1.0
                
                lossG_L1_weighted = (weight * torch.abs(generated_fake_for_g - gt_real)).mean()

                lossG_content = lossG_L1_weighted \
                              + featureLoss(generated_fake_for_g, gt_real) \
                              + colorLoss(generated_fake_for_g,  gt_real)

                ms_ssim_val = ssimLoss(generated_fake_for_g, gt_real)
                loss_ssim   = 1.0 - ms_ssim_val
                lossG_content = lossG_content + lambda_ssim * loss_ssim

                lossG_adv = adversarialLoss(self.discriminator(generated_fake_for_g), target_ones_label)
                lossG = lossG_content + 1e-3 * lossG_adv
                lossG.backward()
                self.optimizerG.step()

                if (currentStep + 1) % self.interval == 0:
                    summaryInfo = {
                        'Input Images':     self.unNorm(input_real),
                        'Generated Images': self.unNorm(generated_fake_for_g),
                        'GT Images':        self.unNorm(gt_real),
                        'Step':             currentStep + 1,
                        'Epoch':            self.currentEpoch,
                        'LossG':            float(lossG.item()),
                        'LossD':            float(lossD.item()),
                        'MS-SSIM':          float(ms_ssim_val.detach().mean().item()),
                        'Loss_SSIM':        float(loss_ssim.detach().mean().item()),
                        'Path':             self.logPath,
                    }
                    tbLogWritter(summaryInfo)
                    self.savingWeights(currentStep)

            self.currentEpoch += 1

        self.savingWeights(currentStep, duplicate=True)
        customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)

    def modelInference(self, testImagesPath=None, outputDir=None, resize=None, validation=None, noiseSet=None,
                       steps=None, inferenceMode=2):
        if not validation:
            self.modelLoad()
            print("\nInferencing on pretrained weights.")

        finalNoiseSet = self.noiseSet if noiseSet is None else [int(n) for n in noiseSet.split(',')]
        if testImagesPath: self.testImagesPath = testImagesPath
        if outputDir: self.resultDir = outputDir

        modelInference = inference(gridSize=0, inputRootDir=self.testImagesPath, outputRootDir=self.resultDir, modelName=self.modelName, validation=validation, inferenceMode=inferenceMode)
        testImageList = modelInference.testingSetProcessor()
        with torch.no_grad():
            for noise in finalNoiseSet:
                for imgPath in testImageList:
                    img_3ch = modelInference.inputForInference(imgPath, noiseLevel=noise).to(self.device)
                    
                    r_plane = img_3ch[:, 0, :, :].unsqueeze(1)
                    g_plane = img_3ch[:, 1, :, :].unsqueeze(1)
                    b_plane = img_3ch[:, 2, :, :].unsqueeze(1)

                    out = self.generator(r_plane, g_plane, b_plane)
                    modelInference.saveModelOutput(out, imgPath, noise, steps)
        print("\nInference completed!")

    def modelSummary(self, input_size=None):
        if not input_size:
            input_size = (1, self.imageH, self.imageW)

        customPrint(Fore.YELLOW + "Generator (Multi-Stream UNet Transformer)", textWidth=self.barLen)
        print(self.generator)
        total_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        print(f"Generator Trainable Parameters: {total_params:,}")
        print("*" * self.barLen); print()

        customPrint(Fore.YELLOW + "Discriminator", textWidth=self.barLen)
        summary(self.discriminator, input_size=(self.inputC, self.imageH, self.imageW))
        print("*" * self.barLen); print()
        try:
            configShower()
        except Exception:
            pass

    def savingWeights(self, currentStep, duplicate=None):
        checkpoint = {
            'step':       currentStep + 1,
            'stateDictG': self.generator.state_dict(),
            'stateDictD': self.discriminator.state_dict(),
            'optimizerG': self.optimizerG.state_dict(),
            'optimizerD': self.optimizerD.state_dict(),
        }
        saveCheckpoint(modelStates=checkpoint, path=self.checkpointPath, modelName=self.modelName)
        if duplicate:
            saveCheckpoint(modelStates=checkpoint, path=self.checkpointPath + "backup_" + str(currentStep) + "/",
                           modelName=self.modelName, backup=None)

    def modelLoad(self):
        customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)
        previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)
        self.generator.load_state_dict(previousWeight['stateDictG'])
        self.discriminator.load_state_dict(previousWeight['stateDictD'])
        self.optimizerG.load_state_dict(previousWeight['optimizerG'])
        self.optimizerD.load_state_dict(previousWeight['optimizerD'])
        self.startSteps = int(previousWeight['step'])
        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)

