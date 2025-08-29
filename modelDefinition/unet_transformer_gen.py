# modelDefinition/unet_transformer_gen.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicBlocks import DoubleConv, Down, Up, OutConv

# (TransformerBlock 클래스는 변경 없이 그대로 사용합니다)
class TransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, n_layers, patch_size=2, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.reshape_conv = nn.Conv2d(embed_dim, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_res = x
        
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).view(b, -1, h // self.patch_size, w // self.patch_size)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.reshape_conv(x)
        
        return x + x_res

# ========================================================== #
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 핵심 수정 사항 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
# ---------------------------------------------------------- #
# 1. R, G, B 각 채널을 독립적으로 처리하는 3개의 인코더(Stream)를 만듭니다.
# 2. 병목(Bottleneck) 구간에서 3개 스트림의 특징을 하나로 합칩니다(Fusion).
# 3. 합쳐진 특징을 Transformer와 디코더가 처리하여 최종 이미지를 복원합니다.
# ========================================================== #
class MultiStreamUNetTransformer(nn.Module):
    def __init__(self, n_classes=3):
        super(MultiStreamUNetTransformer, self).__init__()
        self.n_classes = n_classes

        # --- 3개의 독립적인 인코더 스트림 ---
        # 각 인코더는 1채널(R, G, or B) 이미지를 입력받습니다.
        # 경량화를 위해 채널 수를 기존의 절반(64->32, 128->64 등)으로 줄입니다.
        def create_encoder_stream():
            return nn.ModuleList([
                DoubleConv(1, 32),
                Down(32, 64),
                Down(64, 128),
                Down(128, 256)
            ])

        self.r_encoder = create_encoder_stream()
        self.g_encoder = create_encoder_stream()
        self.b_encoder = create_encoder_stream()

        # --- 트랜스포머 병목 ---
        # 3개 스트림에서 나온 256채널 특징 3개를 합치므로, 입력은 256 * 3 = 768 채널이 됩니다.
        self.transformer = TransformerBlock(channels=768, n_heads=8, n_layers=4, patch_size=2)
                                             
        # --- 하나의 디코더 ---
        # 디코더의 입력 채널 수도 융합된 특징에 맞게 조정합니다.
        self.up1 = Up(768, 256 * 3) # 입력 768 -> 출력 512 (스킵 연결을 위해 채널 수 유지)
        self.up2 = Up(256 * 3, 128 * 3)
        self.up3 = Up(128 * 3, 64 * 3)
        self.outc = OutConv(64 * 3, n_classes)

    def forward(self, r_plane, g_plane, b_plane):
        # 원본 입력을 저장 (최종 결과에 더해주기 위함)
        input_image = torch.cat([r_plane, g_plane, b_plane], dim=1)
        
        # --- 1. 각 채널별 특징 추출 ---
        # R 채널
        r1 = self.r_encoder[0](r_plane)
        r2 = self.r_encoder[1](r1)
        r3 = self.r_encoder[2](r2)
        r4 = self.r_encoder[3](r3)
        
        # G 채널
        g1 = self.g_encoder[0](g_plane)
        g2 = self.g_encoder[1](g1)
        g3 = self.g_encoder[2](g2)
        g4 = self.g_encoder[3](g3)

        # B 채널
        b1 = self.b_encoder[0](b_plane)
        b2 = self.b_encoder[1](b1)
        b3 = self.b_encoder[2](b2)
        b4 = self.b_encoder[3](b3)

        # --- 2. 특징 융합 (Fusion) ---
        # 병목 지점에서 3개 채널의 특징 맵을 채널 축으로 연결합니다.
        fused_bottleneck = torch.cat([r4, g4, b4], dim=1)
        
        # 트랜스포머 처리
        x_transformer = self.transformer(fused_bottleneck)
        
        # --- 3. 이미지 복원 ---
        # 디코더 경로 (스킵 커넥션도 융합된 형태로 연결)
        x = self.up1(x_transformer, torch.cat([r3, g3, b3], dim=1))
        x = self.up2(x, torch.cat([r2, g2, b2], dim=1))
        x = self.up3(x, torch.cat([r1, g1, b1], dim=1))
        logits = self.outc(x)
        
        # 최종 결과는 원본 이미지에 잔차(residual)를 더하는 형태
        return torch.tanh(logits + input_image)
