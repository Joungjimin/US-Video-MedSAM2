import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalContextExchange(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        
        # 3D Depthwise Conv: 시간축(T)을 가로지르는 필터
        self.depthwise_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False
        )
        
        self.pointwise = nn.Conv3d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Channel Attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(channels // 16, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(channels // 16, 8), channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, num_frames):
        """
        x: [B*T, C, H, W] (MedSAM 2의 일반적인 특징 맵 형태)
        num_frames: 비디오 한 개당 프레임 수 (T)
        """
        BT, C, H, W = x.shape
        B = BT // num_frames
        T = num_frames
        
        # 1. 5D로 변환: [B, C, T, H, W]
        identity = x
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        
        # 2. Temporal Fusion 연산
        out = self.depthwise_conv(x)
        out = self.bn1(out)
        attn = self.attention(out)
        out = out * attn
        out = self.pointwise(out)
        out = self.bn2(out)
        
        # 3. 다시 4D로 변환하여 잔차 연결: [B*T, C, H, W]
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(BT, C, H, W)
        return identity + self.alpha * out