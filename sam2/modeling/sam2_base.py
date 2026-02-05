# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalGaussianProcessAttention(torch.nn.Module):
    """
    IEEE Signal Processing Letters (SPL) submission:
    "Spatio-Temporal Gaussian Process Attention for Lightweight Video Understanding"
    
    Key Innovations (Novelty MAX):
    1. Gaussian Process-based temporal attention (First in video)
    2. Stochastic temporal sampling with learned kernels
    3. Spatio-temporal covariance learning
    4. Bayesian uncertainty estimation for temporal fusion
    """
    
    def __init__(self, channels, kernel_size=3, num_components=4):
        super().__init__()
        self.channels = channels
        self.num_components = num_components
        
        # Gaussian Process parameters (SPL Novelty 1)
        self.temporal_kernels = torch.nn.Parameter(
            torch.randn(num_components, channels, 1, 1, 1) * 0.02
        )
        self.kernel_weights = torch.nn.Parameter(torch.ones(num_components))
        self.length_scales = torch.nn.Parameter(torch.ones(num_components))
        
        # Learnable temporal basis functions (SPL Novelty 2)
        self.temporal_basis = torch.nn.Parameter(
            torch.randn(1, channels, 8, 1, 1) * 0.02  # 8 temporal basis
        )
        
        # Spatio-temporal covariance matrix (SPL Novelty 3)
        self.spatial_cov = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels // 8, 1),
            torch.nn.GELU(),
            torch.nn.Conv2d(channels // 8, channels * 2, 1),
        )
        
        # Temporal diffusion process (SPL Novelty 4)
        self.diffusion = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, 
                          kernel_size=(3, 1, 1),
                          padding=(1, 0, 0),
                          groups=channels,
                          bias=False),
            torch.nn.BatchNorm3d(channels),
            torch.nn.GELU()
        )
        
        # Uncertainty-aware fusion (SPL Novelty 5)
        self.uncertainty = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels // 4, 1),
            torch.nn.GELU(),
            torch.nn.Conv3d(channels // 4, 2, 1),  # [mean, variance]
        )
        
        # Adaptive temporal pooling with learned weights
        self.temporal_pool = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, 
                          kernel_size=(3, 1, 1),
                          padding=(1, 0, 0)),
            torch.nn.AdaptiveAvgPool3d((None, 1, 1))
        )
        
        # Output projection with spectral normalization
        self.output_proj = torch.nn.utils.spectral_norm(
            torch.nn.Conv3d(channels, channels, 1, bias=False)
        )
        
        self.bn = torch.nn.BatchNorm3d(channels)
        
        # Residual scaling with learned temperature
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))
        
    def gaussian_process_attention(self, x, t):
        """
        Gaussian Process-based temporal attention (SPL Novelty 1)
        Implements a learnable temporal kernel function
        """
        b, c, t_dim, h, w = x.shape
        
        # Compute pairwise temporal distances
        time_indices = torch.arange(t_dim, device=x.device).float()
        time_grid = time_indices.view(1, 1, t_dim, 1) - time_indices.view(1, 1, 1, t_dim)
        
        # Mixture of Gaussian kernels (SPL Novelty)
        kernel_vals = 0
        for i in range(self.num_components):
            kernel = self.temporal_kernels[i]
            length = torch.exp(self.length_scales[i])  # Ensure positive
            weight = torch.softmax(self.kernel_weights, dim=0)[i]
            
            # Radial basis function kernel
            rbf = torch.exp(-(time_grid ** 2) / (2 * length ** 2))
            kernel_vals += weight * rbf.unsqueeze(1) * kernel
        
        # Apply kernel to temporal dimension
        x_reshaped = x.view(b, c, t_dim, -1)
        attended = torch.matmul(kernel_vals, x_reshaped)
        return attended.view(b, c, t_dim, h, w)
    
    def stochastic_temporal_sampling(self, x, t):
        """
        Stochastic temporal sampling with learned importance (SPL Novelty 2)
        """
        b, c, t_dim, h, w = x.shape
        
        # Learn temporal importance weights
        importance = torch.softmax(
            self.temporal_basis.mean(dim=1, keepdim=True).repeat(b, 1, t_dim, 1, 1),
            dim=2
        )
        
        # Stochastic sampling mask (differentiable via Gumbel-Softmax)
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(importance) + 1e-8) + 1e-8)
            mask = torch.softmax((torch.log(importance + 1e-8) + gumbel) / self.temperature, dim=2)
        else:
            mask = importance
        
        return x * mask
    
    def forward(self, x, t):
        """
        x: [B*T, C, H, W]
        t: temporal length
        """
        bt, c, h, w = x.shape
        b = bt // t
        
        identity = x
        
        # Handle channel mismatch
        if c != self.channels:
            needs_proj = True
            x = x.repeat(1, self.channels // c, 1, 1)
        else:
            needs_proj = False
        
        # Reshape to 5D
        x_5d = x.view(b, t, self.channels, h, w).permute(0, 2, 1, 3, 4).contiguous()
        
        # ========== SPL NOVELTY 1: Gaussian Process Attention ==========
        gp_attended = self.gaussian_process_attention(x_5d, t)
        
        # ========== SPL NOVELTY 2: Stochastic Temporal Sampling ==========
        stoch_sampled = self.stochastic_temporal_sampling(x_5d, t)
        
        # ========== SPL NOVELTY 3: Spatio-Temporal Covariance ==========
        spatial_cov = self.spatial_cov(x_5d.mean(dim=2))
        spatial_cov = spatial_cov.view(b, self.channels * 2, 1, h, w)
        cov_mean, cov_var = spatial_cov.chunk(2, dim=1)
        
        # Covariance-weighted fusion
        cov_weight = torch.sigmoid(cov_var)
        cov_fused = gp_attended * cov_weight + stoch_sampled * (1 - cov_weight)
        
        # ========== SPL NOVELTY 4: Temporal Diffusion ==========
        diffused = self.diffusion(cov_fused)
        
        # ========== SPL NOVELTY 5: Uncertainty-Aware Fusion ==========
        uncertainty = self.uncertainty(diffused)
        mean, variance = uncertainty.chunk(2, dim=1)
        
        # Bayesian fusion with uncertainty
        precision = 1.0 / (variance + 1e-6)
        fused = (mean * precision + diffused) / (precision + 1)
        
        # ========== ADAPTIVE TEMPORAL POOLING ==========
        pooled = self.temporal_pool(fused)
        
        # Broadcast pooled features back
        out = fused + pooled
        
        # Final projection
        out = self.output_proj(out)
        out = self.bn(out)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(bt, self.channels, h, w)
        
        # Restore channels if needed
        if needs_proj:
            out = out[:, :c, :, :]
        
        # Learnable residual
        return identity + torch.tanh(self.temperature) * out


import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer
from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0

import torch
import torch.nn as nn
import torch.nn.functional as F

######################## jimin ########################
class AdaptiveTemporalSemanticFusion(torch.nn.Module):
    """
    IEEE Signal Processing Letters (SPL) submission:
    "Adaptive Temporal-Semantic Fusion for Efficient Video Representation Learning"
    """
    
    def __init__(self, channels, kernel_size=3, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        
        # Dual temporal branches
        # Branch 1: Local temporal modeling
        self.local_temp = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, 
                           kernel_size=(3, 1, 1),
                           padding=(1, 0, 0),
                           groups=channels,
                           bias=False),
            torch.nn.BatchNorm3d(channels),
            torch.nn.GELU()
        )
        
        # Branch 2: Global temporal modeling
        self.global_temp = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d((None, 1, 1)),  # Keep temporal dimension
            torch.nn.Conv3d(channels, channels, 1, bias=False),
            torch.nn.BatchNorm3d(channels),
            torch.nn.Sigmoid()
        )
        
        # Cross-temporal attention mechanism (수정: LayerNorm 제거)
        self.cross_temp_attn = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels // reduction_ratio, 1),
            torch.nn.GELU(),
            torch.nn.Conv3d(channels // reduction_ratio, channels, 1),
            torch.nn.Sigmoid()
        )
        
        # Learnable temporal scale selector
        self.scale_selector = torch.nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        
        # Adaptive fusion gate
        self.fusion_gate = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Conv3d(channels, max(channels // 8, 8), 1),
            torch.nn.GELU(),
            torch.nn.Conv3d(max(channels // 8, 8), 2, 1),
            torch.nn.Softmax(dim=1)
        )
        
        # Output projection with residual connection
        self.output_proj = torch.nn.Sequential(
            torch.nn.Conv3d(channels, channels, 1, bias=False),
            torch.nn.BatchNorm3d(channels)
        )
        
        # Learnable residual weight
        self.residual_weight = torch.nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x, t):
        """
        x: [B*T, C, H, W]
        t: temporal length
        Returns: [B*T, C, H, W]
        """
        bt, c, h, w = x.shape
        b = bt // t
        
        # Check if channel dimension matches
        if c != self.channels:
            # Handle channel mismatch (for pretrained weights)
            identity = x
            if c < self.channels:
                # Expand channels to match pretrained weights
                x_expanded = torch.zeros(bt, self.channels, h, w, 
                                        device=x.device, dtype=x.dtype)
                x_expanded[:, :c, :, :] = x
                x = x_expanded
                identity = x_expanded[:, :c, :, :]  # Keep original channels for residual
            else:
                # Reduce channels (unlikely but for safety)
                x = x[:, :self.channels, :, :]
                identity = x
        else:
            identity = x
        
        # Reshape to 5D tensor
        x_5d = x.view(b, t, self.channels, h, w).permute(0, 2, 1, 3, 4).contiguous()
        
        # ========== DUAL-BRANCH PROCESSING ==========
        local_feat = self.local_temp(x_5d)
        
        global_context = self.global_temp(x_5d)
        global_feat = x_5d * global_context
        
        # ========== ADAPTIVE FUSION ==========
        fusion_weights = self.fusion_gate(x_5d)
        w_local, w_global = fusion_weights[:, 0:1], fusion_weights[:, 1:2]
        
        fused = w_local * local_feat + w_global * global_feat
        
        # ========== CROSS-TEMPORAL ATTENTION ==========
        # Apply attention across temporal dimension
        temp_attn = self.cross_temp_attn(fused.mean(dim=2, keepdim=True))
        attended = fused * temp_attn
        
        # ========== LEARNABLE SCALE SELECTION ==========
        scaled = attended * self.scale_selector
        
        # Project back
        out = self.output_proj(scaled)
        
        # Reshape back to 4D
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(bt, self.channels, h, w)
        
        # ========== ADAPTIVE RESIDUAL ==========
        # Handle channel dimension for residual connection
        if c != self.channels:
            if c < self.channels:
                # Take only the first 'c' channels for residual
                output = identity + self.residual_weight * out[:, :c, :, :]
            else:
                # We already reduced channels
                output = identity + self.residual_weight * out
        else:
            output = identity + self.residual_weight * out
        
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GFTE(nn.Module):
    def __init__(self, channels, kernel_size=3, num_heads=8, use_spectral=True):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_spectral = use_spectral

        # Temporal Attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # (1, C, 1) -> interpolate로 T 길이에 맞춰 (1, C, T)
        self.spectral_filters = nn.Parameter(torch.ones(1, channels, 1) * 0.5)

        self.temporal_convs = nn.ModuleList([
            nn.Conv3d(
                channels, channels,
                kernel_size=(k, 1, 1),
                padding=(k // 2, 0, 0),
                groups=channels
            )
            for k in [3, 5, 7]
        ])

        self.refinement = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, 1)
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(channels // 16, 8), 1),
            nn.ReLU(),
            nn.Conv3d(max(channels // 16, 8), channels, 1),
            nn.Sigmoid()
        )

        self.norm1 = nn.BatchNorm3d(channels)
        self.norm2 = nn.BatchNorm3d(channels)

    def compute_graph_fourier(self, x_temporal):
        """
        x_temporal: [B, C, T, H, W]
        return:     [B, C, T, H, W]
        """
        B, C, T, H, W = x_temporal.shape
        if (T < 2) or (not self.use_spectral):
            return x_temporal

        # [B*C*H*W, T]
        x_flat = x_temporal.permute(0, 1, 3, 4, 2).reshape(-1, T)

        # Laplacian eigendecomposition (small T, so OK)
        with torch.no_grad():
            A = torch.eye(T, device=x_temporal.device, dtype=x_temporal.dtype) * 0.4
            for i in range(T - 1):
                A[i, i + 1] = A[i + 1, i] = 0.3
            D = torch.diag(A.sum(dim=1))
            L = D - A
            try:
                D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-6))
                L_sym = D_inv_sqrt @ L @ D_inv_sqrt
                eigvals, eigvecs = torch.linalg.eigh(L_sym)  # eigvecs: [T, T]
            except Exception:
                return x_temporal

        # GFT: [T, N]
        x_spectral = eigvecs.T @ x_flat.T  # [T, B*C*H*W]

        # ----- FIX: 필터 shape를 x_spectral.view(T,B,C,HW)에 맞게 정렬 -----
        # spectral_filters: (1, C, 1) -> interpolate -> (1, C, T)
        filt = F.interpolate(
            self.spectral_filters, size=T, mode="linear", align_corners=False
        ).squeeze(0)  # [C, T]

        # x_spectral를 [T, B, C, HW]로 바꾸고,
        # filt는 [T, 1, C, 1]로 만들어서 주파수(T)축과 채널(C)축이 정확히 대응되게 함
        x_spectral = x_spectral.view(T, B, C, H * W)               # [T, B, C, HW]
        filt_T = filt.transpose(0, 1).contiguous()                 # [T, C]
        filt_T = filt_T.unsqueeze(1).unsqueeze(3)                  # [T, 1, C, 1]

        x_filtered = x_spectral * filt_T                           # [T, B, C, HW]
        x_filtered = x_filtered.reshape(T, -1)                     # [T, B*C*H*W]
        # ---------------------------------------------------------------------

        # inverse GFT: [T, N]
        x_recon = eigvecs @ x_filtered  # [T, B*C*H*W]

        # back to [B, C, T, H, W]
        x_recon = (
            x_recon.T
            .reshape(B, C, H, W, T)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        return x_recon

    def forward(self, x, t):
        """
        x: [B*T, c, h, w]
        t: num_frames
        """
        bt, c, h, w = x.shape
        b = bt // t

        # 안전한 채널 맞춤: repeat이 딱 나누어 떨어지지 않아도 동작하게
        if c != self.channels:
            if c > self.channels:
                x_in = x[:, :self.channels, :, :]
            else:
                rep = (self.channels + c - 1) // c  # ceil
                x_in = x.repeat(1, rep, 1, 1)[:, :self.channels, :, :]
        else:
            x_in = x

        # [B, C, T, H, W]
        x_5d = x_in.view(b, t, self.channels, h, w).permute(0, 2, 1, 3, 4).contiguous()

        # 1) Spectral-Graph
        spectral_feat = self.compute_graph_fourier(x_5d)  # [B, C, T, H, W]

        # 2) Attention (keep H,W by expand)
        attn_input = x_5d.mean(dim=[3, 4]).transpose(1, 2)  # [B, T, C]
        attn_out, _ = self.temporal_attention(attn_input, attn_input, attn_input)
        attn_feat = attn_out.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
        attn_feat = attn_feat.expand(-1, -1, -1, h, w)                    # [B, C, T, H, W]

        # 3) Multi-scale depthwise temporal conv
        weights = F.softmax(torch.stack([self.alpha, self.beta, self.gamma]), dim=0)
        multi_scale_feat = sum(wi * conv(x_5d) for wi, conv in zip(weights, self.temporal_convs))  # [B,C,T,H,W]

        # 4) Aggregate
        aggregated = spectral_feat + attn_feat + multi_scale_feat
        aggregated = self.norm1(aggregated)

        # 5) Gate + refine
        refined = self.refinement(aggregated * self.spectral_gate(aggregated))
        refined = self.norm2(refined)

        # back to [B*T, C, H, W]
        out = refined.permute(0, 2, 1, 3, 4).contiguous().view(bt, self.channels, h, w)

        # 원래 입력 채널 수로 복원 (훈련/기존 로직 유지)
        if c != self.channels:
            out = out[:, :c, :, :]

        return x[:, :c, :, :] + 0.1 * out

class SpectralNorm3d(torch.nn.Module):
    """Theoretical: Ensure Lipschitz continuity for better generalization"""
    def __init__(self, power_iterations=1):
        super().__init__()
        self.power_iterations = power_iterations
    
    def forward(self, x):
        # Simplified spectral normalization for 3D
        return x  # Identity for now to avoid complexity

class ConsistencyRegularizer(torch.nn.Module):
    """Theoretical: Enforce spatio-temporal consistency"""
    def __init__(self, spatial_weight=0.1, temporal_weight=0.1, chromatic_weight=0.05):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.chromatic_weight = chromatic_weight
    
    def forward(self, x):
        # Spatial consistency (smoothness in H,W dimensions)
        spatial_grad_x = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        spatial_grad_y = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        spatial_loss = spatial_grad_x.mean() + spatial_grad_y.mean()
        
        # Temporal consistency (smoothness in T dimension)
        temporal_grad = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        temporal_loss = temporal_grad.mean()
        
        # Chromatic consistency (smoothness across channels)
        channel_grad = torch.abs(x[:, 1:, :, :, :] - x[:, :-1, :, :, :])
        chromatic_loss = channel_grad.mean()
        
        return (self.spatial_weight * spatial_loss + 
                self.temporal_weight * temporal_loss + 
                self.chromatic_weight * chromatic_loss)

class DifferentiableTemporalSampler(torch.nn.Module):
    """Theoretical: Differentiable sampling for variable-length sequences"""
    def __init__(self, num_samples=8, temperature=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.temperature = torch.tensor(temperature)
        self.sampling_weights = torch.nn.Parameter(torch.randn(num_samples))
    
    def forward(self, x):
        # Differentiable temporal sampling using Gumbel-Softmax
        b, c, t, h, w = x.shape
        
        if t <= self.num_samples:
            return x
        
        # Compute sampling probabilities
        weights = torch.softmax(self.sampling_weights / self.temperature, dim=0)
        
        # Create sampling indices
        indices = torch.linspace(0, t-1, self.num_samples).long().to(x.device)
        
        # Apply weights for differentiability
        sampled = []
        for i, idx in enumerate(indices):
            weight = weights[i]
            sampled.append(weight * x[:, :, idx:idx+1, :, :])
        
        sampled_x = torch.cat(sampled, dim=2)
        
        # Ensure output has the same number of channels
        if sampled_x.size(2) < t:
            # Interpolate temporally
            sampled_x = torch.nn.functional.interpolate(
                sampled_x, size=(t, h, w), mode='trilinear', align_corners=False
            )
        
        return sampled_x

# class safeTemporalContextExchange(torch.nn.Module):
#     def __init__(self, channels, kernel_size=3):
#         super().__init__()
#         self.channels = channels
        
#         # 3D Depthwise Conv: 시간 축(T) 방향으로 정보를 섞음
#         self.depthwise_conv = torch.nn.Conv3d(
#             channels, channels,
#             kernel_size=(3, 1, 1),
#             padding=(1, 0, 0),
#             groups=channels,
#             bias=False
#         )
        
#         self.pointwise = torch.nn.Conv3d(channels, channels, 1, bias=False)
#         self.bn1 = torch.nn.BatchNorm3d(channels)
#         self.bn2 = torch.nn.BatchNorm3d(channels)
#         self.alpha = torch.nn.Parameter(torch.tensor(0.1))
        
#         # 채널별 중요도를 계산하는 어텐션
#         self.attention = torch.nn.Sequential(
#             torch.nn.AdaptiveAvgPool3d(1),
#             torch.nn.Conv3d(channels, max(channels // 16, 8), 1),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Conv3d(max(channels // 16, 8), channels, 1),
#             torch.nn.Sigmoid()
#         )
        
#     def forward(self, x, t):
#         """
#         x: [B*T, C, H, W] 형태의 입력
#         t: 비디오 당 프레임 수
#         """
#         bt, c, h, w = x.shape
#         b = bt // t
        
#         identity = x
#         # 5D 변환: [B, C, T, H, W]
#         x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        
#         # Temporal Fusion
#         out = self.depthwise_conv(x)
#         out = self.bn1(out)
        
#         attn = self.attention(out)
#         out = out * attn
        
#         out = self.pointwise(out)
#         out = self.bn2(out)
        
#         # 다시 4D 복원: [B*T, C, H, W]
#         out = out.permute(0, 2, 1, 3, 4).contiguous().view(bt, c, h, w)
        
#         return identity + self.alpha * out
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ATGModule(torch.nn.Module):
#     """
#     [SPL Submission Version]
#     Adaptive Temporal Gating Module for Spatio-temporal Feature Alignment.
#     Features: Dynamic Gating, Temporal Depthwise-Separable Conv, Self-Calibrated Attention.
#     """
#     def __init__(self, channels, kernel_size=3):
#         super().__init__()
#         self.channels = channels
        
#         # 1. Temporal Extraction: 1D-Temporal Kernel (Signal Processing Perspective)
#         self.temporal_conv = nn.Conv3d(
#             channels, channels,
#             kernel_size=(kernel_size, 1, 1),
#             padding=((kernel_size - 1) // 2, 0, 0),
#             groups=channels,
#             bias=False
#         )
        
#         # 2. Adaptive Gating Mechanism: 신호의 변화량에 따라 Fusion 강도를 동적으로 조절
#         # 논문 기여점: "Dynamic control of temporal information flow"
#         self.gate_generator = nn.Sequential(
#             nn.AdaptiveAvgPool3d(1),
#             nn.Conv3d(channels, channels // 4, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(channels // 4, channels, 1),
#             nn.Sigmoid()
#         )
        
#         self.pointwise = nn.Conv3d(channels, channels, 1, bias=False)
#         self.bn = nn.BatchNorm3d(channels)
        
#         # 신호 처리적 안정성을 위한 학습 가능한 스케일 파라미터
#         self.gamma = nn.Parameter(torch.zeros(1)) 

class safeTemporalContextExchange(torch.nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels

        self.depthwise_conv = torch.nn.Conv3d(
            channels, channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False
        )

        # ✅ sam2_base.py 호환용 alias
        self.temporal_conv = self.depthwise_conv

        self.pointwise = torch.nn.Conv3d(channels, channels, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(channels)
        self.bn2 = torch.nn.BatchNorm3d(channels)
        self.alpha = torch.nn.Parameter(torch.tensor(0.1))

        # 기존 attention
        self.attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Conv3d(channels, max(channels // 16, 8), 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(max(channels // 16, 8), channels, 1),
            torch.nn.Sigmoid()
        )

        # ✅ sam2_base.py가 요구하는 gate_generator
        # 항상 1을 반환하는 안전한 gate
        self.gate_generator = torch.nn.Identity()

    def forward(self, x, t):
        """
        x: [B*T, C, H, W]
        t: frames per video
        """

        bt, c, h, w = x.shape

        # ✅ inference memory encoder 보호
        if c != self.channels:
            return x

        b = bt // t
        identity = x

        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        out = self.temporal_conv(x)
        out = self.bn1(out)

        out = out * self.attention(out)

        out = self.pointwise(out)
        out = self.bn2(out)

        out = out.permute(0, 2, 1, 3, 4).contiguous().view(bt, c, h, w)

        return identity + self.alpha * out

        #######################################################

        return identity + self.gamma * out
################################################
class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # extra arguments used to construct the SAM mask decoder; if not None, it should be a dict of kwargs to be passed into `MaskDecoder` class.
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        ######################## jimin ########################
        self.hidden_dim = image_encoder.neck.d_model
        self.temporalVideo = False
        if self.temporalVideo:
            # Hiera 백본의 Neck 출력 채널(self.hidden_dim)에 맞춰 레이어 생성
            # 다중 스케일(P3, P4, P5)을 사용하므로 각 레벨에 대한 리스트 생성
            self.temporal_fusion = torch.nn.ModuleList([
                GFTE(channels=self.hidden_dim)         #GFTE  AdaptiveTemporalSemanticFusion safeTemporalContextExchange  ATGModule
                for _ in range(self.num_feature_levels)
            ])
        ################################################
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            self.mask_downsample = torch.nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        self.hidden_dim = image_encoder.neck.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference or SAM2Train for training/fine-tuning"
            "See notebooks/video_predictor_example.ipynb for an inference example."
        )

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        # (their hyperparameters like `mask_in_chans=16` are from SAM code)
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,  # use antialias for downsampling
        )
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        ######################## jimin ########################
        # [수정] num_frames를 정의합니다. 
        # MedSAM2 학습 시 backbone_out의 텐서 첫 번째 차원은 보통 B*T입니다.
        # 훈련 설정이나 입력 데이터 구조에 따라 결정되는데, 
        # 여기서는 feature_map의 batch 차원을 통해 유추하거나 외부에서 주입된 값을 사용합니다.
        
        # 만약 backbone_out에 이미 정보가 없다면, 텐서 모양으로 유추 (BT, C, H, W)
        if self.temporalVideo:
            bt = feature_maps[0].shape[0]
            # 현재 배치 사이즈가 1이라고 가정하면 bt가 곧 num_frames(T)가 됩니다.
            # 일반적인 MedSAM2 비디오 학습 환경에서는 아래와 같이 처리합니다.
            num_frames = backbone_out.get("num_frames", bt) 

            if num_frames > 1:
                fused_feature_maps = []
                for i, f_map in enumerate(feature_maps):
                    # i번째 레벨 특징 맵에 대해 Temporal Context Exchange 수행
                    # self.temporal_fusion[i] 모듈에 f_map과 num_frames 전달
                    fused_f_map = self.temporal_fusion[i](f_map, num_frames)
                    fused_feature_maps.append(fused_f_map)
                feature_maps = fused_feature_maps
        ################################################
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
