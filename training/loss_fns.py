# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
from typing import Dict, List

from training.trainer import CORE_LOSS_KEY
from training.utils.distributed import get_world_size, is_dist_avail_and_initialized

# --- Helper Loss Functions ---

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        assert inputs.dim() == 4 and targets.dim() == 4
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs, targets, num_objects, alpha: float = 0.25, gamma: float = 2, loss_on_multimask=False,
):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects
    return loss.mean(1).sum() / num_objects


def iou_loss(inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False):
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

# --- Novel Temporal Loss ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConsistencyLoss(nn.Module):
    """
    [Hybrid Temporal Loss]
    1. Graph Laplacian: 전후 프레임 맥락 유지
    2. Semantic Weight: 예측 확신도에 따른 가중치 부여
    3. Flexible Penalty: 변화량에 따른 차등 페널티 (threshold 로직)
    """
    def __init__(self, 
                 alpha=0.1,         # Pairwise(인접) 로스 비중
                 beta=0.05,        # Graph(전후) 로스 비중
                 threshold=0.1,     # Flexible 모드 임계값
                 low_penalty=0.1,   # 임계값 미만 시 페널티 (미세 조정)
                 high_penalty=1.0,  # 임계값 이상 시 페널티 (급격한 변화 억제)
                 mode='flexible',   # 'flexible' 또는 'strict'
                 use_semantic_weight=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.low_penalty = low_penalty
        self.high_penalty = high_penalty
        self.mode = mode
        self.use_semantic_weight = use_semantic_weight

    def compute_semantic_weights(self, probs):
        """예측의 확신도(Confidence) 기반 시간적 가중치 계산"""
        # probs: [B, 1, T, H, W]
        confidence = 1.0 - 2.0 * torch.abs(probs - 0.5)
        # 공간 축 평균 -> [B, 1, T]
        spatial_confidence = confidence.mean(dim=[-1, -2])
        # 시간 축에 대해 소프트맥스로 정규화
        weights = F.softmax(spatial_confidence * 5.0, dim=-1)
        return weights.unsqueeze(-1).unsqueeze(-1) # [B, 1, T, 1, 1]

    def apply_flexible_penalty(self, diff_magnitude):
        """차이값의 크기에 따라 페널티를 차등 적용"""
        if self.mode == 'flexible':
            return torch.where(
                diff_magnitude < self.threshold,
                diff_magnitude * self.low_penalty,
                diff_magnitude * self.high_penalty
            )
        return diff_magnitude

    def forward(self, logits):
        # 1. 입력 차원 정규화 (MedSAM2의 다양한 입력 대응)
        # 목표: [B, 1, T, H, W]
        if logits.dim() == 3: # [T, H, W]
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif logits.dim() == 4: # [B, T, H, W]
            logits = logits.unsqueeze(1)
        
        B, C, T, H, W = logits.shape
        if T < 2:
            return torch.tensor(0.0, device=logits.device)
        
        probs = torch.sigmoid(logits)

        # 2. 기본 인접 프레임 일관성 (Pairwise)
        # t와 t-1의 차이 계산
        basic_diff = torch.abs(probs[:, :, 1:] - probs[:, :, :-1])
        basic_diff_mean = basic_diff.mean(dim=[-1, -2]) # 공간 평균 [B, 1, T-1]
        
        # Flexible 페널티 적용
        basic_loss = self.apply_flexible_penalty(basic_diff_mean).mean()

        # 3. 전후 맥락 그래프 손실 (Graph Laplacian)
        graph_loss = torch.tensor(0.0, device=logits.device)
        if T > 2:
            center = probs[:, :, 1:-1]
            left = probs[:, :, :-2]
            right = probs[:, :, 2:]
            
            # (중심-왼쪽) + (중심-오른쪽) 의 평균 차이
            graph_diff = (torch.abs(center - left) + torch.abs(center - right)) / 2.0
            graph_diff_mean = graph_diff.mean(dim=[-1, -2]) # [B, 1, T-2]
            
            # Flexible 페널티 적용
            graph_loss = self.apply_flexible_penalty(graph_diff_mean).mean()

        # 4. 의미론적 가중치 적용 (가장 확신 있는 프레임 기준으로 정렬)
        weighted_loss = torch.tensor(0.0, device=logits.device)
        if self.use_semantic_weight and T > 1:
            weights = self.compute_semantic_weights(probs)
            # 가중치가 반영된 프레임 간 차이
            weighted_diff = torch.abs(
                probs[:, :, 1:] * weights[:, :, 1:] - 
                probs[:, :, :-1] * weights[:, :, :-1]
            )
            weighted_loss = weighted_diff.mean()

        # 5. 최종 로스 결합 (기존 alpha, beta 비중 유지 + 가중치 로스)
        total_loss = (self.alpha * basic_loss + 
                      self.beta * graph_loss + 
                      0.05 * weighted_loss)
        
        return total_loss
    
    
class TemporalGraphConsistencyLoss(nn.Module):
    """
    비디오 프레임 간의 시공간적 일관성을 강제하는 손실 함수.
    논문 기여점: Graph Laplacian을 통한 Temporal Smoothing.
    """
    def __init__(self, alpha=0.1, beta=0.05, use_semantic_weight=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_semantic_weight = use_semantic_weight
        
    def compute_semantic_weights(self, probs):
        # 예측의 확신도(Confidence)를 기반으로 시간적 가중치 계산
        confidence = 1.0 - 2.0 * torch.abs(probs - 0.5)
        spatial_confidence = confidence.mean(dim=[-1, -2]) # [T]
        weights = F.softmax(spatial_confidence * 5.0, dim=-1) # [T]
        return weights.view(-1, 1, 1) # [T, 1, 1]

    def forward(self, logits):
        # logits: [T, H, W]
        T, H, W = logits.shape
        if T < 2: 
            return torch.tensor(0.0, device=logits.device)
        
        probs = torch.sigmoid(logits)
        
        # 1. 인접 프레임 간 차이 (Pairwise Consistency)
        diff = torch.abs(probs[1:] - probs[:-1])
        basic_loss = diff.mean()
        
        # 2. 전후 맥락을 고려한 그래프 손실 (Graph Laplacian)
        graph_loss = torch.tensor(0.0, device=logits.device)
        if T > 2:
            center = probs[1:-1]
            left = probs[:-2]
            right = probs[2:]
            graph_loss = (torch.abs(center - left) + torch.abs(center - right)).mean() / 2.0
            
        # 3. 의미론적 가중치 적용
        weighted_loss = torch.tensor(0.0, device=logits.device)
        if self.use_semantic_weight:
            weights = self.compute_semantic_weights(probs)
            weighted_diff = torch.abs(probs[1:] * weights[1:] - probs[:-1] * weights[:-1])
            weighted_loss = weighted_diff.mean()
            
        return self.alpha * basic_loss + self.beta * graph_loss + 0.05 * weighted_loss

# --- Main Multi-Step Loss Class ---

class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # 필수 가중치 확인
        for k in ["loss_mask", "loss_dice", "loss_iou"]:
            assert k in self.weight_dict
        
        # Temporal Loss 초기화 및 가중치 설정
        self.temporal_loss_fn = TemporalConsistencyLoss(alpha=0.1, beta=0.05)
        if "loss_temporal" not in self.weight_dict:
            self.weight_dict["loss_temporal"] = 0.5 # 기본값

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        # 1. 변수 선언 (NameError 방지)
        target_masks = targets.unsqueeze(1).float() # [T, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_temporal": 0}
        
        # 2. 다단계 마스크 손실 업데이트
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        
        # 3. 시간적 일관성 손실 계산 (최종 예측 결과 사용)
        # src_masks_list[-1] shape: [T, M, H, W] -> 0번 마스크 채널 선택
        final_prediction = src_masks_list[-1][:, 0, :, :] # [T, H, W]
        losses["loss_temporal"] = self.temporal_loss_fn(final_prediction)

        # 4. 전체 손실 가중치 합산
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)
        
        loss_multimask = sigmoid_focal_loss(
            src_masks, target_masks, num_objects, alpha=self.focal_alpha, gamma=self.focal_gamma, loss_on_multimask=True,
        )
        loss_multidice = dice_loss(src_masks, target_masks, num_objects, loss_on_multimask=True)
        
        if not self.pred_obj_scores:
            loss_class = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
            target_obj = torch.ones(loss_multimask.shape[0], 1, dtype=loss_multimask.dtype, device=loss_multimask.device)
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[..., None].float()
            loss_class = sigmoid_focal_loss(object_score_logits, target_obj, num_objects, alpha=self.focal_alpha_obj_score, gamma=self.focal_gamma_obj_score)

        loss_multiiou = iou_loss(src_masks, target_masks, ious, num_objects, loss_on_multimask=True, use_l1_loss=self.iou_use_l1_loss)
        
        if loss_multimask.size(1) > 1:
            loss_combo = (loss_multimask * self.weight_dict["loss_mask"] + loss_multidice * self.weight_dict["loss_dice"])
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask, loss_dice, loss_iou = loss_multimask, loss_multidice, loss_multiiou

        losses["loss_mask"] += (loss_mask * target_obj).sum()
        losses["loss_dice"] += (loss_dice * target_obj).sum()
        losses["loss_iou"] += (loss_iou * target_obj).sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        return reduced_loss