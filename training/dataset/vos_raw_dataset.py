# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch
import numpy as np

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    SA1BSegmentLoader,
    NPZSegmentLoader
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset # <--- 필수 임포트

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

######################## jimin ########################
class MedSAM2CurriculumDataset(Dataset):
    def __init__(self, folder, milestones, **kwargs):
        self.base_folder = folder
        self.milestones = milestones
        self.stage = "dense"
        self.temporal_stride = 1  # 신호 샘플링 간격 추가
        self.samples = []
        self._load_stage_data("dense")

    def _load_stage_data(self, stage):
        self.stage = stage
        self.target_path = os.path.join(self.base_folder, self.stage, "uterine_niche")
        
        if not os.path.exists(self.target_path):
            self.target_path = os.path.join(self.base_folder, self.stage)

        if os.path.exists(self.target_path):
            self.samples = sorted([f for f in os.listdir(self.target_path) if f.endswith('.npz')])
        else:
            self.samples = []

        logging.info(f"✅ [Dataset] Stage: {self.stage.upper()} | Samples: {len(self.samples)}")

    import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

class MedSAM2CurriculumDataset2(Dataset):
    """
    [IEEE SPL Submission Version]
    Proposed: Progressive Curriculum with Stochastic Temporal Jittering.
    Innovation: Suppresses temporal aliasing by randomizing sampling offsets within stride windows.
    """
    def __init__(self, folder, milestones, **kwargs):
        self.base_folder = folder
        self.milestones = milestones
        self.stage = "dense"
        self.temporal_stride = 1     # 기본 샘플링 간격
        self.use_stochastic = True  # 🔥 혁신 포인트: 확률적 샘플링 활성화 스위치
        self.samples = []
        self._load_stage_data("dense")

    def _load_stage_data(self, stage):
        self.stage = stage
        # 경로 규칙: base_folder / stage / uterine_niche
        self.target_path = os.path.join(self.base_folder, self.stage, "uterine_niche")
        
        if not os.path.exists(self.target_path):
            self.target_path = os.path.join(self.base_folder, self.stage)

        if os.path.exists(self.target_path):
            self.samples = sorted([f for f in os.listdir(self.target_path) if f.endswith('.npz')])
        else:
            self.samples = []

        logging.info(f"✅ [Curriculum] Stage changed: {self.stage.upper()} | Samples: {len(self.samples)}")

    def update_curriculum_stage(self, epoch):
        """에포크에 따라 단계를 동적으로 변경 (Trainer에서 호출 가능)"""
        target_stage = "dense"
        if epoch >= self.milestones.get("full", 50): target_stage = "full"
        elif epoch >= self.milestones.get("expand", 20): target_stage = "expand"

        if target_stage != self.stage:
            self._load_stage_data(target_stage)
            return True
        return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.samples:
            return None
            
        npz_name = self.samples[idx]
        npz_path = os.path.join(self.target_path, npz_name)
        
        try:
            # 1. NPZ 데이터 로드
            data = np.load(npz_path, allow_pickle=True)
            imgs = data['imgs']    # (T, H, W, 3)
            masks = data['masks']  # (T, H, W)
            T = len(imgs)
            
            # 2. 🔥 [혁신 로직] Stochastic Temporal Jittering
            # SPL 저널 포인트: "Temporal Regularization via Non-uniform Sampling"
            # 단순히 0, 2, 4.. 프레임을 가져오는 대신, 각 구간 [i, i+stride] 내에서 
            # 확률적으로 프레임을 선택하여 시간적 에일리어싱을 방지함.
            if self.stage == "expand" and self.use_stochastic and self.temporal_stride > 1:
                indices = []
                for i in range(0, T, self.temporal_stride):
                    # 구간 내에서 무작위 오프셋(jitter) 선택
                    jitter = torch.randint(0, self.temporal_stride, (1,)).item()
                    chosen_idx = min(i + jitter, T - 1)
                    indices.append(chosen_idx)
                
                imgs = imgs[indices]
                masks = masks[indices]
                # logging.debug(f"Jittered indices: {indices}") # 필요 시 확인용
            
            # 'dense' 단계에서는 정적 특징 강화를 위해 첫 프레임(또는 한 장)만 학습
            elif self.stage == "dense":
                imgs = imgs[:1]
                masks = masks[:1]

            # 3. 데이터 포맷 정규화 (T, H, W, 3) -> (T, 3, H, W)
            if imgs.ndim == 4 and imgs.shape[-1] == 3:
                imgs = imgs.transpose(0, 3, 1, 2)
            
            return {
                "video_id": npz_name.replace(".npz", ""),
                "images": torch.from_numpy(imgs).float(),
                "masks": torch.from_numpy(masks).float(),
                "num_frames": len(imgs)
            }
        except Exception as e:
            logging.error(f"❌ Error loading {npz_path}: {e}")
            # 에러 발생 시 재귀적으로 다음 인덱스 시도
            return self.__getitem__((idx + 1) % len(self.samples))

    def __len__(self):
        return len(self.samples)
#######################################################
class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root, sample_rate=self.sample_rate)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for idx, fpath in enumerate(all_frames[::self.sample_rate]):
            fid = idx # int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class NPZRawDataset5(VOSRawDataset):
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # Read all npz files from folder and its subfolders
        subset = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.endswith('.npz'):
                    # Get the relative path from the root folder
                    rel_path = os.path.relpath(os.path.join(root, file), self.folder)
                    # Remove the .npz extension
                    subset.append(os.path.splitext(rel_path)[0])

        # Read the subset defined in file_list_txt if provided
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f if line.strip() in subset]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")
        
        # Load NPZ file
        npz_data = np.load(npz_path)
        
        # Extract frames and masks
        frames = npz_data['imgs'] / 255.0
        # Expand the grayscale images to three channels
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)
        masks = npz_data['gts']
        
        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]
        
        # Create VOSFrame objects
        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))
        
        # Create VOSVideo object
        video = VOSVideo(video_name, idx, vos_frames)
        
        # Create NPZSegmentLoader
        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])
        
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

class NPZRawDataset(VOSRawDataset):
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # 1. 폴더 내 모든 npz 파일 탐색
        subset = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.endswith('.npz'):
                    rel_path = os.path.relpath(os.path.join(root, file), self.folder)
                    subset.append(os.path.splitext(rel_path)[0])

        # 2. file_list_txt가 제공된 경우 필터링
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f if line.strip() in subset]

        # 3. 제외 목록 처리
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # 4. GT 존재 여부 확인 및 최종 비디오 목록 확정 (핵심 수정 부분)
        print("Filtering videos without Ground Truth (GT)...")
        final_video_list = []
        
        # 제외 목록에 없는 비디오들을 대상으로 검사
        candidate_videos = [v for v in subset if v not in excluded_files]
        
        for video_name in candidate_videos:
            npz_path = os.path.join(self.folder, f"{video_name}.npz")
            try:
                # npz 파일을 로드하여 gts(Ground Truth) 확인
                npz_data = np.load(npz_path)
                # gts 내에 1(객체)이 하나라도 존재하는지 확인
                if 'gts' in npz_data and np.sum(npz_data['gts']) > 0:
                    final_video_list.append(video_name)
            except Exception as e:
                print(f"Error loading {npz_path}: {e}")
                continue

        self.video_names = sorted(final_video_list)
        print(f"Filtering complete. Final dataset size: {len(self.video_names)} videos.")
    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")
        
        # Load NPZ file
        npz_data = np.load(npz_path)
        
        # Extract frames and masks
        frames = npz_data['imgs'] / 255.0
        # Expand the grayscale images to three channels
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)
        masks = npz_data['gts']
        
        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]
        
        # Create VOSFrame objects
        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))
        
        # Create VOSVideo object
        video = VOSVideo(video_name, idx, vos_frames)
        
        # Create NPZSegmentLoader
        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])
        
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

import os
import numpy as np
import torch
import logging

# from training.dataset.vos_raw_dataset import NPZRawDataset
# from training.dataset.vos_video import VOSVideo, VOSFrame
# from training.dataset.vos_segment_loader import NPZSegmentLoader
import os
import numpy as np
import torch
import logging


class AESCurriculumNPZRawDataset(VOSRawDataset):
    """
    Adaptive Entropy Sampling Curriculum Learning (AES-CL)

    ✔ Drop-in replacement for QuantumNPZRawDataset
    ✔ Entropy-based curriculum (image + mask complexity)
    ✔ Stage-wise sample filtering (dense → expand → full)
    """

    def __init__(
        self,
        folder,
        milestones,
        entropy_threshold=0.7,
        sample_rate=1,
        truncate_video=-1,
        **kwargs
    ):
        self.folder = folder
        self.milestones = milestones
        self.entropy_threshold = entropy_threshold
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        self.stage = "dense"
        self.adaptive_factor = 1.0

        self.video_names = []
        self.sample_entropies = {}

        self._load_stage_data("dense")

    # ------------------------------------------------------------------
    # Curriculum control (Trainer에서 epoch마다 호출 가능)
    # ------------------------------------------------------------------
    def update_curriculum_stage(self, epoch, training_loss=None):
        target_stage = "dense"
        if epoch >= self.milestones.get("full", 50):
            target_stage = "full"
        elif epoch >= self.milestones.get("expand", 20):
            target_stage = "expand"

        if training_loss is not None:
            if training_loss < 0.1:
                self.adaptive_factor = min(2.0, self.adaptive_factor * 1.05)
            else:
                self.adaptive_factor = max(0.5, self.adaptive_factor * 0.95)

        if target_stage != self.stage:
            self._load_stage_data(target_stage)
            logging.info(
                f"[AES-CL] Stage → {self.stage} | "
                f"Adaptive factor: {self.adaptive_factor:.2f} | "
                f"Videos: {len(self.video_names)}"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Core loading logic
    # ------------------------------------------------------------------
    def _load_stage_data(self, stage):
        self.stage = stage
        self.video_names = []
        self.sample_entropies.clear()

        target_path = os.path.join(self.folder, self.stage)
        if not os.path.exists(target_path):
            target_path = self.folder

        if not os.path.exists(target_path):
            return

        for fname in sorted(os.listdir(target_path)):
            if not fname.endswith(".npz"):
                continue

            npz_path = os.path.join(target_path, fname)
            try:
                data = np.load(npz_path, allow_pickle=True)
                imgs = data["imgs"]
                masks = data["gts"] if "gts" in data else data["masks"]

                if len(imgs) == 0:
                    continue

                img_entropy = self._compute_image_entropy(imgs[0])
                mask_complexity = self._compute_mask_complexity(masks[0])
                total_entropy = (img_entropy + mask_complexity) / 2.0

                self.sample_entropies[fname] = total_entropy

                if self._entropy_filter(total_entropy):
                    self.video_names.append(os.path.splitext(fname)[0])

            except Exception as e:
                logging.warning(f"[AES-CL] Failed loading {npz_path}: {e}")

    # ------------------------------------------------------------------
    # Entropy logic
    # ------------------------------------------------------------------
    def _entropy_filter(self, entropy):
        if self.stage == "dense":
            return entropy < 0.3 * self.adaptive_factor
        elif self.stage == "expand":
            return entropy < 0.6 * self.adaptive_factor
        else:
            return True

    def _compute_image_entropy(self, image):
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        hist, _ = np.histogram(image.flatten(), bins=32, range=(0, 1))
        prob = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        return entropy / 5.0

    def _compute_mask_complexity(self, mask):
        if np.sum(mask) == 0:
            return 0.0

        from skimage.measure import perimeter, euler_number
        area = np.sum(mask)
        perim = perimeter(mask)
        complexity = perim / (area + 1e-10) * 0.1 + abs(euler_number(mask)) * 0.1
        return min(complexity, 1.0)

    # ------------------------------------------------------------------
    # Required VOSRawDataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.video_names)

    def get_video(self, idx):
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, self.stage, f"{video_name}.npz")
        if not os.path.exists(npz_path):
            npz_path = os.path.join(self.folder, f"{video_name}.npz")

        data = np.load(npz_path, allow_pickle=True)
        frames = data["imgs"]
        masks = data["gts"] if "gts" in data else data["masks"]

        if frames.max() > 1.0:
            frames = frames / 255.0

        if frames.ndim == 3:
            frames = np.repeat(frames[:, None], 3, axis=1)
        elif frames.shape[-1] == 3:
            frames = frames.transpose(0, 3, 1, 2)

        if self.truncate_video > 0:
            frames = frames[: self.truncate_video]
            masks = masks[: self.truncate_video]

        vos_frames = []
        for i, frame in enumerate(frames[:: self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(
                VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame))
            )

        video = VOSVideo(video_name, idx, vos_frames)
        segment_loader = NPZSegmentLoader(masks[:: self.sample_rate])

        return video, segment_loader

class NeuroSpectralNPZRawDataset(NPZRawDataset):
    """
    Neuro-Spectral Curriculum Learning at Raw-Video Level
    - Frequency-domain difficulty ordering
    - Stage-wise video subset expansion
    """

    def __init__(
        self,
        folder,
        milestones,
        sample_rate=1,
        truncate_video=-1,
        **kwargs
    ):
        super().__init__(
            folder=folder,
            sample_rate=sample_rate,
            truncate_video=truncate_video,
        )

        self.milestones = milestones
        self.stage = "dense"

        # 스펙트럼 난이도 계산 (1회)
        self.video_scores = self._compute_spectral_scores()

        # 난이도 기준 정렬 (쉬운 → 어려운)
        self.sorted_videos = sorted(
            self.video_scores.keys(),
            key=lambda k: self.video_scores[k]
        )

        self._apply_stage_filter()

        logging.info(
            f"🧠 [NeuroSpectral] Init | Stage: {self.stage} | "
            f"Videos: {len(self.video_names)}"
        )

    # ------------------------------------------------------------------
    # Spectral difficulty (low freq = easy, high freq = hard)
    # ------------------------------------------------------------------
    def _compute_spectral_scores(self):
        scores = {}

        for v in self.video_names:
            npz_path = os.path.join(self.folder, f"{v}.npz")
            try:
                data = np.load(npz_path, allow_pickle=True)
                imgs = data["imgs"]

                # (T,H,W,3) or (T,H,W)
                if imgs.ndim == 4:
                    gray = np.mean(imgs, axis=(1, 2, 3))
                else:
                    gray = np.mean(imgs, axis=(1, 2))

                # FFT (temporal)
                fft = np.fft.fft(gray)
                mag = np.abs(fft)

                low = np.mean(mag[: len(mag)//4])
                high = np.mean(mag[len(mag)//4:])

                # high / low 비율 → 어려움
                score = high / (low + 1e-6)
                scores[v] = float(score)

            except Exception as e:
                logging.warning(f"[NeuroSpectral] FFT failed: {v} | {e}")
                scores[v] = 1.0

        return scores

    # ------------------------------------------------------------------
    # Curriculum control
    # ------------------------------------------------------------------
    def update_curriculum_stage(self, epoch):
        target_stage = "dense"
        if epoch >= self.milestones.get("full", 50):
            target_stage = "full"
        elif epoch >= self.milestones.get("expand", 20):
            target_stage = "expand"

        if target_stage != self.stage:
            self.stage = target_stage
            self._apply_stage_filter()
            logging.info(
                f"🧠 [NeuroSpectral] Stage → {self.stage.upper()} | "
                f"Videos: {len(self.video_names)}"
            )
            return True

        return False

    def _apply_stage_filter(self):
        N = len(self.sorted_videos)

        if self.stage == "dense":
            keep = int(0.3 * N)
        elif self.stage == "expand":
            keep = int(0.6 * N)
        else:  # full
            keep = N

        self.video_names = self.sorted_videos[: max(1, keep)]

class QuantumNPZRawDataset(NPZRawDataset):
    """
    Quantum-Resonance Curriculum Raw Dataset
    - NPZRawDataset 완전 호환
    - 파일 단위(curriculum at raw-video level)
    """

    def __init__(
        self,
        folder,
        milestones,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        # 👉 부모에서 video_names 생성 (GT filtering 포함)
        super().__init__(
            folder=folder,
            file_list_txt=file_list_txt,
            excluded_videos_list_txt=excluded_videos_list_txt,
            sample_rate=sample_rate,
            truncate_video=truncate_video,
        )

        # curriculum 관련
        self.milestones = milestones
        self.stage = "dense"
        self.epoch = 0

        # quantum 상태
        self.quantum_state = "ground"

        # 초기 필터링
        self._apply_curriculum_filter()

        logging.info(
            f"⚛️ [QuantumNPZRawDataset] Init | Stage={self.stage} | Videos={len(self.video_names)}"
        )

    # --------------------------------------------------
    # Curriculum control
    # --------------------------------------------------
    def update_curriculum_stage(self, epoch):
        self.epoch = epoch

        if epoch >= self.milestones.get("full", 50):
            target_stage = "full"
        elif epoch >= self.milestones.get("expand", 20):
            target_stage = "expand"
        else:
            target_stage = "dense"

        if target_stage != self.stage:
            self.stage = target_stage
            self._update_quantum_state()
            self._apply_curriculum_filter()

            logging.info(
                f"🔄 [QuantumNPZRawDataset] Epoch {epoch} | "
                f"Stage={self.stage.upper()} | "
                f"Quantum={self.quantum_state} | "
                f"Videos={len(self.video_names)}"
            )

            return True
        return False

    def _update_quantum_state(self):
        if self.stage == "dense":
            self.quantum_state = "ground"
        elif self.stage == "expand":
            self.quantum_state = "excited"
        else:
            self.quantum_state = "resonant"

    # --------------------------------------------------
    # Curriculum filtering
    # --------------------------------------------------
    def _apply_curriculum_filter(self):
        """
        파일 단위 필터링
        - dense  : 쉬운 비디오
        - expand : 중간
        - full   : 전체
        """
        if self.stage == "full":
            return  # 전체 사용

        filtered = []
        for v in self.video_names:
            npz_path = os.path.join(self.folder, f"{v}.npz")
            try:
                data = np.load(npz_path, allow_pickle=True)
                imgs = data["imgs"]

                # difficulty proxy: 첫 프레임 entropy
                img0 = imgs[0]
                if img0.ndim == 3:
                    img0 = img0.mean(axis=-1)

                hist, _ = np.histogram(img0.flatten(), bins=32)
                prob = hist / (hist.sum() + 1e-8)
                entropy = -np.sum(prob * np.log(prob + 1e-10))

                if self.stage == "dense":
                    if entropy < 2.5:
                        filtered.append(v)
                elif self.stage == "expand":
                    if entropy < 3.5:
                        filtered.append(v)

            except Exception as e:
                logging.warning(f"[QuantumNPZRawDataset] skip {v}: {e}")

        # 최소 개수 보장
        if len(filtered) < 5:
            filtered = self.video_names[: min(10, len(self.video_names))]

        self.video_names = sorted(filtered)

    # --------------------------------------------------
    # Video access (부모 로직 그대로)
    # --------------------------------------------------
    def get_video(self, idx):
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")

        npz_data = np.load(npz_path)

        frames = npz_data["imgs"] / 255.0
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)
        masks = npz_data["gts"]

        if self.truncate_video > 0:
            frames = frames[: self.truncate_video]
            masks = masks[: self.truncate_video]

        vos_frames = []
        for i, frame in enumerate(frames[:: self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(
                VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame))
            )

        video = VOSVideo(video_name, idx, vos_frames)
        segment_loader = NPZSegmentLoader(masks[:: self.sample_rate])

        return video, segment_loader

class NPZRawDatasetOri(VOSRawDataset):
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # Read all npz files from folder and its subfolders
        subset = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.endswith('.npz'):
                    # Get the relative path from the root folder
                    rel_path = os.path.relpath(os.path.join(root, file), self.folder)
                    # Remove the .npz extension
                    subset.append(os.path.splitext(rel_path)[0])

        # Read the subset defined in file_list_txt if provided
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f if line.strip() in subset]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")
        
        # Load NPZ file
        npz_data = np.load(npz_path)
        
        # Extract frames and masks
        frames = npz_data['imgs'] / 255.0
        # Expand the grayscale images to three channels
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)
        masks = npz_data['gts']
        
        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]
        
        # Create VOSFrame objects
        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))
        
        # Create VOSVideo object
        video = VOSVideo(video_name, idx, vos_frames)
        
        # Create NPZSegmentLoader
        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])
        
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

class SA1BRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)
            subset = [
                path.split(".")[0] for path in subset if path.endswith(".jpg")
            ]  # remove extension

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files and it exists
        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")

        segment_loader = SA1BSegmentLoader(
            video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)


class NPZRawDataset(VOSRawDataset):
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # Read all npz files from folder and its subfolders
        subset = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.endswith('.npz'):
                    # Get the relative path from the root folder
                    rel_path = os.path.relpath(os.path.join(root, file), self.folder)
                    # Remove the .npz extension
                    subset.append(os.path.splitext(rel_path)[0])

        # Read the subset defined in file_list_txt if provided
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f if line.strip() in subset]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")
        
        # Load NPZ file
        npz_data = np.load(npz_path)
        
        # Extract frames and masks
        frames = npz_data['imgs'] / 255.0
        # Expand the grayscale images to three channels
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)
        masks = npz_data['gts']
        
        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]
        
        # Create VOSFrame objects
        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))
        
        # Create VOSVideo object
        video = VOSVideo(video_name, idx, vos_frames)
        
        # Create NPZSegmentLoader
        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])
        
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)