# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Callable, Iterable, List, Optional, Sequence

import torch

from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset

from torch.utils.data.distributed import DistributedSampler



class TorchTrainMixedDataset:
    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        phases_per_epoch: int = 1,
        dataset_prob: Optional[List[float]] = None,
    ) -> None:
        
        """
            Args:
                datasets (List[Dataset]): List of Datasets to be mixed.
                batch_sizes (List[int]): Batch sizes for each dataset in the list.
                num_workers (int): Number of workers per dataloader.
                shuffle (bool): Whether or not to shuffle data.
                pin_memory (bool): If True, use pinned memory when loading tensors from disk.
                drop_last (bool): Whether or not to drop the last batch of data.
                collate_fn (Callable): Function to merge a list of samples into a mini-batch.
                worker_init_fn (Callable): Function to init each dataloader worker.
                phases_per_epoch (int): Number of phases per epoch.
                dataset_prob (List[float]): Probability of choosing the dataloader to sample from. Should sum to 1.0
        """

        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert len(self.datasets) > 0
        for dataset in self.datasets:
            assert not isinstance(dataset, IterableDataset), "Not supported"
            # `RepeatFactorWrapper` requires calling set_epoch first to get its length
            self._set_dataset_epoch(dataset, 0)
        self.phases_per_epoch = phases_per_epoch
        self.chunks = [None] * len(datasets)
        if dataset_prob is None:
            # If not provided, assign each dataset a probability proportional to its length.
            dataset_lens = [
                (math.floor(len(d) / bs) if drop_last else math.ceil(len(d) / bs))
                for d, bs in zip(datasets, batch_sizes)
            ]
            total_len = sum(dataset_lens)
            dataset_prob = torch.tensor([d_len / total_len for d_len in dataset_lens])
        else:
            assert len(dataset_prob) == len(datasets)
            dataset_prob = torch.tensor(dataset_prob)

        logging.info(f"Dataset mixing probabilities: {dataset_prob.tolist()}")
        assert dataset_prob.sum().item() == 1.0, "Probabilities should sum to 1.0"
        self.dataset_prob = dataset_prob

    def _set_dataset_epoch(self, dataset, epoch: int) -> None:
        if hasattr(dataset, "epoch"):
            dataset.epoch = epoch
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def get_loader(self, epoch) -> Iterable:
        # ######################## jimin ########################
        # 커리큘럼 학습 사용 여부 스위치 (True: 단계별 학습, False: 처음부터 전체 비디오 학습)
        temporalVideo = False 
        # #######################################################

        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(
            zip(self.datasets, self.batch_sizes)
        ):
            # [단계 1] 커리큘럼 학습을 사용할 경우 (기존 로직)
            if temporalVideo and self.phases_per_epoch > 1:
                main_epoch = epoch // self.phases_per_epoch
                local_phase = epoch % self.phases_per_epoch

                if local_phase == 0 or self.chunks[d_idx] is None:
                    self._set_dataset_epoch(dataset, main_epoch)
                    g = torch.Generator()
                    g.manual_seed(main_epoch)
                    self.chunks[d_idx] = torch.chunk(
                        torch.randperm(len(dataset), generator=g),
                        self.phases_per_epoch,
                    )
                dataset = Subset(dataset, self.chunks[d_idx][local_phase])
            
            # [단계 2] 커리큘럼을 껐거나, 처음부터 전체 데이터를 쓸 경우
            else:
                self._set_dataset_epoch(dataset, epoch)
                # ######################## jimin ########################
                # 커리큘럼을 끈 경우, 데이터셋 내부의 stage를 강제로 'full'로 고정합니다.
                if hasattr(dataset, "stage"):
                    dataset.stage = "full"
                    logging.info(f"⚠️ [Curriculum Bypass] Stage forced to FULL for dataset {d_idx}")
                # #######################################################

            # 이후 샘플러 및 로더 설정은 동일하게 유지
            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            sampler.set_epoch(epoch)

            batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.drop_last)
            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    batch_sampler=batch_sampler,
                    collate_fn=self.collate_fn,
                    worker_init_fn=self.worker_init_fn,
                )
            )
        return MixedDataLoader(dataloaders, self.dataset_prob)





class MixedDataLoader:
    def __init__(self, dataloaders, dataset_prob):
        self.dataloaders = dataloaders
        # 중요: self.dataset_prob로 할당해야 __iter__에서 호출 가능합니다.
        self.dataset_prob = dataset_prob

    def __iter__(self):
        # 각 데이터셋의 이터레이터 생성
        iters = [iter(loader) for loader in self.dataloaders]
        while True:
            # 설정된 확률(dataset_prob)에 따라 데이터셋 인덱스 선택
            try:
                d_idx = torch.multinomial(self.dataset_prob, 1).item()
                yield next(iters[d_idx])
            except StopIteration:
                # 하나라도 데이터셋이 소진되면 종료
                break
            except Exception as e:
                # 예상치 못한 에러 로깅
                logging.error(f"Error during iteration: {e}")
                break

    def __len__(self):
        return sum(len(loader) for loader in self.dataloaders)

    def __iter__(self):
        # 각 데이터셋의 이터레이터를 생성합니다.
        iters = [iter(loader) for loader in self.dataloaders]
        while True:
            # 설정된 확률(dataset_prob)에 따라 어떤 데이터셋에서 배치를 가져올지 결정합니다.
            d_idx = torch.multinomial(self.dataset_prob, 1).item()
            try:
                yield next(iters[d_idx])
            except StopIteration:
                # 하나라도 데이터셋이 끝나면 반복을 종료합니다.
                break

    def __next__(self):
        """
        Sample a dataloader to sample from based on mixing probabilities. If one of the dataloaders is exhausted, we continue sampling from the other loaders until all are exhausted.
        """
        if self._iter_dls is None:
            raise TypeError(f"{type(self).__name__} object is not an iterator")

        while self._iter_mixing_prob.any():  # at least one D-Loader with non-zero prob.
            dataset_idx = self._iter_mixing_prob.multinomial(
                1, generator=self.random_generator
            ).item()
            try:
                item = next(self._iter_dls[dataset_idx])
                return item
            except StopIteration:
                # No more iterations for this dataset, set it's mixing probability to zero and try again.
                self._iter_mixing_prob[dataset_idx] = 0
            except Exception as e:
                # log and raise any other unexpected error.
                logging.error(e)
                raise e

        # Exhausted all iterators
        raise StopIteration


######################## jimin ########################
import numpy as np
import torch
import torch.nn.functional as F
import pywt
from typing import Dict, List, Optional
import logging

class FAP_CLDataset(Dataset):
    """
    Frequency-Aware Progressive Curriculum Learning for Medical Ultrasound Video
    IEEE SPL-style innovation: Signal processing perspective in curriculum learning
    """
    
    def __init__(
        self,
        folder: str,
        milestones: Dict[str, int],
        wavelet_type: str = 'db4',
        freq_bands: List[str] = ['LL', 'LH', 'HL', 'HH'],  # Wavelet subbands
        curriculum_schedule: str = 'low_to_high',  # 'low_to_high' or 'hybrid'
        **kwargs
    ):
        """
        Args:
            folder: 데이터 폴더 경로
            milestones: {'dense': 0, 'expand': 20, 'full': 50} 같은 curriculum 단계
            wavelet_type: 사용할 wavelet 종류 ('haar', 'db4', 'sym4' 등)
            freq_bands: 사용할 주파수 대역
            curriculum_schedule: 저주파 → 고주파 or hybrid 학습 전략
        """
        self.base_folder = folder
        self.milestones = milestones
        self.wavelet_type = wavelet_type
        self.freq_bands = freq_bands
        self.curriculum_schedule = curriculum_schedule
        
        # 현재 단계와 필터링된 샘플들
        self.stage = "dense"
        self.current_freq_weights = self._get_freq_weights(self.stage)
        self.samples = []
        
        # FFT 기반 데이터 분석 결과 저장
        self.freq_features = {}  # 비디오별 주파수 특징 저장
        
        self._load_and_analyze_data()
        
        logging.info(f"✅ [SignalCurriculum] Stage: {self.stage.upper()} | "
                    f"Freq weights: {self.current_freq_weights} | "
                    f"Samples: {len(self.samples)}")
    
    def _get_freq_weights(self, stage: str) -> Dict[str, float]:
        """단계별 주파수 대역 가중치 할당"""
        if stage == "dense":
            # 초기: 저주파(Low-Low) 강조
            return {'LL': 1.0, 'LH': 0.3, 'HL': 0.3, 'HH': 0.1}
        elif stage == "expand":
            # 중간: 중간 주파수 추가
            return {'LL': 0.7, 'LH': 0.8, 'HL': 0.8, 'HH': 0.4}
        elif stage == "full":
            # 후기: 모든 주파수 균등
            return {'LL': 0.6, 'LH': 0.9, 'HL': 0.9, 'HH': 0.8}
        else:
            return {'LL': 1.0, 'LH': 1.0, 'HL': 1.0, 'HH': 1.0}
    
    def _analyze_video_frequency(self, npz_path: str) -> Dict:
        """비디오의 주파수 특징 분석 (Wavelet Transform 사용)"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            imgs = data['imgs']  # (T, H, W, 3) or (T, H, W)
            
            # 첫 번째 프레임만 분석 (대표성)
            if imgs.ndim == 4:
                frame = imgs[0, :, :, 0]  # 첫 채널 사용
            else:
                frame = imgs[0]
            
            # Wavelet Transform으로 주파수 대역 분해
            coeffs = pywt.dwt2(frame, self.wavelet_type)
            LL, (LH, HL, HH) = coeffs
            
            # 각 대역의 에너지 계산
            energies = {
                'LL': np.mean(np.abs(LL)),
                'LH': np.mean(np.abs(LH)),
                'HL': np.mean(np.abs(HL)),
                'HH': np.mean(np.abs(HH))
            }
            
            # 대역별 엔트로피 (복잡도 측정)
            entropies = {}
            for band_name, band_data in zip(['LL', 'LH', 'HL', 'HH'], [LL, LH, HL, HH]):
                hist, _ = np.histogram(band_data.flatten(), bins=32)
                prob = hist / hist.sum()
                entropies[f'entropy_{band_name}'] = -np.sum(prob * np.log(prob + 1e-10))
            
            return {**energies, **entropies}
        except Exception as e:
            logging.warning(f"Frequency analysis failed for {npz_path}: {e}")
            return {band: 1.0 for band in self.freq_bands}
    
    def _load_and_analyze_data(self):
        """데이터 로드 및 주파수 분석"""
        self.target_path = os.path.join(self.base_folder, self.stage, "uterine_niche")
        
        if not os.path.exists(self.target_path):
            self.target_path = os.path.join(self.base_folder, self.stage)
        
        if not os.path.exists(self.target_path):
            self.samples = []
            return
        
        # 모든 NPZ 파일 수집
        all_samples = sorted([f for f in os.listdir(self.target_path) if f.endswith('.npz')])
        
        # 주파수 특징 분석 및 샘플 필터링
        filtered_samples = []
        self.freq_features.clear()
        
        for sample in all_samples:
            npz_path = os.path.join(self.target_path, sample)
            
            # 주파수 특징 분석
            freq_feats = self._analyze_video_frequency(npz_path)
            self.freq_features[sample] = freq_feats
            
            # 현재 curriculum 단계에 맞는 샘플인지 평가
            if self._should_include_sample(freq_feats):
                filtered_samples.append(sample)
        
        self.samples = filtered_samples
        
        # 샘플별 중요도 가중치 할당 (선택적)
        self.sample_weights = self._compute_sample_weights()
    
    def _should_include_sample(self, freq_feats: Dict) -> bool:
        """주파수 특징 기반 샘플 필터링"""
        # 예: 초기 단계에서는 고주파 노이즈가 많은 샘플 제외
        if self.stage == "dense":
            # 저주파 대비 고주파 에너지 비율이 낮은 샘플 선호
            low_freq_energy = freq_feats.get('LL', 1.0)
            high_freq_energy = freq_feats.get('HH', 0.1)
            return (high_freq_energy / (low_freq_energy + 1e-10)) < 0.3
        
        elif self.stage == "expand":
            # 중간 주파수 대역이 충분히 있는 샘플 선호
            mid_freq_energy = (freq_feats.get('LH', 0) + freq_feats.get('HL', 0)) / 2
            return mid_freq_energy > 0.2
        
        else:  # full stage
            return True  # 모든 샘플 포함
    
    def _compute_sample_weights(self) -> Dict[str, float]:
        """샘플별 학습 가중치 계산 (선택적 샘플링용)"""
        weights = {}
        for sample in self.samples:
            feats = self.freq_features.get(sample, {})
            
            # 가중치 = 현재 단계에서 중요도 높은 주파수 대역의 강도
            weight = 0.0
            for band, band_weight in self.current_freq_weights.items():
                if band in feats:
                    weight += band_weight * feats[band]
            
            # 엔트로피(복잡도) 고려
            entropy = feats.get(f'entropy_{self.freq_bands[0]}', 1.0)
            weight *= (1.0 + 0.2 * entropy)  # 복잡한 샘플에 약간 더 높은 가중치
            
            weights[sample] = max(0.1, weight)  # 최소 가중치 보장
        
        # 정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _apply_frequency_enhancement(self, image: np.ndarray) -> np.ndarray:
        """현재 curriculum 단계에 맞는 주파수 강화 적용"""
        if len(image.shape) == 3:  # RGB
            enhanced_channels = []
            for c in range(image.shape[-1]):
                channel = image[..., c]
                enhanced = self._enhance_single_channel(channel)
                enhanced_channels.append(enhanced)
            result = np.stack(enhanced_channels, axis=-1)
        else:  # Grayscale
            result = self._enhance_single_channel(image)
        
        return np.clip(result, 0, 1)
    
    def _enhance_single_channel(self, channel: np.ndarray) -> np.ndarray:
        """단일 채널 주파수 강화"""
        # Wavelet Transform
        coeffs = pywt.dwt2(channel, self.wavelet_type)
        LL, (LH, HL, HH) = coeffs
        
        # 현재 단계의 가중치로 각 대역 조정
        LL = LL * self.current_freq_weights['LL']
        LH = LH * self.current_freq_weights['LH']
        HL = HL * self.current_freq_weights['HL']
        HH = HH * self.current_freq_weights['HH']
        
        # Inverse Wavelet Transform
        enhanced = pywt.idwt2((LL, (LH, HL, HH)), self.wavelet_type)
        
        # 크기 조정 (경계 문제로 인한 크기 변경 보정)
        h, w = channel.shape
        enhanced = enhanced[:h, :w]
        
        return enhanced
    
    def update_curriculum_stage(self, epoch: int) -> bool:
        """Curriculum 단계 업데이트 (주파수 가중치 재설정)"""
        target_stage = "dense"
        if epoch >= self.milestones.get("full", 50):
            target_stage = "full"
        elif epoch >= self.milestones.get("expand", 20):
            target_stage = "expand"
        
        if target_stage != self.stage:
            self.stage = target_stage
            self.current_freq_weights = self._get_freq_weights(self.stage)
            self._load_and_analyze_data()
            logging.info(f"🔄 [SignalCurriculum] Stage updated: {self.stage.upper()} | "
                        f"Freq weights: {self.current_freq_weights} | "
                        f"Samples: {len(self.samples)}")
            return True
        return False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if not self.samples:
            # fallback
            return self._get_fallback_item(idx)
        
        # 가중치 기반 샘플링 (선택적)
        if hasattr(self, 'sample_weights') and self.sample_weights:
            sample = np.random.choice(
                self.samples, 
                p=[self.sample_weights[s] for s in self.samples]
            )
            npz_path = os.path.join(self.target_path, sample)
        else:
            npz_name = self.samples[idx]
            npz_path = os.path.join(self.target_path, npz_name)
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            imgs = data['imgs'].astype(np.float32) / 255.0
            masks = data['masks'].astype(np.float32)
            
            # 주파수 강화 적용
            enhanced_imgs = []
            for t in range(len(imgs)):
                enhanced = self._apply_frequency_enhancement(imgs[t])
                enhanced_imgs.append(enhanced)
            
            enhanced_imgs = np.stack(enhanced_imgs, axis=0)
            
            # 차원 조정
            if enhanced_imgs.ndim == 4 and enhanced_imgs.shape[-1] == 3:
                enhanced_imgs = enhanced_imgs.transpose(0, 3, 1, 2)
            
            return {
                "video_id": os.path.basename(npz_path).replace(".npz", ""),
                "images": torch.from_numpy(enhanced_imgs).float(),
                "masks": torch.from_numpy(masks).float(),
                "num_frames": len(enhanced_imgs),
                "freq_features": torch.tensor(list(self.freq_features.get(
                    os.path.basename(npz_path), 
                    [1.0]*len(self.freq_bands)
                ))).float(),
                "curriculum_stage": self.stage
            }
        except Exception as e:
            logging.error(f"❌ Error loading {npz_path}: {e}")
            return self._get_fallback_item(idx)
    
    def _get_fallback_item(self, idx):
        """에러 발생 시 기본 아이템 반환"""
        dummy_img = torch.zeros((1, 3, 256, 256))
        dummy_mask = torch.zeros((1, 256, 256))
        return {
            "video_id": f"dummy_{idx}",
            "images": dummy_img,
            "masks": dummy_mask,
            "num_frames": 1,
            "freq_features": torch.ones(len(self.freq_bands)),
            "curriculum_stage": self.stage
        }


class SignalCurriculumDataset(TorchTrainMixedDataset):
    """
    [SPL Submission Version] 
    Stochastic Temporal Resolution Curriculum
    고정된 stride 대신 확률적 지터링을 통해 시간적 에일리어싱을 억제합니다.
    """
    def __init__(self, *args, max_epochs=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        # 0.2까지 Dense, 0.6까지 Stochastic Expand, 이후 Full
        self.milestones = {"dense": 0.2, "expand": 0.6}

    def get_loader(self, epoch) -> Iterable:
        progress = epoch / self.max_epochs
        
        # 혁신: stride를 고정이 아닌 '확률적 샘플링 모드'로 정의
        if progress < self.milestones["dense"]:
            stage, stride, stochastic = "dense", 1, False
        elif progress < self.milestones["expand"]:
            stage, stride, stochastic = "expand", 2, True  # Stochastic 모드 활성화
        else:
            stage, stride, stochastic = "full", 1, False

        dataloaders = []
        for dataset in self.datasets:
            if hasattr(dataset, "stage"): dataset.stage = stage
            if hasattr(dataset, "temporal_stride"): dataset.temporal_stride = stride
            if hasattr(dataset, "use_stochastic"): dataset.use_stochastic = stochastic # 변수 추가
            
            self._set_dataset_epoch(dataset, epoch)
            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            batch_sampler = BatchSampler(sampler, self.batch_sizes[0], drop_last=self.drop_last)
            dataloaders.append(DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=self.collate_fn))
            
        return MixedDataLoader(dataloaders, self.dataset_prob)
    
    
import numpy as np
import torch
import pywt
import random
from typing import Dict, List, Tuple
from scipy import signal

class NeuroSpectralCurriculumDataset(Dataset):
    """
    Neuro-Inspired Spectral-Temporal Curriculum Learning (NIST-CL)
    
    IEEE SPL 핵심 기여:
    1. 뇌의 시각처리 과정 모방 (V1 → V4 → IT)
    2. 스펙트럼-시간 도메인 하이브리드 커리큘럼
    3. Adaptive Stochastic Resonance 기법
    4. 메모리 기반 주파수 가중치 학습
    """
    
    def __init__(
        self,
        folder: str,
        milestones: Dict[str, int],
        wavelet_type: str = 'db4',
        neuro_layers: List[str] = ['V1', 'V2', 'V4', 'IT'],  # 뇌 영역 시뮬레이션
        use_binaural: bool = True,  # 의사 스테레오 청각 효과
        **kwargs
    ):
        """
        Args:
            folder: 데이터 폴더
            milestones: 커리큘럼 마일스톤
            neuro_layers: 모방할 뇌 시각피질 영역
            use_binaural: 의사 스테레오 효과 (주파수 분리)
        """
        self.base_folder = folder
        self.milestones = milestones
        self.wavelet_type = wavelet_type
        self.neuro_layers = neuro_layers
        self.use_binaural = use_binaural
        
        # 신경학적 상태
        self.neuro_stage = "V1_dense"  # 초기: V1 피질 (단순한 특징)
        self.activation_history = []  # 학습 활성화 기록
        self.memory_weights = {}  # 샘플별 주파수 메모리
        
        # 혁신 1: 주파수-시간 결합 가중치
        self.spectro_temporal_weights = self._init_neuro_weights()
        
        # 혁신 2: Adaptive Stochastic Resonance 파라미터
        self.stochastic_resonance = {
            'noise_level': 0.1,  # 초기 노이즈
            'resonance_freq': [],  # 공진 주파수
            'adaptation_rate': 0.01
        }
        
        self.samples = []
        self._load_and_analyze_data()
        
        logging.info(f"🧠 [NIST-CL] Neuro Stage: {self.neuro_stage} | "
                    f"Samples: {len(self.samples)} | "
                    f"SR Noise: {self.stochastic_resonance['noise_level']:.3f}")
    
    def _init_neuro_weights(self) -> Dict:
        """뇌 영역별 스펙트럼-시간 가중치"""
        return {
            'V1_dense': {'low_freq': 0.9, 'mid_freq': 0.2, 'high_freq': 0.05, 'temporal': 0.1},
            'V2_expand': {'low_freq': 0.7, 'mid_freq': 0.6, 'high_freq': 0.3, 'temporal': 0.3},
            'V4_complex': {'low_freq': 0.5, 'mid_freq': 0.8, 'high_freq': 0.6, 'temporal': 0.6},
            'IT_full': {'low_freq': 0.3, 'mid_freq': 0.9, 'high_freq': 0.9, 'temporal': 0.9}
        }
    
    def _analyze_spectro_temporal_features(self, npz_path: str) -> Dict:
        """스펙트럼-시간 결합 특징 분석"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            imgs = data['imgs']
            
            # 시간 축을 따라 STFT (Short-Time Fourier Transform)
            if imgs.ndim == 4:  # (T, H, W, 3)
                video_series = np.mean(imgs, axis=(1, 2, 3))  # 시간별 평균 밝기
            else:  # (T, H, W)
                video_series = np.mean(imgs, axis=(1, 2))
            
            # STFT 분석
            f, t, Zxx = signal.stft(video_series, fs=30, nperseg=min(16, len(video_series)))
            
            # 주요 주파수 대역 에너지
            low_freq_idx = f < 5  # 저주파 (<5Hz)
            mid_freq_idx = (f >= 5) & (f < 15)  # 중주파
            high_freq_idx = f >= 15  # 고주파
            
            features = {
                'low_freq_energy': np.mean(np.abs(Zxx[low_freq_idx, :])),
                'mid_freq_energy': np.mean(np.abs(Zxx[mid_freq_idx, :])),
                'high_freq_energy': np.mean(np.abs(Zxx[high_freq_idx, :])),
                'temporal_coherence': self._compute_temporal_coherence(Zxx),
                'spectral_entropy': self._compute_spectral_entropy(Zxx),
                'harmonic_ratio': self._compute_harmonic_ratio(Zxx, f)
            }
            
            # 혁신: 의사 스테레오 효과 생성
            if self.use_binaural:
                left_ch, right_ch = self._create_binaural_signature(Zxx, f)
                features['binaural_asymmetry'] = np.abs(left_ch - right_ch).mean()
            
            return features
            
        except Exception as e:
            logging.warning(f"Spectro-temporal analysis failed: {e}")
            return {'low_freq_energy': 1.0, 'mid_freq_energy': 0.5, 'high_freq_energy': 0.1,
                   'temporal_coherence': 0.5, 'spectral_entropy': 1.0, 'harmonic_ratio': 0.5}
    
    def _compute_temporal_coherence(self, Zxx: np.ndarray) -> float:
        """시간적 일관성 계산 (phase coherence)"""
        phases = np.angle(Zxx)
        phase_diff = np.diff(phases, axis=1)
        coherence = np.mean(np.cos(phase_diff))  # 코사인 유사도
        return float(np.clip(coherence, 0, 1))
    
    def _compute_spectral_entropy(self, Zxx: np.ndarray) -> float:
        """스펙트럼 엔트로피"""
        power = np.abs(Zxx) ** 2
        power_sum = np.sum(power, axis=0)
        prob = power / (power_sum + 1e-10)
        entropy = -np.sum(prob * np.log2(prob + 1e-10)) / np.log2(power.shape[0])
        return float(entropy)
    
    def _compute_harmonic_ratio(self, Zxx: np.ndarray, f: np.ndarray) -> float:
        """조화 비율 계산 (정상적인 비디오 vs 병리적 비디오)"""
        # 기본 주파수 추정
        power = np.mean(np.abs(Zxx), axis=1)
        fundamental_idx = np.argmax(power)
        fundamental_freq = f[fundamental_idx]
        
        if fundamental_freq < 1e-6:
            return 0.0
            
        # 고조파 위치 확인
        harmonic_indices = []
        for n in range(2, 6):  # 2차~5차 고조파
            target_freq = fundamental_freq * n
            idx = np.argmin(np.abs(f - target_freq))
            harmonic_indices.append(idx)
        
        # 고조파 강도 합
        harmonic_power = sum(power[i] for i in harmonic_indices if i < len(power))
        total_power = np.sum(power)
        
        return float(harmonic_power / (total_power + 1e-10))
    
    def _create_binaural_signature(self, Zxx: np.ndarray, f: np.ndarray) -> Tuple[float, float]:
        """의사 스테레오 청각 효과 생성"""
        # 저주파는 왼쪽, 고주파는 오른쪽으로 분리
        mid_freq = np.median(f)
        left_ch = np.mean(np.abs(Zxx[f <= mid_freq, :]))
        right_ch = np.mean(np.abs(Zxx[f > mid_freq, :]))
        return left_ch, right_ch
    
    def _apply_neuro_inspired_processing(self, image: np.ndarray, sample_id: str) -> np.ndarray:
        """신경학적 영상 처리 (V1 → IT 과정 모방)"""
        weights = self.spectro_temporal_weights[self.neuro_stage]
        
        # Wavelet 분해
        coeffs = pywt.wavedec2(image, self.wavelet_type, level=3)
        
        # 각 주파수 대역에 가중치 적용
        processed_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # 가장 저주파 (LLL)
                weight = weights['low_freq']
            elif i == 1:  # 중간 주파수
                weight = weights['mid_freq']
            else:  # 고주파
                weight = weights['high_freq']
            
            # 혁신: Adaptive Stochastic Resonance 적용
            if self.stochastic_resonance['noise_level'] > 0:
                noise = np.random.randn(*coeff.shape) * self.stochastic_resonance['noise_level']
                # 공진 주파수 강조
                if hasattr(self, 'resonance_freq') and i in self.stochastic_resonance['resonance_freq']:
                    coeff = coeff * 1.5
                coeff = coeff + noise * weight
            
            processed_coeffs.append(coeff * weight)
        
        # Wavelet 재구성
        processed = pywt.waverec2(processed_coeffs, self.wavelet_type)
        
        # 시간적 처리 (간단한 모션 블러 시뮬레이션)
        if weights['temporal'] > 0.3:
            # 가상의 시간적 통합
            kernel_size = int(3 * weights['temporal'])
            if kernel_size > 1:
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
                from scipy import ndimage
                processed = ndimage.convolve(processed, kernel, mode='reflect')
        
        # 메모리 기반 보정
        if sample_id in self.memory_weights:
            mem_weight = self.memory_weights[sample_id].get('enhancement', 1.0)
            processed = processed * mem_weight
        
        return np.clip(processed, 0, 1)
    
    def _update_stochastic_resonance(self, epoch: int, loss_history: List[float] = None):
        """Adaptive Stochastic Resonance 업데이트"""
        # 점진적 노이즈 감소
        decay = np.exp(-epoch / 50)
        self.stochastic_resonance['noise_level'] = 0.1 * decay
        
        # 손실 기반 적응
        if loss_history and len(loss_history) > 10:
            recent_loss = np.mean(loss_history[-5:])
            prev_loss = np.mean(loss_history[-10:-5])
            
            if recent_loss < prev_loss * 0.95:  # 5% 향상
                # 노이즈 더 줄이기
                self.stochastic_resonance['noise_level'] *= 0.9
            elif recent_loss > prev_loss * 1.05:  # 5% 악화
                # 약간의 노이즈 추가 (탐색 촉진)
                self.stochastic_resonance['noise_level'] = min(0.05, 
                    self.stochastic_resonance['noise_level'] * 1.1)
        
        # 공진 주파수 학습
        if epoch % 10 == 0:
            self._learn_resonance_frequencies()
    
    def _learn_resonance_frequencies(self):
        """학습 데이터에서 공진 주파수 발견"""
        if not hasattr(self, 'freq_features') or len(self.freq_features) < 5:
            return
        
        # 모든 샘플의 주파수 특징 평균
        avg_features = {}
        for band in ['low_freq_energy', 'mid_freq_energy', 'high_freq_energy']:
            values = [feat.get(band, 0) for feat in self.freq_features.values()]
            avg_features[band] = np.mean(values)
        
        # 가장 강한 주파수 대역 찾기
        max_band = max(avg_features, key=avg_features.get)
        
        # 공진 주파수 맵핑
        band_to_level = {
            'low_freq_energy': 0,  # 가장 저주파
            'mid_freq_energy': 1,  # 중간
            'high_freq_energy': 2   # 고주파
        }
        
        if max_band in band_to_level:
            self.stochastic_resonance['resonance_freq'] = [band_to_level[max_band]]
            logging.info(f"🧠 Learned resonance frequency: {max_band}")
    
    def _update_memory_weights(self, sample_id: str, loss: float = None):
        """메모리 기반 가중치 학습"""
        if sample_id not in self.memory_weights:
            self.memory_weights[sample_id] = {
                'enhancement': 1.0,
                'difficulty': 0.5,
                'visit_count': 0
            }
        
        mem = self.memory_weights[sample_id]
        mem['visit_count'] += 1
        
        # 손실 기반 적응
        if loss is not None:
            if loss < 0.1:  # 잘 학습됨
                mem['difficulty'] *= 0.95
            else:  # 어려움
                mem['difficulty'] = min(1.0, mem['difficulty'] * 1.05)
            
            # 어려운 샘플은 더 강하게 처리
            mem['enhancement'] = 1.0 + 0.5 * mem['difficulty']
    
    def update_neuro_stage(self, epoch: int, loss_history: List[float] = None) -> bool:
        """신경학적 커리큘럼 단계 업데이트"""
        progress = epoch / self.milestones.get("full", 50)
        
        old_stage = self.neuro_stage
        
        if progress < 0.2:
            self.neuro_stage = "V1_dense"
        elif progress < 0.6:
            self.neuro_stage = "V2_expand"
        elif progress < 0.9:
            self.neuro_stage = "V4_complex"
        else:
            self.neuro_stage = "IT_full"
        
        # Stochastic Resonance 업데이트
        self._update_stochastic_resonance(epoch, loss_history)
        
        if old_stage != self.neuro_stage:
            logging.info(f"🧠 [NIST-CL] Neuro stage updated: {old_stage} → {self.neuro_stage}")
            return True
        return False
    
    def _load_and_analyze_data(self):
        """데이터 로드 및 특징 분석"""
        self.target_path = os.path.join(self.base_folder, self.neuro_stage.split('_')[0], "uterine_niche")
        if not os.path.exists(self.target_path):
            self.target_path = os.path.join(self.base_folder, self.neuro_stage.split('_')[0])
        
        if os.path.exists(self.target_path):
            all_samples = sorted([f for f in os.listdir(self.target_path) if f.endswith('.npz')])
            
            # 스펙트럼-시간 특징 분석
            self.freq_features = {}
            valid_samples = []
            
            for sample in all_samples:
                npz_path = os.path.join(self.target_path, sample)
                features = self._analyze_spectro_temporal_features(npz_path)
                self.freq_features[sample] = features
                
                # 단계별 필터링
                if self._neuro_sample_filter(features):
                    valid_samples.append(sample)
            
            self.samples = valid_samples
        else:
            self.samples = []
    
    def _neuro_sample_filter(self, features: Dict) -> bool:
        """신경학적 샘플 필터링"""
        if "V1" in self.neuro_stage:
            # 저주파 강조, 단순한 샘플
            return features['low_freq_energy'] > 0.7 and features['spectral_entropy'] < 0.6
        elif "V2" in self.neuro_stage:
            # 중간 복잡도
            return features['mid_freq_energy'] > 0.4 and features['temporal_coherence'] > 0.3
        elif "V4" in self.neuro_stage:
            # 고주파 포함
            return features['high_freq_energy'] > 0.2 and features['harmonic_ratio'] > 0.3
        else:  # IT
            # 모든 샘플
            return True
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if not self.samples:
            return self._get_fallback_item(idx)
        
        # 어려움 기반 샘플링
        if hasattr(self, 'memory_weights') and self.memory_weights:
            difficulties = [self.memory_weights.get(s, {}).get('difficulty', 0.5) 
                          for s in self.samples]
            # 어려운 샘플을 더 자주 샘플링 (학습 촉진)
            probs = np.array(difficulties) / np.sum(difficulties)
            sample_idx = np.random.choice(len(self.samples), p=probs)
            sample = self.samples[sample_idx]
        else:
            sample = self.samples[idx]
        
        npz_path = os.path.join(self.target_path, sample)
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            imgs = data['imgs'].astype(np.float32)
            
            if imgs.max() > 1.0:
                imgs = imgs / 255.0
            
            masks = data['masks'].astype(np.float32)
            
            # 신경학적 처리
            processed_imgs = []
            for t in range(len(imgs)):
                if imgs.ndim == 4:
                    img = imgs[t]
                else:
                    img = imgs[t]
                
                # RGB 채널별 처리
                if img.ndim == 3 and img.shape[-1] == 3:
                    channels = []
                    for c in range(3):
                        processed = self._apply_neuro_inspired_processing(img[..., c], sample)
                        channels.append(processed)
                    processed_img = np.stack(channels, axis=-1)
                else:
                    processed_img = self._apply_neuro_inspired_processing(img, sample)
                
                processed_imgs.append(processed_img)
            
            processed_imgs = np.stack(processed_imgs, axis=0).astype(np.float32)
            
            # 차원 조정
            if processed_imgs.ndim == 4 and processed_imgs.shape[-1] == 3:
                processed_imgs = processed_imgs.transpose(0, 3, 1, 2)
            
            # 특징 벡터 준비
            feats = self.freq_features.get(sample, {})
            feature_vector = [
                feats.get('low_freq_energy', 0.5),
                feats.get('mid_freq_energy', 0.5),
                feats.get('high_freq_energy', 0.2),
                feats.get('temporal_coherence', 0.5),
                feats.get('spectral_entropy', 0.5)
            ]
            
            item = {
                "video_id": sample.replace(".npz", ""),
                "images": torch.from_numpy(processed_imgs).float(),
                "masks": torch.from_numpy(masks).float(),
                "num_frames": len(processed_imgs),
                "neuro_features": torch.tensor(feature_vector).float(),
                "neuro_stage": self.neuro_stage,
                "stochastic_noise": self.stochastic_resonance['noise_level']
            }
            
            # 메모리 업데이트를 위한 샘플 ID 저장
            item['sample_id'] = sample
            
            return item
            
        except Exception as e:
            logging.error(f"❌ [NIST-CL] Error loading {npz_path}: {e}")
            return self._get_fallback_item(idx)
    
    def _get_fallback_item(self, idx):
        dummy_img = torch.zeros((1, 3, 256, 256))
        dummy_mask = torch.zeros((1, 256, 256))
        return {
            "video_id": f"dummy_{idx}",
            "images": dummy_img,
            "masks": dummy_mask,
            "num_frames": 1,
            "neuro_features": torch.ones(5) * 0.5,
            "neuro_stage": self.neuro_stage,
            "stochastic_noise": 0.0,
            "sample_id": f"dummy_{idx}"
        }


class AESCurriculumDataset(Dataset):
    """
    Adaptive Entropy Sampling Curriculum Learning
    
    IEEE SPL 장점:
    1. 정보 이론 기반 (Shannon entropy)
    2. 계산 효율적
    3. 해석 가능성 높음
    4. 의료 영상의 불확실성 정량화
    """
    
    def __init__(self, folder, milestones, entropy_threshold=0.7, **kwargs):
        self.base_folder = folder
        self.milestones = milestones
        self.entropy_threshold = entropy_threshold
        self.adaptive_factor = 1.0
        
        self.stage = "dense"
        self.samples = []
        self.sample_entropies = {}  # 샘플별 엔트로피 저장
        
        self._load_and_compute_entropy()
    
    def _compute_image_entropy(self, image):
        """이미지의 정보 엔트로피 계산"""
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)  # 그레이스케일 변환
        
        # 히스토그램 기반 엔트로피
        hist, _ = np.histogram(image.flatten(), bins=32, range=(0, 1))
        prob = hist / hist.sum()
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        
        # 정규화 (0-1)
        return entropy / 5.0  # max entropy for 32 bins is log2(32) = 5
    
    def _compute_mask_complexity(self, mask):
        """마스크의 형태학적 복잡도 계산"""
        if np.sum(mask) == 0:
            return 0.0
        
        from skimage.measure import perimeter, euler_number
        from scipy import ndimage
        
        # 경계 길이 / 면적 비율
        area = np.sum(mask)
        perim = perimeter(mask)
        boundary_complexity = perim / (area + 1e-10)
        
        # 위상학적 복잡도 (오일러 수)
        labeled, num_features = ndimage.label(mask)
        euler = euler_number(mask)
        
        # 정규화된 복잡도 점수
        complexity = boundary_complexity * 0.1 + abs(euler) * 0.1
        return min(complexity, 1.0)
    
    def _load_and_compute_entropy(self):
        """데이터 로드 및 엔트로피 계산"""
        self.target_path = os.path.join(self.base_folder, self.stage, "uterine_niche")
        if not os.path.exists(self.target_path):
            self.target_path = os.path.join(self.base_folder, self.stage)
        
        if not os.path.exists(self.target_path):
            self.samples = []
            return
        
        all_samples = sorted([f for f in os.listdir(self.target_path) if f.endswith('.npz')])
        
        # 엔트로피 기반 샘플 필터링
        self.samples = []
        self.sample_entropies.clear()
        
        for sample in all_samples:
            try:
                data = np.load(os.path.join(self.target_path, sample), allow_pickle=True)
                imgs = data['imgs']
                masks = data['masks']
                
                # 첫 프레임의 엔트로피 계산
                if len(imgs) > 0:
                    img_entropy = self._compute_image_entropy(imgs[0])
                    mask_complexity = self._compute_mask_complexity(masks[0])
                    
                    # 종합 엔트로피 점수
                    total_entropy = (img_entropy + mask_complexity) / 2
                    self.sample_entropies[sample] = total_entropy
                    
                    # 단계별 필터링
                    if self._entropy_filter(total_entropy):
                        self.samples.append(sample)
                        
            except Exception as e:
                logging.warning(f"Entropy computation failed for {sample}: {e}")
    
    def _entropy_filter(self, entropy):
        """엔트로피 기반 샘플 필터링"""
        if self.stage == "dense":
            return entropy < 0.3 * self.adaptive_factor
        elif self.stage == "expand":
            return entropy < 0.6 * self.adaptive_factor
        else:  # full
            return True
    
    def update_curriculum_stage(self, epoch, training_loss=None):
        """커리큘럼 단계 및 적응형 파라미터 업데이트"""
        target_stage = "dense"
        if epoch >= self.milestones.get("full", 50):
            target_stage = "full"
        elif epoch >= self.milestones.get("expand", 20):
            target_stage = "expand"
        
        # 손실 기반 adaptive factor 조정
        if training_loss is not None:
            if training_loss < 0.1:
                self.adaptive_factor = min(2.0, self.adaptive_factor * 1.05)
            else:
                self.adaptive_factor = max(0.5, self.adaptive_factor * 0.95)
        
        if target_stage != self.stage:
            self.stage = target_stage
            self._load_and_compute_entropy()
            logging.info(f"📊 [AES-CL] Stage: {self.stage} | "
                        f"Adaptive factor: {self.adaptive_factor:.2f} | "
                        f"Samples: {len(self.samples)}")
            return True
        return False
    
    def __getitem__(self, idx):
        # 엔트로피 기반 중요도 샘플링
        if self.sample_entropies and len(self.samples) > 0:
            # 낮은 엔트로피 샘플에 높은 확률 (초기), 높은 엔트로피에 높은 확률 (후기)
            if self.stage == "dense":
                # 낮은 엔트로피 선호
                probs = [1.0/(self.sample_entropies[s] + 0.1) for s in self.samples]
            elif self.stage == "expand":
                # 균등
                probs = [1.0] * len(self.samples)
            else:
                # 높은 엔트로피 선호
                probs = [self.sample_entropies[s] + 0.1 for s in self.samples]
            
            probs = np.array(probs) / sum(probs)
            sample_idx = np.random.choice(len(self.samples), p=probs)
            npz_name = self.samples[sample_idx]
        else:
            npz_name = self.samples[idx % len(self.samples)] if self.samples else None
        
        # 나머지 데이터 로딩은 기존과 동일
        # ... (생략)

import numpy as np
import torch
from scipy import ndimage
from typing import Dict, List
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

# class QuantumResonanceCurriculumDataset(Dataset):
#     """
#     Quantum-Inspired Resonance Curriculum Learning (QIR-CL)
#     기존 MedSAM2CurriculumDataset과 완전 호환 (repeat_factors 포함)
#     """
    
#     def __init__(self, folder, milestones, **kwargs):
#         """
#         기존 MedSAM2CurriculumDataset과 동일한 인터페이스
#         """
#         self.base_folder = folder
#         self.milestones = milestones
        
#         # 기존과 동일한 변수들
#         self.stage = "dense"
#         self.samples = []
#         self.repeat_factors = None  # RepeatFactorWrapper를 위해 필요
        
#         # 양자 개념 (내부적으로만 사용)
#         self._quantum_state = "ground"
#         self._coherence = 0.1
#         self._resonance_freqs = [0.25, 0.5, 0.75, 1.0]
        
#         self._load_stage_data("dense")
        
#         # repeat_factors 초기화 (모든 샘플 동일 가중치)
#         if self.samples:
#             self.repeat_factors = torch.ones(len(self.samples))
        
#         logging.info(f"⚛️ [QIR-CL] Initialized | Stage: {self.stage.upper()} | Samples: {len(self.samples)}")
    
#     def _load_stage_data(self, stage):
#         """기존과 동일한 데이터 로드 로직"""
#         self.stage = stage
        
#         # 경로: base_folder / stage / uterine_niche
#         self.target_path = os.path.join(self.base_folder, self.stage, "uterine_niche")
        
#         if not os.path.exists(self.target_path):
#             # 폴더가 없으면 상위 단계 폴더 참조
#             self.target_path = os.path.join(self.base_folder, self.stage)
        
#         if os.path.exists(self.target_path):
#             # 모든 NPZ 파일 수집
#             all_samples = sorted([f for f in os.listdir(self.target_path) if f.endswith('.npz')])
            
#             # 디버그: 폴더 내용 확인
#             logging.info(f"📁 Path: {self.target_path}")
#             logging.info(f"📁 Files in directory: {os.listdir(self.target_path)[:5] if os.path.exists(self.target_path) else 'Directory not found'}")
            
#             if not all_samples:
#                 logging.warning(f"No .npz files found in {self.target_path}")
#                 self.samples = []
#             else:
#                 # 양자 상태에 따른 샘플 필터링
#                 self.samples = self._quantum_filter_samples(all_samples)
                
#                 # 공명 주파수 업데이트
#                 self._update_resonance_frequencies()
#         else:
#             logging.warning(f"Target path does not exist: {self.target_path}")
#             self.samples = []
    
#     def _quantum_filter_samples(self, all_samples):
#         """양자 상태에 따른 샘플 필터링"""
#         if self._quantum_state == "ground" and len(all_samples) > 5:
#             # 초기: 작은 파일만 선택 (단순한 샘플)
#             filtered = []
#             for sample in all_samples:
#                 try:
#                     path = os.path.join(self.target_path, sample)
#                     size_mb = os.path.getsize(path) / (1024 * 1024)  # MB 단위
#                     if size_mb < 5.0:  # 5MB 미만의 작은 파일
#                         filtered.append(sample)
#                 except:
#                     filtered.append(sample)
            
#             # 최소 3개는 보장
#             if len(filtered) < 3:
#                 filtered = all_samples[:min(10, len(all_samples))]
            
#             return filtered
        
#         elif self._quantum_state == "excited" and len(all_samples) > 10:
#             # 중간: 중간 크기 샘플
#             return all_samples[:len(all_samples)//2]
        
#         else:
#             # 모든 샘플
#             return all_samples
    
#     def _update_resonance_frequencies(self):
#         """공명 주파수 업데이트"""
#         if not self.samples:
#             self._resonance_freqs = [0.25, 0.5, 0.75, 1.0]
#             return
        
#         try:
#             # 무작위 샘플로 주파수 분석
#             sample_idx = np.random.randint(0, len(self.samples))
#             sample_path = os.path.join(self.target_path, self.samples[sample_idx])
            
#             data = np.load(sample_path, allow_pickle=True)
#             img = data['imgs'][0]
            
#             # 그레이스케일 변환
#             if img.ndim == 3 and img.shape[-1] == 3:
#                 img = np.mean(img, axis=2)
            
#             # 간단한 FFT 분석
#             fft = np.fft.fft2(img)
#             magnitude = np.abs(np.fft.fftshift(fft))
            
#             h, w = magnitude.shape
#             energies = [
#                 np.mean(magnitude[:h//2, :w//2]),  # 저주파
#                 np.mean(magnitude[:h//2, w//2:]),  # 중저주파
#                 np.mean(magnitude[h//2:, :w//2]),  # 중고주파
#                 np.mean(magnitude[h//2:, w//2:])   # 고주파
#             ]
            
#             max_energy = max(energies) if max(energies) > 0 else 1.0
#             self._resonance_freqs = [e/max_energy for e in energies]
            
#         except Exception as e:
#             logging.warning(f"Resonance frequency update failed: {e}")
#             self._resonance_freqs = [0.25, 0.5, 0.75, 1.0]
    
#     def _apply_quantum_enhancement(self, image):
#         """양자 공명 기반 이미지 향상"""
#         if image.ndim == 3 and image.shape[-1] == 3:
#             # RGB: 채널별 처리
#             enhanced_channels = []
#             for c in range(3):
#                 channel = image[:, :, c]
#                 enhanced = self._enhance_channel(channel)
#                 enhanced_channels.append(enhanced)
#             result = np.stack(enhanced_channels, axis=-1)
#         else:
#             result = self._enhance_channel(image)
        
#         return np.clip(result, 0, 1)
    
#     def _enhance_channel(self, channel):
#         """단일 채널 양자 향상"""
#         # FFT 변환
#         fft = np.fft.fft2(channel)
#         fshift = np.fft.fftshift(fft)
#         magnitude = np.abs(fshift)
#         phase = np.angle(fshift)
        
#         rows, cols = channel.shape
#         crow, ccol = rows // 2, cols // 2
        
#         # 양자 상태에 따른 증폭
#         if self._quantum_state == "ground":
#             # 저주파 강조
#             radius = 30
#             mask = np.zeros((rows, cols))
#             y, x = np.ogrid[:rows, :cols]
#             mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
#             magnitude[mask_area] *= 1.3
            
#         elif self._quantum_state == "excited":
#             # 중주파 강조
#             inner_radius = 20
#             outer_radius = 60
#             y, x = np.ogrid[:rows, :cols]
#             dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
#             mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
#             magnitude[mask] *= 1.5
            
#         elif self._quantum_state == "coherent":
#             # 고주파 강조
#             radius = 30
#             y, x = np.ogrid[:rows, :cols]
#             mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
#             magnitude[~mask_area] *= 1.4
            
#         else:  # resonant
#             # 모든 주파수 균형
#             magnitude *= 1.2
        
#         # 위상 조정 (결맞음도에 따라)
#         phase_shift = self._coherence * 0.5
#         phase = phase + phase_shift
        
#         # 역변환
#         enhanced = magnitude * np.exp(1j * phase)
#         f_ishift = np.fft.ifftshift(enhanced)
#         img_back = np.fft.ifft2(f_ishift)
#         result = np.real(img_back)
        
#         return result
    
#     # RepeatFactorWrapper를 위한 메서드들
#     def set_epoch(self, epoch):
#         """RepeatFactorWrapper 호환성"""
#         if hasattr(self, '_set_dataset_epoch'):
#             self._set_dataset_epoch(self, epoch)
    
#     @property
#     def epoch(self):
#         """RepeatFactorWrapper 호환성"""
#         return getattr(self, '_epoch', 0)
    
#     @epoch.setter
#     def epoch(self, value):
#         """RepeatFactorWrapper 호환성"""
#         self._epoch = value
    
#     def update_curriculum_stage(self, epoch):
#         """
#         기존 MedSAM2CurriculumDataset과 동일한 인터페이스
#         Returns: bool (stage가 변경되었는지 여부)
#         """
#         # 스테이지 결정
#         target_stage = "dense"
#         if epoch >= self.milestones.get("full", 50):
#             target_stage = "full"
#         elif epoch >= self.milestones.get("expand", 20):
#             target_stage = "expand"
        
#         # 양자 상태 업데이트
#         old_state = self._quantum_state
#         if epoch < 10:
#             self._quantum_state = "ground"
#             self._coherence = 0.1
#         elif epoch < 25:
#             self._quantum_state = "excited"
#             self._coherence = 0.4
#         elif epoch < 40:
#             self._quantum_state = "coherent"
#             self._coherence = 0.7
#         else:
#             self._quantum_state = "resonant"
#             self._coherence = 0.9
        
#         # 스테이지 또는 양자 상태 변경 확인
#         stage_changed = target_stage != self.stage
#         quantum_state_changed = old_state != self._quantum_state
        
#         if stage_changed or quantum_state_changed:
#             self._load_stage_data(target_stage)
            
#             # repeat_factors 업데이트
#             if self.samples:
#                 self.repeat_factors = torch.ones(len(self.samples))
#                 # 양자 상태에 따른 가중치 조정
#                 if self._quantum_state == "resonant":
#                     # 고급 단계: 복잡한 샘플에 더 높은 가중치
#                     self.repeat_factors = self._calculate_quantum_weights()
            
#             logging.info(f"🔄 [QIR-CL] Stage: {self.stage.upper()} | "
#                         f"Quantum State: {self._quantum_state} | "
#                         f"Coherence: {self._coherence:.2f} | "
#                         f"Samples: {len(self.samples)}")
#             return True
        
#         return False
    
#     def _calculate_quantum_weights(self):
#         """양자 가중치 계산"""
#         if not self.samples:
#             return torch.ones(0)
        
#         weights = []
#         for sample in self.samples:
#             try:
#                 path = os.path.join(self.target_path, sample)
#                 data = np.load(path, allow_pickle=True)
#                 img = data['imgs'][0]
                
#                 # 이미지 엔트로피로 복잡도 추정
#                 if img.ndim == 3:
#                     img = np.mean(img, axis=2)
                
#                 hist, _ = np.histogram(img.flatten(), bins=32)
#                 prob = hist / hist.sum()
#                 entropy = -np.sum(prob * np.log(prob + 1e-10))
                
#                 # 엔트로피가 높을수록 높은 가중치
#                 weight = 1.0 + 0.5 * (entropy / np.log(32))  # 1.0 ~ 1.5
#                 weights.append(weight)
#             except:
#                 weights.append(1.0)
        
#         return torch.tensor(weights, dtype=torch.float32)
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         if not self.samples:
#             return self._get_fallback_item(idx)
        
#         # 샘플 인덱스 조정
#         sample_idx = idx % len(self.samples)
#         npz_name = self.samples[sample_idx]
#         npz_path = os.path.join(self.target_path, npz_name)
        
#         try:
#             data = np.load(npz_path, allow_pickle=True)
#             imgs = data['imgs']    # (T, H, W, 3)
#             masks = data['masks']  # (T, H, W)
            
#             # 정규화
#             if imgs.max() > 1.0:
#                 imgs = imgs.astype(np.float32) / 255.0
#             else:
#                 imgs = imgs.astype(np.float32)
            
#             masks = masks.astype(np.float32)
            
#             # 양자 향상 적용
#             enhanced_imgs = []
#             for t in range(len(imgs)):
#                 enhanced = self._apply_quantum_enhancement(imgs[t])
#                 enhanced_imgs.append(enhanced)
            
#             enhanced_imgs = np.stack(enhanced_imgs, axis=0)
            
#             # 차원 조정: (T, H, W, 3) -> (T, 3, H, W)
#             if enhanced_imgs.ndim == 4 and enhanced_imgs.shape[-1] == 3:
#                 enhanced_imgs = enhanced_imgs.transpose(0, 3, 1, 2)
            
#             return {
#                 "video_id": npz_name.replace(".npz", ""),
#                 "images": torch.from_numpy(enhanced_imgs).float(),
#                 "masks": torch.from_numpy(masks).float(),
#                 "num_frames": len(imgs)
#             }
            
#         except Exception as e:
#             logging.error(f"❌ [QIR-CL] Error loading {npz_path}: {e}")
#             # 다음 샘플 시도
#             next_idx = (sample_idx + 1) % len(self.samples)
#             return self.__getitem__(next_idx)
    
#     def _get_fallback_item(self, idx):
#         """에러 시 폴백 아이템"""
#         dummy_img = torch.zeros((1, 3, 256, 256))
#         dummy_mask = torch.zeros((1, 256, 256))
#         return {
#             "video_id": f"dummy_{idx}",
#             "images": dummy_img,
#             "masks": dummy_mask,
#             "num_frames": 1
#         }



#######################################################