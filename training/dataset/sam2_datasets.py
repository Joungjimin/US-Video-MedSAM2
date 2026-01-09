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
class SignalCurriculumDataset(TorchTrainMixedDataset):
    """
    [SPL Version] Progressive Temporal Resolution Curriculum
    전체 학습 에포크를 기준으로 샘플링 주파수(Stride)를 동적으로 조절합니다.
    """
    def __init__(self, *args, max_epochs=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        # 단계 설정: 20%까지 Dense, 50%까지 Expand(Stride 2), 이후 Full
        self.milestones = {"dense": 0.2, "expand": 0.5}

    def get_loader(self, epoch) -> Iterable:
        # 1. 현재 진행률에 따른 신호 처리 단계 및 Stride 계산
        progress = epoch / self.max_epochs
        
        if progress < self.milestones["dense"]:
            current_stage, current_stride = "dense", 1
        elif progress < self.milestones["expand"]:
            current_stage, current_stride = "expand", 2 # 신호를 성기게 샘플링 (Aliasing 방지)
        else:
            current_stage, current_stride = "full", 1    # 신호 전체 복구

        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(zip(self.datasets, self.batch_sizes)):
            # 2. 하위 데이터셋 클래스에 스테이지와 스트라이드 주입 (가장 중요)
            if hasattr(dataset, "stage"):
                dataset.stage = current_stage
            if hasattr(dataset, "temporal_stride"):
                dataset.temporal_stride = current_stride
            
            # 로그 기록 (저널 데이터용)
            if epoch % 5 == 0:
                logging.info(f"🚀 [Signal Curriculum] Epoch {epoch}: Stage={current_stage.upper()}, Stride={current_stride}")

            self._set_dataset_epoch(dataset, epoch)
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
#######################################################