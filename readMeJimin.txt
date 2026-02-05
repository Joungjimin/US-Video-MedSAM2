conda activate /media/sde/zhengzhimin/environment/medsam2

CUDA_VISIBLE_DEVICES=0 python training/train.py \
    -c configs/GFTE_3 \
    --num-gpus 1 \
    --num-nodes 1

<val>
python /media/sde/zhengzhimin/MedSAM2/medsam2_infer_MRI.py   --ckpt /media/sde/zhengzhimin/MedSAM2/work_dirsRe/Quantum/D3_Quantum_GFTE_TempLoss/checkpoints/checkpoint_70.pt  --data_root /media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/MRI/uterine_niche1


<dataLoader>
/training/dataset/sam2_datasets.py	> /media/sde/zhengzhimin/MedSAM2/training/dataset/sam2_datasets_module.py 로

yaml에 TorchTrainMixedDataset 를 MedSAM2CurriculumDataset


<loss>
loss_fns_addLoss.py 를 loss_fns.py로 바꿔서 


<module>
sam2/modeling/sam2_base.py    >	self.temporalVideo 로 조절


