# # for i in range(4):
# #     import os
# #     import nibabel as nib
# #     import numpy as np

# #     IMG_DIR = f"/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/data{i+1}/train/images"
# #     LBL_DIR = f"/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/data{i+1}/train/labels"
# #     OUT_DIR = f"/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/preprocessed{i+1}/train/uterine_niche"

# #     os.makedirs(OUT_DIR, exist_ok=True)
# #     img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".nii.gz")])

# #     for fname in img_files:
# #         vid = fname.replace(".nii.gz", "")
# #         img_path = os.path.join(IMG_DIR, fname)
# #         lbl_path = os.path.join(LBL_DIR, fname)

# #         if not os.path.exists(lbl_path): continue

# #         # 1. 원본 로드
# #         img_data = nib.load(img_path).get_fdata()
# #         mask_data = nib.load(lbl_path).get_fdata()

# #         # 2. 스마트 축 정렬 (640이 아닌 축을 프레임으로 간주하여 맨 앞으로 이동)
# #         # 현재 사용자님의 shape이 (640, 100, 640)이라면 f_axis는 1이 됩니다.
# #         current_shape = img_data.shape
# #         f_axis = -1
# #         for i, dim in enumerate(current_shape):
# #             if dim != 640:
# #                 f_axis = i
# #                 break
        
# #         if f_axis == 1: # (640, 100, 640) -> (100, 640, 640)
# #             img = img_data.transpose(1, 0, 2)
# #             mask = mask_data.transpose(1, 0, 2)
# #         elif f_axis == 2: # (640, 640, 100) -> (100, 640, 640)
# #             img = img_data.transpose(2, 0, 1)
# #             mask = mask_data.transpose(2, 0, 1)
# #         else:
# #             img = img_data
# #             mask = mask_data

# #         # 3. 데이터 타입 및 값 고정
# #         img = img.astype(np.float64)
# #         mask = mask.astype(np.int64)

# #         # 4. 저장
# #         out_path = os.path.join(OUT_DIR, f"{vid}.npz")
# #         np.savez_compressed(out_path, imgs=img, gts=mask)
        
# #         print(f"[FIXED] {vid} | Final Shape: {img.shape}")

# #     print("\n✅ 모든 데이터가 (Frames, 640, 640) 형식으로 수정되었습니다.")
# # # import numpy as np

# # # # 기존 잘 되는 파일
# # # old_data = np.load("/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/preprocessed1/train/uterine_niche/1.npz")
# # # # 새로 만든 파일
# # # new_data = np.load("/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/preprocessed2/train/uterine_niche/2.npz")

# # # print("--- [Keys Check] ---")
# # # print(f"Old keys: {list(old_data.keys())}")
# # # print(f"New keys: {list(new_data.keys())}")

# # # for key in old_data.keys():
# # #     if key in new_data:
# # #         print(f"\n--- [Key: {key}] ---")
# # #         print(f"Old - shape: {old_data[key].shape}, dtype: {old_data[key].dtype}")
# # #         print(f"New - shape: {new_data[key].shape}, dtype: {new_data[key].dtype}")
# # #         print(f"Old - Range: [{old_data[key].min()}, {old_data[key].max()}]")
# # #         print(f"New - Range: [{new_data[key].min()}, {new_data[key].max()}]")




# import os
# import numpy as np

# BASE_DIR = "/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data"

# for d in range(1, 5):
#     SRC_DIR = f"{BASE_DIR}/preprocessed{d}/val/uterine_niche"
#     OUT_DIR = f"{BASE_DIR}/preprocessed{d}_half_even/val/uterine_niche"

#     os.makedirs(OUT_DIR, exist_ok=True)

#     npz_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith(".npz")])

#     print(f"\n==============================")
#     print(f"📁 Dataset preprocessed{d}")
#     print(f"==============================")

#     for fname in npz_files:
#         src_path = os.path.join(SRC_DIR, fname)
#         out_path = os.path.join(OUT_DIR, fname)

#         data = np.load(src_path)
#         imgs = data["imgs"]   # (F, 640, 640)
#         gts  = data["gts"]

#         total_frames = imgs.shape[0]

#         # ✅ 짝수 프레임 인덱스만
#         even_indices = list(range(0, total_frames, 2))

#         imgs_even = imgs[even_indices]
#         gts_even  = gts[even_indices]

#         np.savez_compressed(
#             out_path,
#             imgs=imgs_even,
#             gts=gts_even
#         )

#         # 🔍 출력
#         print(f"\n▶ {fname}")
#         print(f"  - Original frames: {total_frames}")
#         print(f"  - Kept frames (even indices):")
#         print(f"    {even_indices}")
#         print(f"  - Total kept: {len(even_indices)}")

#     print(f"\n✅ Finished preprocessed{d}")


import nibabel as nib
import numpy as np

nii_path = "/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/data1/val/images/7.nii.gz"

img = nib.load(nii_path)
data = img.get_fdata()

print(f"📦 File: {nii_path}")
print(f"🧠 Original shape: {data.shape}")

# 프레임 축 자동 판별 (640이 아닌 축)
f_axis = None
for i, dim in enumerate(data.shape):
    if dim != 640:
        f_axis = i
        break

if f_axis is None:
    print("❌ 프레임 축을 찾을 수 없음")
else:
    num_frames = data.shape[f_axis]
    print(f"🎞  Detected frame axis: {f_axis}")
    print(f"🎞  Total frames: {num_frames}\n")

    for idx in range(num_frames):
        print(f"frame_{idx:05d}")
