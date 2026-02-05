import os
import cv2
import numpy as np

SRC_ROOT = "/media/sde/zhengzhimin/MedSAM2/abl"
DST_ROOT = "/media/sde/zhengzhimin/MedSAM2/abl/abl_cropped"

CROP_SIZE = 256  # 필요하면 384 등으로 변경

os.makedirs(DST_ROOT, exist_ok=True)

for model_name in os.listdir(SRC_ROOT):
    model_dir = os.path.join(SRC_ROOT, model_name)
    if not os.path.isdir(model_dir):
        continue

    out_dir = os.path.join(DST_ROOT, model_name)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0

    for fname in os.listdir(model_dir):
        if not fname.endswith(".png"):
            continue

        src_path = os.path.join(model_dir, fname)
        mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape

        ys, xs = np.where(mask > 0)

        if len(xs) > 0:
            # ✅ mask 있는 경우 → 객체 중심
            cx = (xs.min() + xs.max()) // 2
            cy = (ys.min() + ys.max()) // 2
        else:
            # ✅ mask 없는 경우 → 이미지 중앙
            cx = w // 2
            cy = h // 2

        half = CROP_SIZE // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        # 경계 보정 (항상 CROP_SIZE 유지)
        if (x2 - x1) < CROP_SIZE:
            if x1 == 0:
                x2 = min(w, x1 + CROP_SIZE)
            elif x2 == w:
                x1 = max(0, x2 - CROP_SIZE)

        if (y2 - y1) < CROP_SIZE:
            if y1 == 0:
                y2 = min(h, y1 + CROP_SIZE)
            elif y2 == h:
                y1 = max(0, y2 - CROP_SIZE)

        crop = mask[y1:y2, x1:x2]

        dst_path = os.path.join(out_dir, fname)
        cv2.imwrite(dst_path, crop)
        saved += 1

    print(f"[DONE] {model_name} → saved {saved} images")
