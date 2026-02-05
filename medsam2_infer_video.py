# import argparse
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from sam2.build_sam import build_sam2_video_predictor
# import shutil
# from collections import defaultdict
# import csv

# # ===============================
# # CONFIG
# # ===============================
# ALL_CLASSES = [1, 2]   # 모든 비디오에서 반드시 출력할 클래스


# # ===============================
# # Fair Segmentation Metrics
# # ===============================
# class FairSegMetrics:
#     def __init__(self, thr=0.5, eps=1e-6):
#         self.thr = thr
#         self.eps = eps

#     def __call__(self, logits, target):
#         prob = torch.sigmoid(logits)
#         pred = (prob > self.thr).float()
#         gt = (target > 0.5).float()

#         if gt.shape != pred.shape:
#             gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")

#         p = pred.flatten(1).cpu()
#         t = gt.flatten(1).cpu()

#         inter = (p * t).sum(-1)
#         union = (p + t).clamp(max=1).sum(-1)

#         dice = (2 * inter + self.eps) / (p.sum(-1) + t.sum(-1) + self.eps)
#         iou = (inter + self.eps) / (union + self.eps)
#         acc = (p == t).float().mean(-1)

#         return dice, iou, acc


# # ===============================
# # Visualization helpers
# # ===============================
# def save_mask(mask, path):
#     mask = (mask > 0).astype(np.uint8) * 255
#     Image.fromarray(mask).save(path)

# def save_overlay(img, mask, path, color=(255, 0, 0), alpha=0.5):
#     if img.ndim == 2:
#         img = np.stack([img]*3, axis=-1)
#     elif img.shape[-1] == 1:
#         img = np.repeat(img, 3, axis=-1)

#     img = img.astype(np.float32)
#     overlay = img.copy()
#     m = mask.astype(bool)
#     overlay[m] = alpha * np.array(color) + (1 - alpha) * overlay[m]
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#     Image.fromarray(overlay).save(path)


# # ===============================
# # Video evaluation + inference saving
# # ===============================
# @torch.inference_mode()
# def evaluate_video(predictor, npz_path, out_dir):
#     data = np.load(npz_path)
#     imgs, gts = data["imgs"], data["gts"]
#     video_name = os.path.splitext(os.path.basename(npz_path))[0]

#     # 1) dump frames
#     tmp_dir = os.path.join(out_dir, "tmp", video_name)
#     os.makedirs(tmp_dir, exist_ok=True)

#     for i, img in enumerate(imgs):
#         if img.dtype != np.uint8:
#             img = (img * 255).astype(np.uint8) if img.max() <= 1 else np.clip(img, 0, 255).astype(np.uint8)
#         Image.fromarray(img).save(os.path.join(tmp_dir, f"{i:05d}.jpg"))

#     state = predictor.init_state(video_path=tmp_dir)
#     metric_fn = FairSegMetrics()

#     # 2) find first GT frame
#     start = -1
#     for t in range(len(gts)):
#         if np.any(gts[t] > 0):
#             start = t
#             for oid in np.unique(gts[t]):
#                 if oid > 0:
#                     predictor.add_new_mask(state, t, int(oid), (gts[t] == oid))
#             break

#     if start == -1:
#         shutil.rmtree(tmp_dir)
#         return None, None

#     vis_root = os.path.join(out_dir, "vis", video_name)
#     os.makedirs(vis_root, exist_ok=True)

#     records = []  # [class_id, dice, iou, acc]

#     # 3) propagate
#     for fidx, obj_ids, logits in predictor.propagate_in_video(state, start):
#         gt = gts[fidx]
#         if gt.sum() == 0:
#             continue

#         gt_list = [torch.from_numpy(gt == oid).float() for oid in obj_ids]
#         gt_tensor = torch.stack(gt_list).to(logits.device).unsqueeze(1)

#         dice, iou, acc = metric_fn(logits, gt_tensor)

#         prob = torch.sigmoid(logits).cpu().numpy()
#         pred = prob > 0.5

#         img = imgs[fidx]
#         if img.dtype != np.uint8:
#             img = (img * 255).astype(np.uint8)

#         frame_dir = os.path.join(vis_root, f"frame_{fidx:05d}")
#         os.makedirs(frame_dir, exist_ok=True)

#         for i, oid in enumerate(obj_ids):
#             d, j, a = dice[i].item(), iou[i].item(), acc[i].item()
#             records.append([int(oid), d, j, a])

#             save_mask(pred[i, 0], os.path.join(frame_dir, f"class{oid}_pred.png"))
#             save_mask(gt == oid, os.path.join(frame_dir, f"class{oid}_gt.png"))
#             save_overlay(img, pred[i, 0], os.path.join(frame_dir, f"class{oid}_overlay.png"))

#     shutil.rmtree(tmp_dir)
#     return video_name, records


# # ===============================
# # Main
# # ===============================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt", required=True)
#     parser.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
#     parser.add_argument("--data_root", required=True)
#     parser.add_argument("--out_root", default="./eval_results_fair")
#     args = parser.parse_args()

#     exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
#     out_dir = os.path.join(args.out_root, exp_name)
#     os.makedirs(out_dir, exist_ok=True)

#     predictor = build_sam2_video_predictor(args.cfg, ckpt_path=None)
#     sd = torch.load(args.ckpt, map_location="cpu")
#     sd = sd.get("model", sd)
#     predictor.load_state_dict(sd, strict=False)
#     predictor.to("cuda")

#     csv_path = os.path.join(out_dir, "evaluation_summary.csv")

#     global_video_means = defaultdict(list)

#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Video", "Class", "Mean_Dice", "Mean_IoU", "Mean_PixelAcc", "N"])

#         for fn in sorted(os.listdir(args.data_root)):
#             if not fn.endswith(".npz"):
#                 continue

#             vname, recs = evaluate_video(
#                 predictor,
#                 os.path.join(args.data_root, fn),
#                 out_dir
#             )
#             if recs is None:
#                 continue

#             video_class_stats = defaultdict(list)
#             for cid, d, j, a in recs:
#                 video_class_stats[cid].append([d, j, a])

#             for cid in ALL_CLASSES:
#                 if cid in video_class_stats:
#                     arr = np.array(video_class_stats[cid], dtype=float)
#                     mean_vals = arr.mean(axis=0)
#                     n = len(arr)
#                 else:
#                     mean_vals = np.array([0.0, 0.0, 0.0])
#                     n = 0

#                 writer.writerow([
#                     vname,
#                     cid,
#                     f"{mean_vals[0]:.4f}",
#                     f"{mean_vals[1]:.4f}",
#                     f"{mean_vals[2]:.4f}",
#                     n
#                 ])

#                 global_video_means[cid].append(mean_vals)

#         # global (video-balanced)
#         for cid in ALL_CLASSES:
#             arr = np.array(global_video_means[cid], dtype=float)
#             writer.writerow([
#                 "ALL",
#                 cid,
#                 f"{arr[:,0].mean():.4f}",
#                 f"{arr[:,1].mean():.4f}",
#                 f"{arr[:,2].mean():.4f}",
#                 len(arr)
#             ])

#     print("\n✅ Done")
#     print(f"📄 CSV saved to: {csv_path}")
#     print(f"🖼  Inference images saved to: {out_dir}/vis")


# if __name__ == "__main__":
#     main()


import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import shutil
from collections import defaultdict
import csv
import random

# ===============================
# 🔥 1. GLOBAL DETERMINISM SETUP
# ===============================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===============================
# CONFIG
# ===============================
ALL_CLASSES = [1, 2]

# ===============================
# Fair Segmentation Metrics
# ===============================
class FairSegMetrics:
    def __init__(self, thr=0.5, eps=1e-6):
        self.thr = thr
        self.eps = eps

    def __call__(self, logits, target):
        prob = torch.sigmoid(logits)
        pred = (prob > self.thr).float()
        gt = (target > 0.5).float()

        if gt.shape != pred.shape:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="nearest")

        p = pred.flatten(1).cpu()
        t = gt.flatten(1).cpu()

        inter = (p * t).sum(-1)
        union = (p + t).clamp(max=1).sum(-1)

        dice = (2 * inter + self.eps) / (p.sum(-1) + t.sum(-1) + self.eps)
        iou = (inter + self.eps) / (union + self.eps)
        acc = (p == t).float().mean(-1)

        return dice, iou, acc

# ===============================
# Visualization helpers
# ===============================
def save_mask(mask, path):
    mask = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)

def save_overlay(img, mask, path, color=(255, 0, 0), alpha=0.5):
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img.astype(np.float32)
    overlay = img.copy()
    m = mask.astype(bool)
    overlay[m] = alpha * np.array(color) + (1 - alpha) * overlay[m]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)

# ===============================
# Video evaluation + inference
# ===============================
@torch.inference_mode()
def evaluate_video(predictor, npz_path, out_dir):
    data = np.load(npz_path)
    #imgs, gts = data["imgs"], data["gts"]
    imgs = data["imgs"]
    gts = data["gts"] if "gts" in data.files else None
    video_name = os.path.splitext(os.path.basename(npz_path))[0]

    tmp_dir = os.path.join(out_dir, "tmp", video_name)
    os.makedirs(tmp_dir, exist_ok=True)

    for i, img in enumerate(imgs):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(tmp_dir, f"{i:05d}.jpg"))

    # 🔥 2. INIT STATE (deterministic)
    state = predictor.init_state(video_path=tmp_dir)
    metric_fn = FairSegMetrics()

    # 🔥 3. FIXED FIRST GT FRAME (earliest, deterministic)
    start = -1
    if gts is not None:
        for t in range(len(gts)):
            if np.any(gts[t] > 0):
                start = t
                # 🔥 4. SORT OBJECT IDS
                for oid in sorted(np.unique(gts[t])):
                    if oid > 0:
                        predictor.add_new_mask(
                            state,
                            t,
                            int(oid),
                            (gts[t] == oid)
                        )
                break
    else:
        # GT 없는 MRI → metric 계산 스킵
        pass

    if start == -1:
        shutil.rmtree(tmp_dir)
        return None, None

    vis_root = os.path.join(out_dir, "vis", video_name)
    os.makedirs(vis_root, exist_ok=True)

    records = []

    for fidx, obj_ids, logits in predictor.propagate_in_video(state, start):
        gt = gts[fidx]
        # if gt.sum() == 0:
        #     continue

        gt_list = [torch.from_numpy(gt == oid).float() for oid in obj_ids]
        gt_tensor = torch.stack(gt_list).to(logits.device).unsqueeze(1)

        dice, iou, acc = metric_fn(logits, gt_tensor)

        prob = torch.sigmoid(logits).cpu().numpy()
        pred = prob > 0.5

        img = imgs[fidx]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        frame_dir = os.path.join(vis_root, f"frame_{fidx:05d}")
        os.makedirs(frame_dir, exist_ok=True)

        for i, oid in enumerate(obj_ids):
            d, j, a = dice[i].item(), iou[i].item(), acc[i].item()
            records.append([int(oid), d, j, a])

            save_mask(pred[i, 0], os.path.join(frame_dir, f"class{oid}_pred.png"))
            save_mask(gt == oid, os.path.join(frame_dir, f"class{oid}_gt.png"))
            save_overlay(img, pred[i, 0], os.path.join(frame_dir, f"class{oid}_overlay.png"))

    shutil.rmtree(tmp_dir)
    return video_name, records

# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", default="./eval_results_fair_MRI")
    args = parser.parse_args()

    exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    out_dir = os.path.join(args.out_root, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    predictor = build_sam2_video_predictor(args.cfg, ckpt_path=None)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    predictor.load_state_dict(sd, strict=False)

    predictor.to("cuda")
    predictor.eval()   # 🔥 5. EVAL MODE FIXED

    csv_path = os.path.join(out_dir, "evaluation_summary.csv")
    global_video_means = defaultdict(list)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Class", "Mean_Dice", "Mean_IoU", "Mean_PixelAcc", "N"])

        for fn in sorted(os.listdir(args.data_root)):
            if not fn.endswith(".npz"):
                continue

            vname, recs = evaluate_video(
                predictor,
                os.path.join(args.data_root, fn),
                out_dir
            )
            if recs is None:
                continue

            video_class_stats = defaultdict(list)
            for cid, d, j, a in recs:
                video_class_stats[cid].append([d, j, a])

            for cid in ALL_CLASSES:
                if cid in video_class_stats:
                    arr = np.array(video_class_stats[cid])
                    mean_vals = arr.mean(axis=0)
                    n = len(arr)
                else:
                    mean_vals = np.zeros(3)
                    n = 0

                writer.writerow([
                    vname,
                    cid,
                    f"{mean_vals[0]:.4f}",
                    f"{mean_vals[1]:.4f}",
                    f"{mean_vals[2]:.4f}",
                    n
                ])

                global_video_means[cid].append(mean_vals)

        for cid in ALL_CLASSES:
            arr = np.array(global_video_means[cid])
            writer.writerow([
                "ALL",
                cid,
                f"{arr[:,0].mean():.4f}",
                f"{arr[:,1].mean():.4f}",
                f"{arr[:,2].mean():.4f}",
                len(arr)
            ])

    print("\n✅ Deterministic evaluation done")
    print(f"📄 CSV saved to: {csv_path}")
    print(f"🖼  Visualizations saved to: {out_dir}/vis")

if __name__ == "__main__":
    main()