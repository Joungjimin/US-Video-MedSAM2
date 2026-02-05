# import os
# from collections import defaultdict

# root = "/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/MRI/ZGQSSWMRSM_20220125_102618_289000"

# series_info = defaultdict(list)

# for dirpath, dirnames, filenames in os.walk(root):
#     ima_files = [f for f in filenames if f.endswith(".IMA")]
#     if ima_files:
#         series_name = os.path.basename(dirpath)
#         series_info[series_name] = sorted(ima_files)

# for series, files in series_info.items():
#     print("=" * 60)
#     print("Series folder :", series)
#     print("Num slices    :", len(files))
#     print("Example file  :", files[0])



import glob
import pydicom
import numpy as np

series_dir = "/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/MRI/ZGQSSWMRSM_20220125_102618_289000/_MPR_RANGE__0004"

files = sorted(glob.glob(series_dir + "/*.IMA"))
ds = pydicom.dcmread(files[0])

print("Modality:", ds.Modality)
print("SeriesDescription:", getattr(ds, "SeriesDescription", None))
print("Rows x Cols:", ds.Rows, ds.Columns)
print("PixelSpacing:", ds.PixelSpacing)
print("SliceThickness:", getattr(ds, "SliceThickness", None))
print("ImageOrientationPatient:", ds.ImageOrientationPatient)
print("ImagePositionPatient (first slice):", ds.ImagePositionPatient)

# slice 간격 계산
positions = []
for f in files:
    d = pydicom.dcmread(f, stop_before_pixels=True)
    positions.append(d.ImagePositionPatient)

positions = np.array(positions, dtype=float)
if len(positions) > 1:
    diffs = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    print("Estimated slice spacing (mm):",
          f"mean={diffs.mean():.3f}, min={diffs.min():.3f}, max={diffs.max():.3f}")
