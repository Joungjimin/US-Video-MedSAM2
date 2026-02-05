import os
import glob
import numpy as np
import cv2
import SimpleITK as sitk
import os
import glob
import numpy as np
import cv2
import SimpleITK as sitk

def convert_all_classes_to_3d(root_path, output_base_path):
    # 1. 비디오 폴더 목록 가져오기
    video_folders = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

    for video in video_folders:
        video_dir = os.path.join(root_path, video)
        frame_dirs = sorted(glob.glob(os.path.join(video_dir, "frame_*")))
        
        if not frame_dirs:
            continue

        # 2. [핵심] 모든 프레임 폴더를 뒤져서 존재하는 '모든' 클래스 이름을 수집합니다.
        all_found_classes = set()
        for f_dir in frame_dirs:
            # 해당 프레임 폴더 내의 classN_pred.png 파일들을 찾음
            preds = [f.replace('_pred.png', '') for f in os.listdir(f_dir) if f.endswith('_pred.png')]
            all_found_classes.update(preds)
        
        unique_classes = sorted(list(all_found_classes))

        print(f"\n[Video {video}] 처리 시작")
        print(f"- 발견된 클래스 목록: {unique_classes}")

        # 이미지 크기 확인 (첫 번째 유효한 이미지 기준)
        sample_img = None
        for f_dir in frame_dirs:
            for cls in unique_classes:
                p = os.path.join(f_dir, f"{cls}_pred.png")
                if os.path.exists(p):
                    sample_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    break
            if sample_img is not None: break
        
        if sample_img is None: 
            print(f"  ! Video {video}: 유효한 마스크 이미지를 찾을 수 없습니다.")
            continue
        
        h, w = sample_img.shape

        # 3. 발견된 모든 클래스에 대해 각각 3D 볼륨 생성
        for cls in unique_classes:
            volume_data = []
            
            for f_dir in frame_dirs:
                img_path = os.path.join(f_dir, f"{cls}_pred.png")
                
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    volume_data.append(img)
                else:
                    # 해당 프레임에 해당 클래스가 없는 경우, 검은색(0) 배경으로 채움
                    volume_data.append(np.zeros((h, w), dtype=np.uint8))

            # NumPy 배열을 SimpleITK 이미지로 변환 및 저장
            volume_array = np.array(volume_data, dtype=np.uint8)
            sitk_image = sitk.GetImageFromArray(volume_array)
            
            # Slicer에서 볼 때의 간격 설정 (필요시 조정)
            sitk_image.SetSpacing([1.0, 1.0, 1.0])

            os.makedirs(output_base_path, exist_ok=True)
            output_filename = f"video_{video}_{cls}.nii.gz"
            output_path = os.path.join(output_base_path, output_filename)
            
            sitk.WriteImage(sitk_image, output_path)
            print(f"  └─ {cls} 저장 완료: {output_filename}")

# --- 실행 경로 설정 ---
# 7, 9, 15 등의 폴더가 들어있는 'vis' 폴더의 경로를 적어주세요.
# root_directory = '/media/sde/zhengzhimin/MedSAM2/results_FINAL/D1_Quantum_GFTE_TempLoss/vis'
# # 저장될 위치
# output_directory = '/media/sde/zhengzhimin/MedSAM2/results_FINAL/D1_Quantum_GFTE_TempLoss/vis_3d_slicer_all'

# convert_all_classes_to_3d(root_directory, output_directory)
# 설정
for i in range(2):
    root_directory = f'/media/sde/zhengzhimin/MedSAM2/eval_results_MRI/D3_Quantum_GFTE_TempLoss{i+1}/vis'
    output_directory = f'/media/sde/zhengzhimin/MedSAM2/results_FINAL3D/MRI{i+1}_3d'
    print('root_directory',root_directory)

    convert_all_classes_to_3d(root_directory, output_directory)