import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm

def preprocess_nifti_to_npz(base_path, output_path):
    # 1. 경로 설정
    image_dir = os.path.join(base_path, "images")
    label_dir = os.path.join(base_path, "labels")
    
    # 출력 폴더 생성
    os.makedirs(output_path, exist_ok=True)

    # 2. 이미지 파일 목록 가져오기
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
    
    print(f"총 {len(image_files)}개의 파일을 찾았습니다.")

    for img_path in tqdm(image_files):
        # 파일 이름 추출 (예: 1.nii.gz -> 1)
        file_name = os.path.basename(img_path).replace(".nii.gz", "")
        
        # 라벨 경로 매칭
        lbl_path = os.path.join(label_dir, f"{file_name}.nii.gz")
        
        if not os.path.exists(lbl_path):
            print(f"경고: {file_name}에 해당하는 라벨 파일이 없습니다. 건너뜁니다.")
            continue

        # 3. 데이터 로드
        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        # numpy 배열로 변환 (float32/int16 권장)
        img_data = img_nii.get_fdata().astype(np.float32)
        lbl_data = lbl_nii.get_fdata().astype(np.uint8)  # 라벨은 보통 정수형

        # 4. NPZ로 저장 (압축 모드)
        save_file_path = os.path.join(output_path, f"{file_name}.npz")
        np.savez_compressed(
            save_file_path,
            image=img_data,
            label=lbl_data
        )

if __name__ == "__main__":
    # 설정하신 경로
    input_base = "/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/data2/train"
    output_base = "/media/sde/zhengzhimin/MedSAM2/data/MedSAM2_Data/data2_preprocessed/preprocessed/train"

    preprocess_nifti_to_npz(input_base, output_base)
    print("변환이 완료되었습니다.")