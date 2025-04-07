# -*- coding: utf-8 -*-
# ==============================================================================
# Pancreas Segmentation Training Script (for Local Execution with GPU)
# ==============================================================================

# ==============================================================================
# 0. Check Environment (Optional but Recommended)
# ==============================================================================
# Check if we are in Google Colab (should be False for local execution)
import sys
IN_COLAB = 'google.colab' in sys.modules
# print(f"Running in Google Colab: {IN_COLAB}") # Moved inside main block

# ==============================================================================
# 1. Imports (Keep all imports at the top level)
# ==============================================================================
import os
import zipfile # 압축 해제용 (zip)
import tarfile # 압축 해제용 (tar)
import json    # 메타데이터 처리용 (json)
import nibabel as nib # NIfTI 처리 (주로 MONAI 내에서 사용)
import numpy as np
import pandas as pd # 메타데이터 처리용
from scipy.ndimage import zoom # 사용되지 않음 (MONAI Resize 사용)
from sklearn.model_selection import train_test_split # 데이터 분할
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import monai
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss # Combined Dice + CrossEntropy Loss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet # MONAI의 UNet 사용 가능
from monai.transforms import (
    Activations, AsDiscrete, Compose, EnsureChannelFirstd, LoadImaged,
    Orientationd, RandFlipd, RandRotate90d, RandShiftIntensityd, RandGaussianNoised,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped, Spacingd,
)
from monai.utils import set_determinism
import matplotlib.pyplot as plt
from tqdm import tqdm # 진행상황 표시
import gc # 메모리 관리
import datetime # 로그 및 모델 저장명용
import traceback # 오류 스택 트레이스 출력용
from collections import Counter # 경로 준비 시 사용 가능
import time
# Optional: Import torchinfo for model summary
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False


# ==============================================================================
# 2. Configuration (Define constants and hyperparameters globally)
# ==============================================================================

# --- 경로 설정 ---
# !!! 중요: 로컬 환경에 맞게 이 경로를 수정하세요 !!!
# DRIVE_BASE_PATH: Task07_Pancreas.tar와 t1.zip 파일이 있는 *로컬 폴더* 경로
DRIVE_BASE_PATH = 'C:/Users/21/Desktop/췌장암' # <<<--- 사용자의 실제 로컬 데이터 폴더 경로로 반드시 수정하세요!

# BASE_PATH: 작업 결과물(압축 해제 데이터, 로그, 모델 등)이 저장될 기본 경로
BASE_PATH = './' # 현재 스크립트 실행 위치를 기준으로 설정

# --- 데이터 파일 경로 (위의 DRIVE_BASE_PATH 기준) ---
TAR_CANCER_PATH = os.path.join(DRIVE_BASE_PATH, 'Task07_Pancreas.tar')
ZIP_NORMAL_PATH = os.path.join(DRIVE_BASE_PATH, 't1.zip') # 만약 파일명이 다르다면 수정

# --- 작업 경로 (BASE_PATH 기준) ---
WORK_DIR = os.path.join(BASE_PATH, 'pancreas_project')
EXTRACT_BASE_PATH = os.path.join(WORK_DIR, 'data')
CANCER_EXTRACT_PATH = os.path.join(EXTRACT_BASE_PATH, 'Task07_Pancreas')
NORMAL_EXTRACT_PATH = os.path.join(EXTRACT_BASE_PATH, 't1')

# --- 메타데이터 경로 (JSON) ---
# 압축 해제 후 예상되는 경로. get_adjusted_root 함수가 내부적으로 조정할 수 있음.
JSON_CANCER_METADATA_PATH_TEMPLATE = os.path.join(CANCER_EXTRACT_PATH, 'Task07_Pancreas', 'dataset.json')

# --- MONAI 데이터 처리 파라미터 ---
TARGET_SPATIAL_SHAPE = (64, 96, 96) # (D, H, W) - 리사이징 목표 크기
HU_WINDOW = (-100, 240) # CT Hounsfield Unit Windowing
FG_LABELS = (1, 2) # 사용되진 않지만 정보용 (Loss에서 background 포함 여부로 처리)
CACHE_DATASET = True # True: 데이터셋을 메모리에 캐시 (빠름, 메모리 많이 사용), False: 매번 로드
NUM_WORKERS = 4 # 데이터 로딩 워커 수 (로컬 CPU 코어 수에 맞게 조절, 예: 4, 8)

# --- 학습 파라미터 ---
MAX_EPOCHS = 100
BATCH_SIZE = 2 # 로컬 GPU 메모리에 따라 조절 (RTX 2060이면 1 또는 2가 적절할 수 있음)
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
BALANCE_DATA = True # 데이터 불균형 시 오버샘플링 여부

# --- 모델 파라미터 ---
MODEL_NAME = "UNet" # "UNet" (직접 구현) 또는 "MONAI_UNet"
MODEL_FILTERS = [16, 32, 64, 128] # 필터 수 (메모리 부족 시 줄이기)
DROPOUT_RATE = 0.15
ACTIVATION_FN_TYPE = 'leaky_relu'
LEAKY_RELU_NEGATIVE_SLOPE = 0.01 # LeakyReLU 사용 시

# --- 결과 저장 및 체크포인트 경로 ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs", TIMESTAMP)
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_model_{TIMESTAMP}.pth")
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
SAVE_CHECKPOINT_FREQ = 5 # 몇 에폭마다 latest 체크포인트 저장할지
RESUME_CHECKPOINT = None # 이어할 체크포인트 경로 지정 (e.g., LATEST_CHECKPOINT_PATH) or True

# --- 디바이스 설정 (실행 시점에 결정되지만 변수는 여기서 선언) ---
DEVICE = None


# ==============================================================================
# 3. Utility Functions (Define globally)
# ==============================================================================

# 압축 해제 함수
def unzip_data(archive_path, extract_to):
    """압축 해제 함수 (zip/tar 지원, 폴더 존재 시 건너뛰기)"""
    print(f"\nChecking extraction for '{os.path.basename(archive_path)}' to '{extract_to}'...")
    try:
        if os.path.exists(extract_to) and any(os.scandir(extract_to)):
            print(f"  Directory '{extract_to}' already exists and is not empty. Skipping extraction.")
            return True
        elif os.path.exists(extract_to):
            print(f"  Directory '{extract_to}' exists but is empty. Proceeding with extraction.")
        else:
            os.makedirs(extract_to, exist_ok=True)
            print(f"  Created directory '{extract_to}'.")
    except OSError as e:
        print(f"  Warning: Could not check/create directory {extract_to}: {e}.")

    print(f"  Starting extraction...")
    if not os.path.exists(archive_path):
        print(f"  Error: Archive file not found: '{archive_path}'."); return False

    try:
        if archive_path.lower().endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc=f'  Extracting {os.path.basename(archive_path)} (zip)'):
                    try:
                        target_path = os.path.join(extract_to, member.filename)
                        if not os.path.abspath(target_path).startswith(os.path.abspath(extract_to)):
                            print(f"  Warning: Skipping potentially unsafe path {member.filename}")
                            continue
                        zip_ref.extract(member, extract_to)
                    except Exception as e: print(f"  Warning: Could not extract {member.filename} from zip. Error: {e}")
        elif archive_path.lower().endswith(('.tar', '.tar.gz', '.tgz')):
            # filter='data' is recommended from Python 3.12+ for security
            filter_arg = {}
            if sys.version_info >= (3, 12):
                 filter_arg['filter'] = 'data'

            with tarfile.open(archive_path, 'r:*') as tar_ref:
                members = tar_ref.getmembers()
                for member in tqdm(members, desc=f'  Extracting {os.path.basename(archive_path)} (tar)'):
                    try:
                        target_path = os.path.join(extract_to, member.name)
                        if not os.path.abspath(target_path).startswith(os.path.abspath(extract_to)):
                           print(f"  Warning: Skipping potentially unsafe path {member.name}")
                           continue
                        # Pass filter arg only if defined (for compatibility)
                        tar_ref.extract(member, path=extract_to, set_attrs=False, **filter_arg)
                    except Exception as e: print(f"  Warning: Could not extract {member.name} from tar. Error: {e}")
        else:
            print(f"  Error: Unsupported archive format for {archive_path}. Only .zip and .tar(.gz) are supported.")
            return False
        print(f"  Successfully extracted.")
        return True
    except Exception as e:
        print(f"  Error during extraction: {e}"); traceback.print_exc(); return False

# 경로 조정 함수
def get_adjusted_root(extract_path, expected_top_level=""):
    """압축 해제 후 실제 데이터 루트 폴더 찾기"""
    if not extract_path or not os.path.exists(extract_path): return None
    try:
        items = os.listdir(extract_path)
        if len(items) == 1 and os.path.isdir(os.path.join(extract_path, items[0])):
            adjusted = os.path.join(extract_path, items[0])
            print(f"  Adjusted root (single inner folder): {adjusted}")
            return adjusted
        if expected_top_level:
            potential_root = os.path.join(extract_path, expected_top_level)
            if os.path.isdir(potential_root):
                print(f"  Adjusted root (expected top level): {potential_root}")
                return potential_root
        print(f"  Using original extraction path as root: {extract_path}")
        return extract_path
    except Exception as e:
        print(f"  Error adjusting path: {e}"); return extract_path

# 학습 history 시각화 함수
def plot_segmentation_history(log_file_path):
    """로그 파일(CSV)에서 학습 history 읽어 그래프 출력"""
    try:
        history_df = pd.read_csv(log_file_path)
        if history_df.empty:
            print("Log file is empty. No history to plot.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        epochs = history_df['epoch']

        has_loss = 'train_loss' in history_df.columns and 'val_loss' in history_df.columns
        has_dice = 'val_dice' in history_df.columns
        has_iou = 'val_iou' in history_df.columns

        num_plots = sum([has_loss, has_dice, has_iou])
        if num_plots == 0:
            print("No suitable metrics found in log file to plot.")
            return

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False) # Ensure axes is always 2D array
        axes = axes.flatten() # Flatten to 1D array for easy indexing
        plot_idx = 0

        if has_loss:
            ax = axes[plot_idx]
            ax.plot(epochs, history_df['train_loss'], label='Training Loss', lw=2)
            ax.plot(epochs, history_df['val_loss'], label='Validation Loss', lw=2)
            ax.set_title('Model Loss', fontsize=14); ax.set_xlabel('Epochs'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(True);
            plot_idx += 1

        if has_dice:
            ax = axes[plot_idx]
            if 'train_dice' in history_df.columns: # Optional: plot train dice if logged
                ax.plot(epochs, history_df['train_dice'], label='Training Dice', lw=2)
            ax.plot(epochs, history_df['val_dice'], label='Validation Dice', lw=2)
            ax.set_title('Validation Dice Coefficient', fontsize=14); ax.set_xlabel('Epochs'); ax.set_ylabel('Dice Coeff'); ax.legend(); ax.grid(True);
            plot_idx += 1

        if has_iou:
            ax = axes[plot_idx]
            if 'train_iou' in history_df.columns: # Optional: plot train iou if logged
                 ax.plot(epochs, history_df['train_iou'], label='Training IoU', lw=2)
            ax.plot(epochs, history_df['val_iou'], label='Validation IoU', lw=2)
            ax.set_title('Validation Mean IoU', fontsize=14); ax.set_xlabel('Epochs'); ax.set_ylabel('Mean IoU'); ax.legend(); ax.grid(True);
            plot_idx += 1

        plt.tight_layout(); plt.show()

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
    except Exception as e:
        print(f"Error plotting history from log file: {e}")
        traceback.print_exc()

# 슬라이스 비교 시각화 함수
def display_slice_comparison(image_vol, true_mask_vol, predicted_mask_vol, slice_index, main_title="Slice Comparison"):
    """특정 슬라이스 비교 시각화 (PyTorch Tensor, Channel-First 지원)"""
    def prep_slice(vol_tensor, is_mask=False):
        if not isinstance(vol_tensor, torch.Tensor):
            print(f"Warning: Input is not a PyTorch tensor ({type(vol_tensor)}). Attempting conversion.")
            try: vol_tensor = torch.tensor(vol_tensor)
            except Exception as e: print(f"Error converting input to tensor: {e}"); return np.zeros(TARGET_SPATIAL_SHAPE[1:])

        vol_np = vol_tensor.detach().cpu().numpy()

        if vol_np.ndim == 4:
            vol_np = np.squeeze(vol_np, axis=0) if vol_np.shape[0] == 1 else vol_np[0] # Use first channel if multi-channel

        if vol_np.ndim != 3:
            print(f"Error: Expected 3D volume, got shape {vol_np.shape}"); return np.zeros(TARGET_SPATIAL_SHAPE[1:])

        depth = vol_np.shape[0]
        nonlocal slice_index
        if not (0 <= slice_index < depth):
            print(f"Invalid slice_index {slice_index}. Adjusting to {depth // 2}")
            slice_index = depth // 2

        s = vol_np[slice_index]
        if s.ndim != 2: print(f"Error: Slice is not 2D (shape: {s.shape})."); return np.zeros(TARGET_SPATIAL_SHAPE[1:])

        dtype = np.uint8 if is_mask else np.float32
        return s.astype(dtype)

    image_slice = prep_slice(image_vol, is_mask=False)
    true_mask_slice = prep_slice(true_mask_vol, is_mask=True)
    pred_mask_slice = prep_slice(predicted_mask_vol, is_mask=True)

    plt.figure(figsize=(18, 6))
    vol_depth = image_vol.shape[1] if image_vol.ndim == 4 else image_vol.shape[0]
    plt.suptitle(f"{main_title} - Slice {slice_index}/{vol_depth-1}", fontsize=16)

    img_min, img_max = 0.0, 1.0 # Assuming normalized image

    plt.subplot(1, 3, 1); plt.imshow(image_slice, cmap='gray', vmin=img_min, vmax=img_max); plt.title('Original Image'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(image_slice, cmap='gray', vmin=img_min, vmax=img_max); plt.imshow(np.ma.masked_where(true_mask_slice == 0, true_mask_slice), cmap='jet', alpha=0.5); plt.title('True Mask Overlay'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(image_slice, cmap='gray', vmin=img_min, vmax=img_max); plt.imshow(np.ma.masked_where(pred_mask_slice == 0, pred_mask_slice), cmap='jet', alpha=0.5); plt.title('Predicted Mask Overlay'); plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

# NIfTI 파일 찾기 함수 (MONAI 딕셔너리 형식)
def find_nifti_files_monai(directory, subfolder_name, label_val, image_key="image", label_key="label"):
    """지정된 하위 폴더에서 .nii/.nii.gz 파일을 찾아 MONAI 딕셔너리 리스트 반환"""
    file_dicts = []
    search_path = os.path.join(directory, subfolder_name)
    if not os.path.isdir(search_path):
        print(f"Warning: Image subfolder '{subfolder_name}' not found in '{directory}'")
        return file_dicts
    try:
        for filename in os.listdir(search_path):
            if filename.lower().endswith(('.nii', '.nii.gz')) and not filename.startswith('.'):
                file_dicts.append({
                    image_key: os.path.join(search_path, filename),
                    label_key: None, # 라벨은 나중에 매칭
                    'class': label_val # 클래스 라벨 (정상 0, 암 1)
                })
    except Exception as e:
        print(f"Error finding NIfTI files in '{search_path}': {e}")
    return file_dicts

# 데이터 준비 및 분할 함수 (MONAI 버전)
def prepare_and_split_files_monai(cancer_root, normal_root, cancer_json_path,
                                  image_key="image", label_key="label",
                                  image_folder="imagesTr", label_folder="labelsTr",
                                  balance=True, test_size=0.2, random_state=42):
    """JSON과 파일 매칭을 통해 데이터 파일 딕셔너리 리스트를 준비하고 분할"""
    cancer_files = []
    normal_files = []

    # --- 암 데이터 준비 (JSON 기반) ---
    print("\n--- Preparing Cancer Data Files (from JSON) ---")
    processed_cancer_count = 0
    skipped_cancer_count = 0
    if cancer_root and cancer_json_path and os.path.exists(cancer_json_path):
        try:
            with open(cancer_json_path, 'r') as f: metadata = json.load(f)
            training_data = metadata.get('training', [])
            print(f"Found {len(training_data)} entries in cancer metadata JSON.")

            for item in tqdm(training_data, desc="Processing Cancer JSON entries"):
                img_rel_path = item.get('image')
                lbl_rel_path = item.get('label')
                if not img_rel_path or not lbl_rel_path: skipped_cancer_count += 1; continue

                img_abs_path = os.path.abspath(os.path.join(cancer_root, img_rel_path.lstrip('./\\')))
                lbl_abs_path = os.path.abspath(os.path.join(cancer_root, lbl_rel_path.lstrip('./\\')))

                if os.path.isfile(img_abs_path) and os.path.isfile(lbl_abs_path):
                    cancer_files.append({image_key: img_abs_path, label_key: lbl_abs_path, 'class': 1})
                    processed_cancer_count += 1
                else: skipped_cancer_count += 1

            print(f"Successfully prepared {processed_cancer_count} valid cancer file pairs.")
            if skipped_cancer_count > 0: print(f"Skipped {skipped_cancer_count} cancer entries.")
        except Exception as e: print(f"Error processing Cancer JSON metadata: {e}"); traceback.print_exc()
    else: print("Cancer root path or JSON metadata path is invalid. Skipping cancer data.")

    # --- 정상 데이터 준비 (파일 매칭 기반) ---
    print("\n--- Preparing Normal Data Files (Matching Files) ---")
    processed_normal_count = 0
    skipped_normal_count = 0
    if normal_root and os.path.isdir(normal_root):
        normal_image_dicts = find_nifti_files_monai(normal_root, image_folder, 0, image_key, label_key)
        normal_label_search_path = os.path.join(normal_root, label_folder)
        print(f"Found {len(normal_image_dicts)} potential normal images.")
        print(f"Searching for matching normal labels in: {normal_label_search_path}")

        if not os.path.isdir(normal_label_search_path):
             print(f"Warning: Normal label folder '{label_folder}' not found in '{normal_root}'. Cannot match normal labels.")
        else:
            label_files = {}
            for fname in os.listdir(normal_label_search_path):
                 if fname.lower().endswith(('.nii', '.nii.gz')) and not fname.startswith('.'):
                     base_name = fname.replace('.nii.gz', '').replace('.nii', '')
                     label_files[base_name] = os.path.join(normal_label_search_path, fname)

            for img_dict in tqdm(normal_image_dicts, desc="Matching Normal files"):
                img_path = img_dict[image_key]
                img_filename = os.path.basename(img_path)
                img_base_name = img_filename.replace('_0000.nii.gz', '').replace('.nii.gz', '').replace('.nii', '') # More robust name matching
                label_match_name = img_base_name # Assume label has same base name

                if label_match_name in label_files:
                    lbl_path = label_files[label_match_name]
                    if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                        img_dict[label_key] = lbl_path
                        normal_files.append(img_dict)
                        processed_normal_count += 1
                    else: skipped_normal_count += 1
                else: skipped_normal_count += 1

            print(f"Successfully prepared {processed_normal_count} valid normal file pairs.")
            if skipped_normal_count > 0: print(f"Skipped {skipped_normal_count} normal files/pairs.")
    else: print("Normal root path is invalid or not found. Skipping normal data.")

    # --- 데이터 밸런싱 및 분할 ---
    n_cancer = len(cancer_files); n_normal = len(normal_files)
    print(f"\nTotal valid data files prepared: {n_cancer + n_normal}")
    print(f"Valid counts - Cancer: {n_cancer}, Normal: {n_normal}")

    all_files = cancer_files + normal_files
    if not all_files: print("Error: No valid data files collected."); return [], []

    labels = [f['class'] for f in all_files]

    if balance and n_cancer > 0 and n_normal > 0 and n_cancer != n_normal:
        print(f"\nBalancing data (using oversampling)...")
        minority_files, majority_files = (cancer_files, normal_files) if n_cancer < n_normal else (normal_files, cancer_files)
        minority_class_name = "Cancer" if n_cancer < n_normal else "Normal"
        majority_class_name = "Normal" if n_cancer < n_normal else "Cancer"
        n_minority, n_majority = len(minority_files), len(majority_files)

        print(f"Oversampling {minority_class_name} class from {n_minority} to match {majority_class_name} class {n_majority}...")
        oversample_indices = np.random.choice(n_minority, size=n_majority - n_minority, replace=True)
        oversampled_files = [minority_files[i] for i in oversample_indices]
        all_files = majority_files + minority_files + oversampled_files
        labels = [f['class'] for f in all_files]
        print(f"Total files after balancing: {len(all_files)}")
        print(f"Balanced counts - Cancer: {sum(l==1 for l in labels)}, Normal: {sum(l==0 for l in labels)}")
    elif balance and (n_cancer == 0 or n_normal == 0):
        print("Warning: Cannot balance data - one class has zero samples.")

    # --- 분할 ---
    train_files, val_files = [], []
    if not all_files: print("Error: No files to split."); return [], []

    try:
        unique_labels, counts = np.unique(labels, return_counts=True)
        can_stratify = len(unique_labels) >= 2 and all(c >= 2 for c in counts)
        stratify_param = labels if can_stratify else None
        split_type = "stratified" if can_stratify else "regular"

        if len(all_files) >= 2 and 0 < test_size < 1:
             print(f"Performing {split_type} split.")
             train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state, stratify=stratify_param)
        elif len(all_files) > 0:
             print("Warning: Splitting not possible or test_size is invalid. Using all data for training.")
             train_files = all_files; val_files = []
        else: print("Error: No data left to split."); return [], []
    except ValueError as e:
        print(f"Warning: Error during train_test_split (likely too few samples for stratification): {e}. Performing regular split.")
        if len(all_files) >= 2 and 0 < test_size < 1:
             train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state)
        elif len(all_files) > 0: train_files = all_files; val_files = []
        else: return [], []

    print("\nSplitting data complete:")
    print(f"  Training files: {len(train_files)}")
    print(f"  Validation files: {len(val_files)}")
    if train_files: print(f"  Training distribution - Cancer: {sum(f['class']==1 for f in train_files)}, Normal: {sum(f['class']==0 for f in train_files)}")
    if val_files: print(f"  Validation distribution - Cancer: {sum(f['class']==1 for f in val_files)}, Normal: {sum(f['class']==0 for f in val_files)}")

    return train_files, val_files

# Activation 함수 반환 헬퍼
def get_activation(activation_type, negative_slope=0.01):
    if activation_type.lower() == 'relu': return nn.ReLU(inplace=True)
    elif activation_type.lower() == 'leaky_relu': return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    else: raise ValueError(f"Unsupported activation function type: {activation_type}")

# ==============================================================================
# 5. Model Definition (Define globally)
# ==============================================================================

# Custom 3D U-Net Class
class Custom3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=[16, 32, 64, 128], dropout_rate=0.1, activation='relu', leaky_slope=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.leaky_slope = leaky_slope

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        def conv_block(ic, oc, act, ls, dr):
            layers = [
                nn.Conv3d(ic, oc, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(oc), get_activation(act, ls),
                nn.Conv3d(oc, oc, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(oc), get_activation(act, ls),
            ]
            if dr > 0.0: layers.append(nn.Dropout3d(dr))
            return nn.Sequential(*layers)

        # Encoder Path
        current_channels = in_channels
        # print("Encoder Path (Structure):") # Print moved to main block
        for i, f in enumerate(filters):
            # print(f"  Layer {i+1} - Filters: {f}") # Print moved to main block
            current_dropout = dropout_rate * (i / (len(filters) - 1)) if len(filters) > 1 and dropout_rate > 0 else 0.0
            encoder = conv_block(current_channels, f, self.activation, self.leaky_slope, current_dropout)
            pool = nn.MaxPool3d(kernel_size=2, stride=2)
            self.encoders.append(encoder)
            self.pools.append(pool)
            current_channels = f

        # Bottleneck
        bn_filters = filters[-1] * 2
        # print(f"Bottleneck - Filters: {bn_filters}") # Print moved to main block
        self.bottleneck = conv_block(current_channels, bn_filters, self.activation, self.leaky_slope, dropout_rate)

        # Decoder Path
        # print("Decoder Path (Structure):") # Print moved to main block
        current_channels = bn_filters
        reversed_filters = list(reversed(filters))
        for i, f in enumerate(reversed_filters):
            # print(f"  Layer {i+1} - Filters: {f}") # Print moved to main block
            upconv = nn.ConvTranspose3d(current_channels, f, kernel_size=2, stride=2)
            self.upconvs.append(upconv)
            concat_channels = f + f
            current_dropout = dropout_rate * ((len(filters) - 1 - i) / (len(filters) - 1)) if len(filters) > 1 and dropout_rate > 0 else 0.0
            decoder = conv_block(concat_channels, f, self.activation, self.leaky_slope, current_dropout)
            self.decoders.append(decoder)
            current_channels = f

        # Output Layer
        self.output_conv = nn.Conv3d(current_channels, out_channels, kernel_size=1)
        # print(f"Output Layer - Output Channels: {out_channels}") # Print moved to main block

    def forward(self, x):
        skips = []
        # Encoder
        for i in range(len(self.filters)):
            x = self.encoders[i](x); skips.append(x); x = self.pools[i](x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder
        skips = list(reversed(skips))
        for i in range(len(self.filters)):
            x = self.upconvs[i](x)
            skip_connection = skips[i]
            if x.shape[2:] != skip_connection.shape[2:]: # Basic shape check
                # MONAI UNet handles this better, basic crop for demo
                print(f"Warning: Skip connection shape mismatch! Upconv: {x.shape}, Skip: {skip_connection.shape}")
                target_shape = x.shape[2:]
                skip_shape = skip_connection.shape[2:]
                try:
                    crop_slices = [slice((skip_shape[d] - target_shape[d]) // 2, (skip_shape[d] + target_shape[d]) // 2) for d in range(3)]
                    skip_connection = skip_connection[(slice(None), slice(None)) + tuple(crop_slices)]
                except Exception as crop_e:
                    print(f"  Error during skip connection cropping: {crop_e}. Trying padding.")
                    # Alternative: Padding (might be needed if upconv is larger)
                    padding = []
                    for d in range(3):
                         pad_total = target_shape[d] - skip_shape[d]
                         pad_before = pad_total // 2
                         pad_after = pad_total - pad_before
                         padding.extend([pad_before, pad_after]) # Pad D, H, W
                    # Reverse padding order for F.pad (W, H, D)
                    padding = padding[::-1]
                    if all(p >= 0 for p in padding):
                        skip_connection = nn.functional.pad(skip_connection, padding)
                        print(f"  Padded skip connection to: {skip_connection.shape}")
                    else:
                         print(f"  Cannot resolve shape mismatch automatically. Check model architecture and input size.")
                         # You might need to raise an error here or implement more robust handling
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoders[i](x)
        # Output
        x = self.output_conv(x)
        return x

# ==============================================================================
# 4. Data Loading & Preprocessing Definition (Define globally)
# ==============================================================================
image_key = "image"
label_key = "label"

# Training Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=[image_key, label_key]),
        EnsureChannelFirstd(keys=[image_key, label_key]),
        Orientationd(keys=[image_key, label_key], axcodes="RAS"),
        ScaleIntensityRanged(keys=[image_key], a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=[image_key, label_key], source_key=image_key), # allow_smaller=True by default until monai 1.5
        Resized(keys=[image_key, label_key], spatial_size=TARGET_SPATIAL_SHAPE, mode=("area", "nearest")),
        RandFlipd(keys=[image_key, label_key], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=[image_key, label_key], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=[image_key, label_key], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=[image_key, label_key], prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandShiftIntensityd(keys=[image_key], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=[image_key], prob=0.3, mean=0.0, std=0.05),
        EnsureTyped(keys=[image_key, label_key], dtype=torch.float32),
    ]
)

# Validation Transforms
val_transforms = Compose(
    [
        LoadImaged(keys=[image_key, label_key]),
        EnsureChannelFirstd(keys=[image_key, label_key]),
        Orientationd(keys=[image_key, label_key], axcodes="RAS"),
        ScaleIntensityRanged(keys=[image_key], a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=[image_key, label_key], source_key=image_key), # allow_smaller=True by default until monai 1.5
        Resized(keys=[image_key, label_key], spatial_size=TARGET_SPATIAL_SHAPE, mode=("area", "nearest")),
        EnsureTyped(keys=[image_key, label_key], dtype=torch.float32),
    ]
)

# ==============================================================================
# Loss & Metrics (Define globally)
# ==============================================================================
loss_function = DiceCELoss(to_onehot_y=True, softmax=True,
                           include_background=True)
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
post_label = Compose([AsDiscrete(to_onehot=3)]) # (원-핫 변환)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)


# ==============================================================================
# Main Execution Block - Guarded for Multiprocessing
# ==============================================================================
if __name__ == '__main__':

    # Add freeze_support() for Windows multiprocessing compatibility
    # This line is crucial when using DataLoader with num_workers > 0 on Windows
    from multiprocessing import freeze_support
    freeze_support()

    # --- Print Setup Info ONLY in Main Process ---
    print(f"Running in Google Colab: {IN_COLAB}")
    print("--- 1. Setup ---")
    if not IN_COLAB:
        print("Not running in Colab. Skipping Drive mount.")
    # Assuming libraries are installed externally via pip/conda before running script
    print("필요 라이브러리 설치 중 (PyTorch, MONAI, etc.)... (Skipped in script, assumed installed)")
    print("라이브러리 설치 완료. (Assumed)")

    print(f"\nMONAI version: {monai.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available, running on CPU.")
    set_determinism(seed=RANDOM_STATE) # Use RANDOM_STATE defined globally

    print("\n--- 2. Configuration ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine device
    print(f"Using device: {DEVICE}")
    # Create output directories here, within the main block
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Configuration setup complete.")
    print(f"  Target Spatial Shape (D, H, W): {TARGET_SPATIAL_SHAPE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Epochs: {MAX_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Activation Function: {ACTIVATION_FN_TYPE}")
    print(f"  Model Name: {MODEL_NAME}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"  Resume Checkpoint: {RESUME_CHECKPOINT}")
    # Check if the user path placeholder is still there
    if DRIVE_BASE_PATH == './path/to/your/local/data/':
         print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("!!! WARNING: 'DRIVE_BASE_PATH' needs to be updated with    !!!")
         print("!!!          your actual local data directory path.        !!!")
         print("!!!          Script might fail if data is not found.       !!!")
         print(f"!!! Current Path: {DRIVE_BASE_PATH}                            !!!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
         time.sleep(5) # Give user time to see the warning


    print("\n--- 3. Utility Functions ---")
    print("Utility functions defined globally.")

    print("\n--- 4. Data Loading & Preprocessing Definition (MONAI) ---")
    print("MONAI data transforms defined globally.")

    print("\n--- 5. Model Definition (PyTorch) ---")
    print("PyTorch Model Class, Loss, and Metrics defined globally.")
    print(f"Using Loss: {type(loss_function).__name__}")

    # ==============================================================================
    # 6. Main Execution Steps (Inside the Guard)
    # ==============================================================================
    print("\n--- 6. Main Execution ---")

    # --- 6.1. Unzipping Data and Adjusting Paths ---
    print("\n--- 6.1. Unzipping Data and Adjusting Paths ---")
    unzip_success_cancer = unzip_data(TAR_CANCER_PATH, CANCER_EXTRACT_PATH)
    unzip_success_normal = unzip_data(ZIP_NORMAL_PATH, NORMAL_EXTRACT_PATH)

    cancer_root_adj = get_adjusted_root(CANCER_EXTRACT_PATH, "Task07_Pancreas") if unzip_success_cancer else None
    normal_root_adj = get_adjusted_root(NORMAL_EXTRACT_PATH, "t1") if unzip_success_normal else None # Adjusted for t1 folder

    # Adjust JSON path based on potentially adjusted cancer root
    current_json_path = None
    if cancer_root_adj:
        current_json_path = os.path.join(cancer_root_adj, 'dataset.json')
        if not os.path.exists(current_json_path):
             print(f"Warning: dataset.json not found directly in {cancer_root_adj}. Trying template path.")
             current_json_path = JSON_CANCER_METADATA_PATH_TEMPLATE # Fallback
             if not os.path.exists(current_json_path):
                  print(f"Error: dataset.json also not found at template path: {current_json_path}")
                  current_json_path = None # Ensure it's None if not found

    print("\n--- Final Data Paths (Adjusted) ---")
    print(f"Cancer Data Root: {cancer_root_adj} (Exists: {os.path.exists(cancer_root_adj) if cancer_root_adj else False})")
    print(f"Normal Data Root: {normal_root_adj} (Exists: {os.path.exists(normal_root_adj) if normal_root_adj else False})")
    print(f"Cancer Metadata JSON: {current_json_path} (Exists: {os.path.exists(current_json_path) if current_json_path else False})")

    if not cancer_root_adj or not current_json_path:
        raise ValueError("Essential Cancer data path or metadata JSON missing or not found. Check paths and extraction.")

    # --- 6.2. Preparing and Splitting Data Files ---
    print("\n--- 6.2. Preparing and Splitting Data Files (for MONAI) ---")
    train_files, val_files = prepare_and_split_files_monai(
        cancer_root=cancer_root_adj,
        normal_root=normal_root_adj,
        cancer_json_path=current_json_path, # Use the determined JSON path
        image_key=image_key, label_key=label_key, # Pass keys defined globally
        balance=BALANCE_DATA,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE
    )
    if not train_files: raise ValueError("Failed to prepare training files.")
    if not val_files and VALIDATION_SPLIT > 0: print("Warning: Validation files list is empty.")

    # --- 6.3. Creating MONAI Datasets and DataLoaders ---
    print("\n--- 6.3. Creating MONAI Datasets and DataLoaders ---")
    train_ds, val_ds = None, None
    train_loader, val_loader = None, None
    try:
        if train_files:
            dataset_class = CacheDataset if CACHE_DATASET else Dataset
            cache_args = {'cache_rate': 1.0, 'num_workers': NUM_WORKERS} if CACHE_DATASET else {}
            print(f"Creating {dataset_class.__name__} for training...")
            train_ds = dataset_class(data=train_files, transform=train_transforms, **cache_args)
            print(f"Training {dataset_class.__name__} created.")
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            print(f"Training DataLoader created. Samples: {len(train_ds)}, Batches/Epoch: {len(train_loader)}")

        if val_files:
            print("Creating regular Dataset for validation...")
            val_ds = Dataset(data=val_files, transform=val_transforms)
            print("Validation Dataset created.")
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            print(f"Validation DataLoader created. Samples: {len(val_ds)}, Batches/Epoch: {len(val_loader)}")
        else: print("No validation files, skipping validation dataset/loader creation.")
    except Exception as e: print(f"Error creating MONAI datasets/dataloaders: {e}"); traceback.print_exc(); raise

    # --- (Optional) Visualize a sample ---
    if train_loader and len(train_loader)>0 : # Check if loader has content
        try:
            print("\nChecking sample from training dataloader...")
            check_data = monai.utils.first(train_loader)
            if check_data and image_key in check_data and label_key in check_data:
                 check_image = check_data[image_key][0]
                 check_label = check_data[label_key][0]
                 print(f"Sample image shape: {check_image.shape}, dtype: {check_image.dtype}")
                 print(f"Sample label shape: {check_label.shape}, dtype: {check_label.dtype}")
                 print(f"Sample image intensity range: {check_image.min():.2f} to {check_image.max():.2f}")
                 print(f"Sample label unique values: {torch.unique(check_label)}")
                 # Visualize middle slice
                 if check_image.shape[1] > 0: # Check depth dim exists
                     slice_idx = check_image.shape[1] // 2
                     plt.figure("Check Sample", (12, 6))
                     plt.subplot(1, 2, 1); plt.title("Image Slice"); plt.imshow(check_image[0, slice_idx, :, :].detach().cpu(), cmap="gray"); plt.axis('off')
                     plt.subplot(1, 2, 2); plt.title("Label Slice"); plt.imshow(check_label[0, slice_idx, :, :].detach().cpu(), cmap="jet", alpha=0.6); plt.axis('off')
                     plt.show()
            else: print("Could not retrieve valid sample batch from train_loader.")
        except Exception as e: print(f"Error checking/visualizing sample: {e}")


    # --- 6.4. Building Model and Setup ---
    print("\n--- 6.4. Building Model and Setup ---")
    model = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    try:
        if MODEL_NAME == "UNet":
            print(f"Building Custom 3D UNet with activation: {ACTIVATION_FN_TYPE}")
            model = Custom3DUNet(in_channels=1, out_channels=3, filters=MODEL_FILTERS, dropout_rate=DROPOUT_RATE, activation=ACTIVATION_FN_TYPE, leaky_slope=LEAKY_RELU_NEGATIVE_SLOPE)
            # Print structure details for custom model
            print("Encoder Path (Structure):")
            for i, f in enumerate(MODEL_FILTERS): print(f"  Layer {i+1} - Filters: {f}")
            print(f"Bottleneck - Filters: {MODEL_FILTERS[-1] * 2}")
            print("Decoder Path (Structure):")
            for i, f in enumerate(reversed(MODEL_FILTERS)): print(f"  Layer {i+1} - Filters: {f}")
            print(f"Output Layer - Output Channels: 3")

        elif MODEL_NAME == "MONAI_UNet":
             # Parameters defined globally
             MONAI_UNET_SPATIAL_DIMS = 3
             MONAI_UNET_CHANNELS = (16, 32, 64, 128, 256)
             MONAI_UNET_STRIDES = (2, 2, 2, 2)
             MONAI_UNET_NUM_RES_UNITS = 2
             print("Building MONAI UNet...")
             model = UNet(
                 spatial_dims=MONAI_UNET_SPATIAL_DIMS, in_channels=1, out_channels=3,
                 channels=MONAI_UNET_CHANNELS, strides=MONAI_UNET_STRIDES,
                 num_res_units=MONAI_UNET_NUM_RES_UNITS, act='PRELU', norm='BATCH', dropout=DROPOUT_RATE
             )
        else: raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

        model = model.to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True, min_lr=1e-7)
        early_stopping_patience = 15
        early_stopping_counter = 0
        best_val_loss = float('inf')
        best_metric = -1.0
        print("Model, Optimizer, Scheduler initialized.")

        if TORCHINFO_AVAILABLE:
            try:
                input_size = (BATCH_SIZE, 1, *TARGET_SPATIAL_SHAPE)
                print(f"\nModel Summary (Input size: {input_size}):")
                summary(model, input_size=input_size, device=str(DEVICE), col_names=["input_size", "output_size", "num_params", "mult_adds"], verbose=0)
            except Exception as e: print(f"\nError generating model summary: {e}")
        else: print("\n'torchinfo' not installed. Skipping model summary."); print(f"Model architecture:\n{model}")
    except Exception as e: print(f"Error building model or setup: {e}"); traceback.print_exc(); raise

    # --- 6.5. Checkpoint Loading ---
    start_epoch = 0
    if RESUME_CHECKPOINT:
        ckpt_path_to_load = None
        if isinstance(RESUME_CHECKPOINT, str) and os.path.exists(RESUME_CHECKPOINT): ckpt_path_to_load = RESUME_CHECKPOINT
        elif RESUME_CHECKPOINT is True and os.path.exists(LATEST_CHECKPOINT_PATH): ckpt_path_to_load = LATEST_CHECKPOINT_PATH

        if ckpt_path_to_load:
            print(f"\n--- Resuming Training from Checkpoint: {ckpt_path_to_load} ---")
            try:
                checkpoint = torch.load(ckpt_path_to_load, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_metric = checkpoint.get('best_metric', -1.0)
                if 'scheduler_state_dict' in checkpoint and hasattr(lr_scheduler, 'load_state_dict'):
                     lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("LR Scheduler state loaded.")
                print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}.")
                print(f"Previous best validation loss: {best_val_loss:.4f}, best metric: {best_metric:.4f}")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting training from epoch 0.")
                start_epoch = 0; best_val_loss = float('inf'); best_metric = -1.0
        else: print("Resume specified but checkpoint not found. Starting training from epoch 0.")

    # --- 6.6. Training Model ---
    print("\n--- 6.6. Training Model ---")
    history_log = []
    log_file = os.path.join(LOG_DIR, f"training_log_{TIMESTAMP}.csv")
    try: # Ensure log directory exists
         os.makedirs(LOG_DIR, exist_ok=True)
         with open(log_file, 'w') as f:
              f.write("epoch,train_loss,val_loss,val_dice,val_iou,lr,time_epoch\n")
    except OSError as e: print(f"Warning: Could not create log directory/file: {e}")


    if model and train_loader:
        print(f"Starting training from epoch {start_epoch} up to {MAX_EPOCHS} epochs...")
        print(f"  Device: {DEVICE}")
        print(f"  Batch Size: {BATCH_SIZE}, Steps/Epoch: {len(train_loader)}")
        print(f"  Validation Steps: {len(val_loader) if val_loader else 'N/A'}")
        print(f"  Best Model Path: {BEST_MODEL_PATH}")
        print(f"  Latest Checkpoint Path: {LATEST_CHECKPOINT_PATH}")

        best_metric_epoch = -1 # Initialize best epoch tracker

        for epoch in range(start_epoch, MAX_EPOCHS):
            epoch_start_time = time.time()
            print("-" * 40); print(f"Epoch {epoch}/{MAX_EPOCHS - 1}")

            # Training Phase
            model.train(); epoch_loss = 0; step = 0
            train_pbar = tqdm(train_loader, desc="Training", leave=False)
            for batch_data in train_pbar:
                step += 1
                inputs, labels = batch_data[image_key].to(DEVICE), batch_data[label_key].to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            epoch_loss /= step
            print(f"Epoch {epoch} average training loss: {epoch_loss:.4f}")

            # Validation Phase
            val_loss = float('nan'); current_val_dice = float('nan'); current_val_iou = float('nan')
            if val_loader:
                model.eval(); val_epoch_loss = 0; val_step = 0
                dice_metric.reset(); iou_metric.reset()
                val_pbar = tqdm(val_loader, desc="Validation", leave=False)
                with torch.no_grad():
                    for val_batch_data in val_pbar:
                        val_step += 1
                        val_images, val_labels = val_batch_data[image_key].to(DEVICE), val_batch_data[label_key].to(DEVICE)
                        val_outputs = model(val_images)
                        v_loss = loss_function(val_outputs, val_labels)
                        val_epoch_loss += v_loss.item()
                        val_outputs_processed = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels_processed = [post_label(i) for i in decollate_batch(val_labels)]
                        dice_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                        iou_metric(y_pred=val_outputs_processed, y=val_labels_processed)
                        val_pbar.set_postfix({"val_loss": f"{v_loss.item():.4f}"})

                val_loss = val_epoch_loss / val_step
                current_val_dice = dice_metric.aggregate().item()
                current_val_iou = iou_metric.aggregate().item()
                print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
                print(f"Epoch {epoch} Validation Dice: {current_val_dice:.4f}")
                print(f"Epoch {epoch} Validation IoU: {current_val_iou:.4f}")

                lr_scheduler.step(val_loss)

                # Checkpointing & Early Stopping (Based on Dice Score)
                is_best_metric = current_val_dice > best_metric
                if is_best_metric:
                    best_metric = current_val_dice; best_metric_epoch = epoch
                    torch.save({
                        'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                        'best_val_loss': best_val_loss, 'best_metric': best_metric}, # Save current best loss/metric
                        BEST_MODEL_PATH
                    )
                    print(f"Saved new best model based on Dice: {best_metric:.4f} at epoch {epoch}")

                if val_loss < best_val_loss: # Check loss for early stopping counter
                     best_val_loss = val_loss
                     early_stopping_counter = 0
                     print(f"Validation loss improved to {best_val_loss:.4f}.")
                else:
                     early_stopping_counter += 1
                     print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break # Exit training loop
            else: # No validation loader
                is_best_train_loss = epoch_loss < best_val_loss # Use train loss if no validation
                if is_best_train_loss:
                     best_val_loss = epoch_loss; best_metric_epoch = epoch
                     torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                                  'best_val_loss': best_val_loss, 'best_metric': -1.0}, BEST_MODEL_PATH)
                     print(f"Saved model based on improved training loss: {best_val_loss:.4f} at epoch {epoch}")
                lr_scheduler.step(epoch_loss) # Step scheduler based on training loss

            # Save Latest Checkpoint Periodically
            if epoch % SAVE_CHECKPOINT_FREQ == 0 or epoch == MAX_EPOCHS - 1:
                 torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                              'best_val_loss': best_val_loss, 'best_metric': best_metric}, LATEST_CHECKPOINT_PATH)
                 print(f"Saved latest checkpoint at epoch {epoch}")

            # Log History
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            log_entry = {'epoch': epoch, 'train_loss': epoch_loss, 'val_loss': val_loss,
                         'val_dice': current_val_dice, 'val_iou': current_val_iou, 'lr': current_lr, 'time_epoch': epoch_time}
            history_log.append(log_entry)
            try: # Append to log file
                with open(log_file, 'a') as f:
                    f.write(f"{epoch},{epoch_loss:.6f},{val_loss:.6f},{current_val_dice:.6f},{current_val_iou:.6f},{current_lr:.8f},{epoch_time:.2f}\n")
            except IOError as e: print(f"Warning: Could not write to log file: {e}")

            print(f"Epoch {epoch} finished in {epoch_time:.2f} seconds. LR: {current_lr:.8f}")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"\n--- Model Training Finished (Stopped at epoch {epoch}) ---")
        if val_loader and best_metric_epoch != -1:
            print(f"Best Validation Dice: {best_metric:.4f} achieved at epoch {best_metric_epoch}")
            print(f"Best Validation Loss (at time of best Dice): {best_val_loss:.4f}") # Note: this might not be the absolute lowest loss if Dice improved later
        elif not val_loader and best_metric_epoch != -1:
             print(f"Best Training Loss: {best_val_loss:.4f} achieved at epoch {best_metric_epoch}")

    else: print("Skipping training: Model not built or training dataloader is empty.")

    # --- 6.7. Visualizing Training History ---
    print("\n--- 6.7. Visualizing Training History ---")
    if os.path.exists(log_file): plot_segmentation_history(log_file)
    else: print("Training log file not found. Cannot plot history.")

    # --- 6.8. Evaluating Best Model ---
    print("\n--- 6.8. Evaluating Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from: {BEST_MODEL_PATH}")
        try:
            # Re-initialize model architecture before loading state_dict
            if MODEL_NAME == "UNet": eval_model = Custom3DUNet(in_channels=1, out_channels=1, filters=MODEL_FILTERS, dropout_rate=DROPOUT_RATE, activation=ACTIVATION_FN_TYPE, leaky_slope=LEAKY_RELU_NEGATIVE_SLOPE)
            elif MODEL_NAME == "MONAI_UNet": eval_model = UNet(spatial_dims=MONAI_UNET_SPATIAL_DIMS, in_channels=1, out_channels=1, channels=MONAI_UNET_CHANNELS, strides=MONAI_UNET_STRIDES, num_res_units=MONAI_UNET_NUM_RES_UNITS, act='PRELU', norm='BATCH', dropout=DROPOUT_RATE)
            else: raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

            checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            eval_model.load_state_dict(checkpoint['model_state_dict'])
            eval_model.to(DEVICE); eval_model.eval()
            print("Best model loaded successfully for evaluation.")

            best_epoch_eval = checkpoint.get('epoch', 'N/A'); best_metric_eval = checkpoint.get('best_metric', np.nan); best_loss_eval = checkpoint.get('best_val_loss', np.nan)
            print(f"Best model was saved at Epoch: {best_epoch_eval}, Metric (Dice): {best_metric_eval:.4f}, Loss: {best_loss_eval:.4f}")

            if val_loader:
                print("Evaluating best model on validation set...")
                eval_dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False); eval_iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
                eval_loss_total = 0.0; eval_steps = 0
                eval_pbar = tqdm(val_loader, desc="Evaluating Best Model", leave=False)
                with torch.no_grad():
                    for eval_batch_data in eval_pbar:
                        eval_steps += 1
                        eval_images, eval_labels = eval_batch_data[image_key].to(DEVICE), eval_batch_data[label_key].to(DEVICE)
                        eval_outputs = eval_model(eval_images)
                        eval_loss = loss_function(eval_outputs, eval_labels); eval_loss_total += eval_loss.item()
                        eval_outputs_processed = [post_pred(i) for i in decollate_batch(eval_outputs)]
                        eval_labels_processed = [post_label(i) for i in decollate_batch(eval_labels)]
                        eval_dice_metric(y_pred=eval_outputs_processed, y=eval_labels_processed)
                        eval_iou_metric(y_pred=eval_outputs_processed, y=eval_labels_processed)

                final_avg_loss = eval_loss_total / eval_steps
                final_dice = eval_dice_metric.aggregate().item()
                final_iou = eval_iou_metric.aggregate().item()
                print("\nValidation Results (Best Model):")
                print(f"  Average Loss: {final_avg_loss:.4f}")
                print(f"  Average Dice: {final_dice:.4f}")
                print(f"  Average Mean IoU: {final_iou:.4f}")
                del eval_model, checkpoint # Clean up model from eval
            else: print("No validation data available to evaluate the best model.")
        except Exception as e: print(f"Error loading or evaluating best model: {e}"); traceback.print_exc()
    else: print("Best model checkpoint not found. Skipping evaluation.")

    # --- 6.9. Predicting and Visualizing Sample ---
    print("\n--- 6.9. Predicting and Visualizing Sample ---")
    # Predict only if best model was loaded/re-loaded and validation data exists
    eval_model_loaded_for_viz = False
    if os.path.exists(BEST_MODEL_PATH) and val_loader:
        try:
             # Load the model again specifically for visualization if it was deleted after eval
             if 'eval_model' not in locals() or eval_model is None:
                  print(f"Reloading best model for visualization from: {BEST_MODEL_PATH}")
                  if MODEL_NAME == "UNet": eval_model = Custom3DUNet(in_channels=1, out_channels=1, filters=MODEL_FILTERS, dropout_rate=DROPOUT_RATE, activation=ACTIVATION_FN_TYPE, leaky_slope=LEAKY_RELU_NEGATIVE_SLOPE)
                  elif MODEL_NAME == "MONAI_UNet": eval_model = UNet(spatial_dims=MONAI_UNET_SPATIAL_DIMS, in_channels=1, out_channels=1, channels=MONAI_UNET_CHANNELS, strides=MONAI_UNET_STRIDES, num_res_units=MONAI_UNET_NUM_RES_UNITS, act='PRELU', norm='BATCH', dropout=DROPOUT_RATE)
                  checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
                  eval_model.load_state_dict(checkpoint['model_state_dict'])
                  eval_model.to(DEVICE); eval_model.eval()
                  eval_model_loaded_for_viz = True
                  print("Best model reloaded successfully.")
             else:
                  eval_model.eval() # Ensure model is in eval mode if it still exists
                  eval_model_loaded_for_viz = True

             if eval_model_loaded_for_viz:
                  print("Visualizing prediction on a validation sample...")
                  val_iterator = iter(val_loader)
                  sample_batch = next(val_iterator)
                  sample_image = sample_batch[image_key][0]
                  true_mask = sample_batch[label_key][0]
                  print(f"\nPredicting on validation sample...")
                  print(f"  Input Image Shape: {sample_image.shape}")
                  print(f"  True Mask Shape: {true_mask.shape}")
                  sample_image_batch = sample_image.unsqueeze(0).to(DEVICE)
                  with torch.no_grad(): predicted_logits = eval_model(sample_image_batch)[0]
                  predicted_mask_processed = post_pred(predicted_logits)
                  print(f"  Predicted Mask Shape (processed): {predicted_mask_processed.shape}")
                  print(f"  Predicted Mask Unique Values: {torch.unique(predicted_mask_processed)}")
                  slice_to_show = sample_image.shape[1] // 2 # Depth axis 1
                  display_slice_comparison(sample_image, true_mask, predicted_mask_processed, slice_to_show, main_title="Validation Sample Prediction (Best Model)")
                  del sample_batch, sample_image, true_mask, sample_image_batch, predicted_logits, predicted_mask_processed, val_iterator # Clean up viz tensors
        except StopIteration: print("Validation dataset iterator exhausted. Cannot get sample.")
        except NameError: print("Skipping prediction/visualization: 'eval_model' not defined (check eval step).")
        except Exception as e: print(f"An error occurred during prediction or visualization: {e}"); traceback.print_exc()
        finally:
            # Clean up the model loaded specifically for visualization if necessary
            if 'eval_model' in locals() and eval_model is not None:
                del eval_model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("Skipping prediction/visualization: Best model not found or no validation data.")


    print("\n--- Script Finished ---")

# <<< End of if __name__ == '__main__': block >>>