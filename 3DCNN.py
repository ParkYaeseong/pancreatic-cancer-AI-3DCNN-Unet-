# -*- coding: utf-8 -*-
# ==============================================================================
# Pancreas Cancer Classification Training Script (for Local Execution with GPU)
# ==============================================================================

# ==============================================================================
# 0. Check Environment (Optional but Recommended)
# ==============================================================================
import sys
IN_COLAB = 'google.colab' in sys.modules

# ==============================================================================
# 1. Imports (Keep all imports at the top level)
# ==============================================================================
import os
import zipfile
import tarfile
import json
import nibabel as nib
import numpy as np
import pandas as pd
# from scipy.ndimage import zoom # No longer needed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score # For classification metrics
import torch
import torch.nn as nn
import torch.nn.functional as F # Added for layers like AdaptiveAvgPool3d, Flatten
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader # Use MONAI DataLoader
import monai
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch, list_data_collate # Added list_data_collate
# from monai.losses import DiceCELoss # Changed to BCEWithLogitsLoss
# from monai.metrics import DiceMetric, MeanIoU # Changed to classification metrics
from monai.metrics import ROCAUCMetric # Use MONAI's AUC metric
from monai.networks.nets import UNet # Keep for potential encoder reuse, or remove if not using
from monai.transforms import (
    Activations, AsDiscrete, Compose, EnsureChannelFirstd, LoadImaged,
    Orientationd, RandFlipd, RandRotate90d, RandShiftIntensityd, RandGaussianNoised,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped, Spacingd,
    EnsureType, # Added EnsureType for label tensor conversion
    Lambda # Added Lambda for custom label processing
)
from monai.utils import set_determinism, first
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import datetime
import traceback
from collections import Counter
import time
try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False


# ==============================================================================
# 2. Configuration (Updated for Classification)
# ==============================================================================

# --- 경로 설정 ---
DRIVE_BASE_PATH = 'C:/Users/21/Desktop/췌장암' # <<<--- 사용자의 실제 로컬 데이터 폴더 경로로 반드시 수정하세요!
BASE_PATH = './'
TAR_CANCER_PATH = os.path.join(DRIVE_BASE_PATH, 'Task07_Pancreas.tar')
ZIP_NORMAL_PATH = os.path.join(DRIVE_BASE_PATH, 't1.zip')

# --- 작업 경로 ---
WORK_DIR = os.path.join(BASE_PATH, 'pancreas_classification_project') # Updated project name
EXTRACT_BASE_PATH = os.path.join(WORK_DIR, 'data')
CANCER_EXTRACT_PATH = os.path.join(EXTRACT_BASE_PATH, 'Task07_Pancreas_extracted') # Renamed
NORMAL_EXTRACT_PATH = os.path.join(EXTRACT_BASE_PATH, 't1_extracted') # Renamed
CANCER_EXPECTED_SUBFOLDER = "Task07_Pancreas"
NORMAL_EXPECTED_SUBFOLDER = "t1"

# --- MONAI 데이터 처리 파라미터 ---
TARGET_SPACING = (1.5, 1.5, 2.0)    # Voxel Spacing (X, Y, Z) - From Colab
TARGET_SPATIAL_SHAPE = (96, 96, 96) # Resize shape (D, H, W) - From Colab
HU_WINDOW = (-57, 164) # HU Windowing 
NUM_CLASSES = 1        # ** 변경: 이진 분류 (암=1, 정상=0)
CACHE_DATASET = True
CACHE_RATE = 1.0
NUM_WORKERS = 4
IMAGE_KEY = "image"
LABEL_KEY = "class_label" # ** 변경: 키 이름을 클래스 레이블로 변경
IMAGE_SUBFOLDER = "imagesTr" # 이미지 파일 하위 폴더

# --- 학습 파라미터 ---
MAX_EPOCHS = 50 # 분류는 Segmentation보다 빠르게 수렴할 수 있으므로 줄여서 시작 가능
BATCH_SIZE = 4 # 분류는 일반적으로 Segmentation보다 메모리를 적게 사용하므로 늘릴 수 있음 (GPU 메모리 확인 필요)
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
BALANCE_DATA = True # 데이터 불균형 중요
EARLY_STOPPING_PATIENCE = 10 # AUC 기준으로 조기 종료 patience
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2
MIN_LR = 1e-7

# --- 모델 파라미터 ---
MODEL_NAME = "Simple3DClassifier" # ** 변경: 모델 이름 지정
# Simple3DClassifier Parameters
CLASSIFIER_CHANNELS = [16, 32, 64, 128] # 인코더 채널 수
CLASSIFIER_STRIDES = [(2, 2, 2)] * len(CLASSIFIER_CHANNELS) # 각 Conv 후 풀링 대신 stride 사용 가능
CLASSIFIER_FC_DIMS = [256] # 분류기 FC 레이어 차원 (마지막 출력 제외)
CLASSIFIER_DROPOUT = 0.2 # 분류기 드롭아웃 비율

# --- 결과 저장 및 체크포인트 경로 ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = os.path.join(WORK_DIR, "outputs_classification", TIMESTAMP) # Updated output dir
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f"best_model_auc_{TIMESTAMP}.pth") # ** 변경: AUC 기준 저장
LATEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
SAVE_CHECKPOINT_FREQ = 5
RESUME_CHECKPOINT = None

# --- 디바이스 설정 ---
DEVICE = None


# ==============================================================================
# 3. Utility Functions (Data preparation function updated)
# ==============================================================================

# 압축 해제 함수 
def unzip_data(archive_path, extract_to):
    """압축 해제 함수 (zip/tar 지원, 폴더 존재 시 건너뛰기, 경로 안전성 강화)"""
    print(f"\nChecking extraction for '{os.path.basename(archive_path)}' to '{extract_to}'...")
    archive_abs_path = os.path.abspath(archive_path)
    extract_to_abs_path = os.path.abspath(extract_to)

    if not os.path.exists(archive_abs_path):
        print(f"  Error: Archive file not found: '{archive_abs_path}'. Cannot extract.")
        return False

    try:
        # Check if extraction target exists and is not empty
        if os.path.exists(extract_to_abs_path) and os.listdir(extract_to_abs_path):
            print(f"  Directory '{extract_to_abs_path}' already exists and is not empty. Skipping extraction.")
            return True
        elif os.path.exists(extract_to_abs_path):
             print(f"  Directory '{extract_to_abs_path}' exists but is empty. Proceeding with extraction.")
        else:
             os.makedirs(extract_to_abs_path, exist_ok=True)
             print(f"  Created directory '{extract_to_abs_path}'.")
    except OSError as e:
        print(f"  Error checking/creating directory {extract_to_abs_path}: {e}. Cannot extract.")
        return False

    print(f"  Starting extraction from '{archive_abs_path}'...")
    extracted_ok = False
    try:
        if archive_path.lower().endswith('.zip'):
            with zipfile.ZipFile(archive_abs_path, 'r') as zip_ref:
                members = zip_ref.infolist()
                for member in tqdm(members, desc=f'  Extracting {os.path.basename(archive_path)} (zip)'):
                    try:
                        target_path = os.path.join(extract_to_abs_path, member.filename)
                        target_abs_path = os.path.abspath(target_path)
                        if not target_abs_path.startswith(extract_to_abs_path):
                            print(f"  Warning: Skipping potentially unsafe path in zip: {member.filename}")
                            continue
                        zip_ref.extract(member, extract_to_abs_path)
                    except Exception as e:
                        print(f"  Warning: Could not extract {member.filename} from zip. Error: {e}")
                extracted_ok = True

        elif archive_path.lower().endswith(('.tar', '.tar.gz', '.tgz', '.gz')): # Added .gz for robustness
            filter_arg = {}
            if sys.version_info >= (3, 12): filter_arg['filter'] = 'data'
            elif sys.version_info >= (3, 8): filter_arg['numeric_owner'] = False

            with tarfile.open(archive_abs_path, 'r:*') as tar_ref:
                safe_members = []
                for member in tar_ref.getmembers():
                    try:
                        target_path = os.path.join(extract_to_abs_path, member.name)
                        target_abs_path = os.path.abspath(target_path)
                        if not target_abs_path.startswith(extract_to_abs_path):
                             print(f"  Warning: Skipping potentially unsafe path in tar: {member.name}")
                             continue
                        safe_members.append(member)
                    except Exception as path_e:
                        print(f"  Warning: Could not process member {member.name} due to path error: {path_e}")

                for member in tqdm(safe_members, desc=f'  Extracting {os.path.basename(archive_path)} (tar)'):
                     try:
                        tar_ref.extract(member, path=extract_to_abs_path, set_attrs=False, **filter_arg)
                     except Exception as e:
                        print(f"  Warning: Could not extract {member.name} from tar. Error: {e}")
                extracted_ok = True
        else:
            print(f"  Error: Unsupported archive format for '{archive_path}'. Only .zip and .tar(.gz) are supported.")
            return False

        if extracted_ok: print(f"  Successfully finished extraction attempt.")
        return True

    except (zipfile.BadZipFile, tarfile.TarError) as archive_err:
        print(f"  Error: Archive file '{archive_path}' is corrupted or invalid: {archive_err}"); traceback.print_exc(); return False
    except Exception as e:
        print(f"  Error during extraction process: {e}"); traceback.print_exc(); return False


# 경로 조정 함수 (이전과 동일)
def find_data_root_after_extraction(extract_path, expected_subfolder=""):
    """압축 해제 후 실제 데이터 루트 폴더 찾기 (예상 폴더 우선 확인)"""
    print(f"  Finding actual data root within: {extract_path}")
    if not extract_path or not os.path.isdir(extract_path):
        print("    Error: Extraction path does not exist or is not a directory.")
        return None
    try:
        items = [d for d in os.listdir(extract_path) if not d.startswith('.') and not d.startswith('__MACOSX')]

        if expected_subfolder:
            potential_root = os.path.join(extract_path, expected_subfolder)
            if os.path.isdir(potential_root):
                print(f"    Found expected subfolder: {potential_root}")
                return potential_root

        if len(items) == 1 and os.path.isdir(os.path.join(extract_path, items[0])):
            adjusted_root = os.path.join(extract_path, items[0])
            print(f"    Found single inner folder: {adjusted_root}")
            return adjusted_root

        print(f"    Using original extraction path as root: {extract_path}")
        return extract_path
    except Exception as e:
        print(f"    Error adjusting path: {e}. Using original path: {extract_path}")
        return extract_path


class EnsureLabelTensorShape:
    """
    MONAI Transform: 클래스 레이블이 [1,] 형태의 float32 텐서인지 확인하고 변환합니다.
    데이터 로더의 multiprocessing에서 lambda 대신 사용하기 위함 (pickle 가능).
    """
    def __init__(self, label_key="class_label"):
        self.label_key = label_key

    def __call__(self, data):
        # 원본 딕셔너리 수정을 피하기 위해 복사본 사용 (선택 사항)
        d = dict(data)
        label = d.get(self.label_key)

        if label is not None:
            # 텐서가 아니면 텐서로 변환
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
            # float 타입과 [1,] 형태로 변환
            label = label.float().reshape(1)
            d[self.label_key] = label
        else:
            # 레이블 키가 없는 경우 경고 (필요시 처리)
            print(f"Warning: Label key '{self.label_key}' not found in data dictionary: {d.get(IMAGE_KEY+'_meta_dict', {}).get('filename_or_obj', 'N/A')}")

        # 이미지 키는 EnsureTyped에서 처리되었으므로 여기서는 레이블만 처리
        return d

# 학습 history 시각화 함수 (** 수정: 분류 지표(AUC, Acc) 플롯 추가 **)
def plot_classification_history(log_file_path, save_dir):
    """로그 파일(CSV)에서 분류 학습 history 읽어 그래프 출력 및 저장"""
    print(f"\nPlotting classification training history from: {log_file_path}")
    plot_save_path = os.path.join(save_dir, "classification_training_history.png")
    try:
        if not os.path.exists(log_file_path):
             print(f"  Error: Log file not found at {log_file_path}. Cannot plot history.")
             return

        history_df = pd.read_csv(log_file_path)
        if history_df.empty:
            print("  Log file is empty. No history to plot.")
            return

        plt.style.use('seaborn-v0_8-darkgrid')
        epochs = history_df['epoch'] + 1 # 1-based epochs

        # Check available columns
        has_train_loss = 'train_loss' in history_df.columns and history_df['train_loss'].notna().any()
        has_val_loss = 'val_loss' in history_df.columns and history_df['val_loss'].notna().any()
        has_val_auc = 'val_auc' in history_df.columns and history_df['val_auc'].notna().any()
        has_val_acc = 'val_accuracy' in history_df.columns and history_df['val_accuracy'].notna().any()

        num_plots = sum([(has_train_loss or has_val_loss), has_val_auc, has_val_acc])
        if num_plots == 0:
            print("  No plottable metrics (loss, auc, accuracy) found in log file.")
            return

        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), squeeze=False)
        axes = axes.flatten()
        plot_idx = 0
        font_dict = {'fontsize': 12}

        # Plot 1: Loss
        if has_train_loss or has_val_loss:
            ax = axes[plot_idx]
            if has_train_loss: ax.plot(epochs, history_df['train_loss'], label='Training Loss', lw=2, marker='o', markersize=4)
            if has_val_loss: ax.plot(epochs, history_df['val_loss'], label='Validation Loss', lw=2, marker='s', markersize=4)
            ax.set_title('Model Loss', **font_dict); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.legend(); ax.grid(True, linestyle=':')
            plot_idx += 1

        # Plot 2: AUC
        if has_val_auc:
            ax = axes[plot_idx]
            ax.plot(epochs, history_df['val_auc'], label='Validation AUC', lw=2, color='green', marker='^', markersize=4)
            ax.set_title('Validation ROC AUC', **font_dict); ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
            if history_df['val_auc'].notna().any():
                best_auc_epoch = history_df['val_auc'].idxmax()
                best_auc_val = history_df['val_auc'].max()
                ax.scatter(epochs[best_auc_epoch], best_auc_val, color='red', s=100, label=f'Best AUC: {best_auc_val:.4f}\n(Epoch {epochs[best_auc_epoch]})', zorder=5)
            ax.legend(); ax.grid(True, linestyle=':')
            ax.set_ylim(bottom=max(0, history_df['val_auc'].min() - 0.1 if history_df['val_auc'].notna().any() else 0), top=1.0)
            plot_idx += 1

        # Plot 3: Accuracy
        if has_val_acc:
            ax = axes[plot_idx]
            ax.plot(epochs, history_df['val_accuracy'], label='Validation Accuracy', lw=2, color='purple', marker='d', markersize=4)
            ax.set_title('Validation Accuracy', **font_dict); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
            if history_df['val_accuracy'].notna().any():
                best_acc_epoch = history_df['val_accuracy'].idxmax()
                best_acc_val = history_df['val_accuracy'].max()
                ax.scatter(epochs[best_acc_epoch], best_acc_val, color='blue', s=80, label=f'Best Acc: {best_acc_val:.4f}\n(Epoch {epochs[best_acc_epoch]})', zorder=5)
            ax.legend(); ax.grid(True, linestyle=':')
            ax.set_ylim(bottom=max(0, history_df['val_accuracy'].min() - 0.1 if history_df['val_accuracy'].notna().any() else 0), top=1.0)
            plot_idx += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle("Classification Training History", fontsize=16, y=0.98)

        try:
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            print(f"  Classification history plot saved to: {plot_save_path}")
        except Exception as save_err:
            print(f"  Warning: Could not save classification plot: {save_err}")

        plt.show()

    except pd.errors.EmptyDataError: print("  Error: Log file is empty or corrupted.")
    except KeyError as e: print(f"  Error: Missing expected column in log file: {e}")
    except Exception as e: print(f"  Error plotting history from log file: {e}"); traceback.print_exc()


# 데이터 준비 및 분할 함수 (** 수정: 분류 작업에 맞게 변경 **)
def prepare_classification_files(cancer_extract_root, normal_extract_root,
                                 cancer_expected_subfolder, normal_expected_subfolder,
                                 image_key="image", label_key="class_label", # Use label_key for class
                                 image_folder="imagesTr",
                                 balance=True, test_size=0.2, random_state=42):
    """암/정상 폴더에서 이미지 파일을 찾아 클래스 레이블과 함께 파일 리스트 생성 및 분할"""
    print("\n--- Preparing Data Files for Classification ---")
    all_files = []
    processed_counts = {'cancer': 0, 'normal': 0}
    skipped_counts = {'cancer': 0, 'normal': 0}

    # --- 데이터 루트 경로 조정 ---
    cancer_root_adj = find_data_root_after_extraction(cancer_extract_root, cancer_expected_subfolder)
    normal_root_adj = find_data_root_after_extraction(normal_extract_root, normal_expected_subfolder)

    print(f"\nAdjusted Data Roots:")
    print(f"  Cancer Root: {cancer_root_adj} (Exists: {os.path.isdir(cancer_root_adj) if cancer_root_adj else False})")
    print(f"  Normal Root: {normal_root_adj} (Exists: {os.path.isdir(normal_root_adj) if normal_root_adj else False})")

    # --- 데이터 파일 찾기 (암: 1, 정상: 0) ---
    def find_images_with_label(root_dir, img_subfolder, class_val):
        """주어진 루트와 하위 폴더에서 이미지 파일을 찾아 클래스 레이블과 함께 리스트 반환"""
        file_list = []
        count = 0
        skipped = 0
        if not root_dir:
            print(f"  Skipping class {class_val}: Root directory is invalid.")
            return file_list, count, skipped

        search_path = os.path.join(root_dir, img_subfolder)
        print(f"  Searching for class {class_val} images in: {search_path}")
        if not os.path.isdir(search_path):
            print(f"  Warning: Image subfolder '{img_subfolder}' not found in '{root_dir}' for class {class_val}.")
            return file_list, count, skipped

        try:
            for filename in os.listdir(search_path):
                if filename.lower().endswith(('.nii', '.nii.gz')) and not filename.startswith('.'):
                    img_path = os.path.join(search_path, filename)
                    if os.path.isfile(img_path):
                        file_list.append({image_key: img_path, label_key: class_val})
                        count += 1
                    else:
                        skipped += 1
        except Exception as e:
            print(f"  Error finding images in '{search_path}': {e}")
            skipped = len(os.listdir(search_path)) # Assume all failed if error during listing
        print(f"  Found {count} images for class {class_val} (Skipped/Invalid: {skipped}).")
        return file_list, count, skipped

    # 암 데이터 (Class 1)
    cancer_files, processed_counts['cancer'], skipped_counts['cancer'] = find_images_with_label(
        cancer_root_adj, image_folder, 1
    )
    all_files.extend(cancer_files)

    # 정상 데이터 (Class 0)
    normal_files, processed_counts['normal'], skipped_counts['normal'] = find_images_with_label(
        normal_root_adj, image_folder, 0
    )
    all_files.extend(normal_files)


    # --- Summary and Data Splitting ---
    n_cancer = processed_counts['cancer']
    n_normal = processed_counts['normal']
    n_total = n_cancer + n_normal

    print("\n--- Data Preparation Summary ---")
    print(f"  Total valid image files found: {n_total}")
    print(f"    Cancer (Class 1): {n_cancer} (Skipped: {skipped_counts['cancer']})")
    print(f"    Normal (Class 0): {n_normal} (Skipped: {skipped_counts['normal']})")

    if n_total == 0:
        print("\nError: No valid data files collected. Cannot proceed.")
        return [], []

    labels = [f[label_key] for f in all_files] # Get class labels (0 or 1)

    # --- Data Balancing (Oversampling) ---
    # (Oversampling logic remains the same, just uses 'class_label')
    if balance and n_cancer > 0 and n_normal > 0 and n_cancer != n_normal:
        print(f"\nBalancing data via oversampling...")
        # Separate files by class
        cancer_files_list = [f for f in all_files if f[label_key] == 1]
        normal_files_list = [f for f in all_files if f[label_key] == 0]

        if n_cancer < n_normal:
            minority_files, majority_files = cancer_files_list, normal_files_list
            minority_class_name, majority_class_name = "Cancer", "Normal"
        else:
            minority_files, majority_files = normal_files_list, cancer_files_list
            minority_class_name, majority_class_name = "Normal", "Cancer"

        n_minority, n_majority = len(minority_files), len(majority_files)

        print(f"  Minority class: {minority_class_name} ({n_minority} samples)")
        print(f"  Majority class: {majority_class_name} ({n_majority} samples)")
        print(f"  Oversampling {minority_class_name} class by {n_majority - n_minority} samples...")

        rng = np.random.RandomState(random_state)
        oversample_indices = rng.choice(n_minority, size=n_majority - n_minority, replace=True)
        oversampled_files = [minority_files[i] for i in oversample_indices]

        all_files = majority_files + minority_files + oversampled_files
        labels = [f[label_key] for f in all_files] # Update labels list
        print(f"  Total files after balancing: {len(all_files)}")
        print(f"  Balanced counts - Cancer: {sum(l==1 for l in labels)}, Normal: {sum(l==0 for l in labels)}")
    elif balance and (n_cancer == 0 or n_normal == 0):
        print("\nWarning: Cannot balance data - only one class has samples.")
    elif balance and n_cancer == n_normal:
         print("\nData is already balanced.")
    else:
        print("\nData balancing is disabled.")

    # --- Data Splitting (Train/Validation) ---
    # (Splitting logic remains the same, uses class labels for stratification)
    print("\n--- Splitting Data (Train/Validation) ---")
    train_files, val_files = [], []
    if not all_files: print("  Error: No files to split."); return [], []
    if test_size <= 0 or test_size >= 1 or len(all_files) < 2:
        print("  Warning: Invalid validation_split size or too few samples. Using all data for training.")
        train_files = all_files; val_files = []
    else:
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            can_stratify = len(unique_labels) >= 2 and all(c >= 2 for c in counts)
            stratify_param = labels if can_stratify else None
            split_type = "stratified" if can_stratify else "regular"
            print(f"  Performing {split_type} split (Test Size: {test_size}, Random State: {random_state}).")
            train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state, stratify=stratify_param)
        except ValueError as e:
            print(f"  Warning: Error during {split_type} split: {e}. Falling back to regular split.")
            train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state, stratify=None)

    print("\nSplitting complete:")
    print(f"  Training files: {len(train_files)}")
    print(f"  Validation files: {len(val_files)}")
    if train_files: print(f"    Training distribution - Cancer (1): {sum(f[label_key]==1 for f in train_files)}, Normal (0): {sum(f[label_key]==0 for f in train_files)}")
    if val_files: print(f"    Validation distribution - Cancer (1): {sum(f[label_key]==1 for f in val_files)}, Normal (0): {sum(f[label_key]==0 for f in val_files)}")

    return train_files, val_files


# Activation 함수 반환 헬퍼 (이전과 동일)
def get_activation(activation_type, negative_slope=0.01):
    if activation_type.lower() == 'relu': return nn.ReLU(inplace=True)
    elif activation_type.lower() == 'leaky_relu': return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    elif activation_type.lower() == 'prelu': return nn.PReLU()
    else: raise ValueError(f"Unsupported activation function type: {activation_type}")


# ==============================================================================
# 5. Model Definition (** 변경: 3D CNN 분류기 **)
# ==============================================================================

class Simple3DClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, channels=None, strides=None, fc_dims=None, dropout=0.2):
        super().__init__()
        if channels is None: channels = [16, 32, 64, 128]
        if strides is None: strides = [(2, 2, 2)] * len(channels) # Default stride 2 for pooling effect
        if fc_dims is None: fc_dims = [256]
        if len(channels) != len(strides):
            raise ValueError("Length of channels and strides must match")

        self.encoder = nn.Sequential()
        current_channels = in_channels
        print("\nBuilding Simple 3D Classifier Encoder:")
        for i, (ch, st) in enumerate(zip(channels, strides)):
            self.encoder.add_module(f"conv{i+1}", nn.Conv3d(current_channels, ch, kernel_size=3, stride=st, padding=1, bias=False))
            self.encoder.add_module(f"bn{i+1}", nn.BatchNorm3d(ch))
            self.encoder.add_module(f"relu{i+1}", nn.ReLU(inplace=True))
            # Optional: Add MaxPool3d here instead of stride if preferred
            # self.encoder.add_module(f"pool{i+1}", nn.MaxPool3d(kernel_size=2, stride=2))
            print(f"  Encoder Block {i+1}: In={current_channels}, Out={ch}, Stride={st}")
            current_channels = ch

        # Global pooling to get fixed-size vector
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        print(f"Added AdaptiveAvgPool3d")

        self.flatten = nn.Flatten()

        # Fully Connected Classifier Head
        self.classifier = nn.Sequential()
        print("Building Classifier Head:")
        in_features = current_channels # Features after pooling and flattening
        for i, hidden_dim in enumerate(fc_dims):
            self.classifier.add_module(f"fc{i+1}", nn.Linear(in_features, hidden_dim))
            self.classifier.add_module(f"fc_relu{i+1}", nn.ReLU(inplace=True))
            if dropout > 0:
                self.classifier.add_module(f"fc_dropout{i+1}", nn.Dropout(dropout))
            print(f"  FC Layer {i+1}: In={in_features}, Out={hidden_dim}, Dropout={dropout if dropout > 0 else 0.0}")
            in_features = hidden_dim

        # Final Output Layer (single logit for BCEWithLogitsLoss)
        self.classifier.add_module("fc_out", nn.Linear(in_features, num_classes))
        print(f"  Output Layer: In={in_features}, Out={num_classes}")


    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


# ==============================================================================
# 4. Data Loading & Preprocessing Definition (** 변경: 분류에 맞게 수정 **)
# ==============================================================================

# --- Training Transforms ---
train_transforms = Compose(
    [
        LoadImaged(keys=[IMAGE_KEY]),
        EnsureChannelFirstd(keys=[IMAGE_KEY]),
        Spacingd(keys=[IMAGE_KEY], pixdim=TARGET_SPACING, mode="bilinear"),
        Orientationd(keys=[IMAGE_KEY], axcodes="RAS"),
        ScaleIntensityRanged(keys=[IMAGE_KEY], a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=[IMAGE_KEY], source_key=IMAGE_KEY, margin=10),
        Resized(keys=[IMAGE_KEY], spatial_size=TARGET_SPATIAL_SHAPE, mode="area"),
        # --- Augmentations (Only on Image) ---
        RandFlipd(keys=[IMAGE_KEY], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=[IMAGE_KEY], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=[IMAGE_KEY], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=[IMAGE_KEY], prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandShiftIntensityd(keys=[IMAGE_KEY], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=[IMAGE_KEY], prob=0.15, mean=0.0, std=0.05),
        EnsureTyped(keys=[IMAGE_KEY], dtype=torch.float32), # 이미지만 타입 변환
        # ** 변경: Lambda 대신 정의된 클래스 사용 **
        EnsureLabelTensorShape(label_key=LABEL_KEY),
    ]
)


# --- Validation Transforms ---
val_transforms = Compose(
    [
        LoadImaged(keys=[IMAGE_KEY]),
        EnsureChannelFirstd(keys=[IMAGE_KEY]),
        Spacingd(keys=[IMAGE_KEY], pixdim=TARGET_SPACING, mode="bilinear"),
        Orientationd(keys=[IMAGE_KEY], axcodes="RAS"),
        ScaleIntensityRanged(keys=[IMAGE_KEY], a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=[IMAGE_KEY], source_key=IMAGE_KEY, margin=10),
        Resized(keys=[IMAGE_KEY], spatial_size=TARGET_SPATIAL_SHAPE, mode="area"),
        EnsureTyped(keys=[IMAGE_KEY], dtype=torch.float32),
        # ** 변경: Lambda 대신 정의된 클래스 사용 **
        EnsureLabelTensorShape(label_key=LABEL_KEY),
    ]
)


# ==============================================================================
# Loss & Metrics (** 변경: 분류용 Loss 및 Metrics **)
# ==============================================================================
# Use BCEWithLogitsLoss for binary classification with single logit output
loss_function = nn.BCEWithLogitsLoss()

# Post-processing for metrics: Apply Sigmoid, then threshold
# Input is raw logit from model (Batch, 1)
# Output is binary prediction (Batch, 1)
post_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
# Labels are already 0 or 1, just ensure correct type/shape if needed (handled in transform)
# post_label = Compose([EnsureType()]) # Not strictly needed if label is already tensor

# Define metrics for classification
# MONAI's ROCAUC metric accumulates stats internally
auc_metric = ROCAUCMetric()


# ==============================================================================
# Main Execution Block - Guarded for Multiprocessing
# ==============================================================================
if __name__ == '__main__':

    # Add freeze_support() for Windows multiprocessing compatibility
    from multiprocessing import freeze_support
    freeze_support()

    print("="*60)
    print(" Pancreas Cancer Classification Training Script (Local GPU)")
    print("="*60)
    # ... [Setup & Config Printing - Largely the same, check paths] ...
    # (System Check, Configuration Summary as before, ensure paths are correct)
    print(f"Running in Google Colab: {IN_COLAB}")
    print("--- 1. System & Library Check ---")
    # ... (library version checks) ...
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA device count: {torch.cuda.device_count()}")
        print(f"    Current CUDA device: {torch.cuda.current_device()}")
        print(f"    Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else: print("    CUDA not available, selecting CPU.")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Selected device: {DEVICE}")
    set_determinism(seed=RANDOM_STATE)
    print(f"  Determinism set with seed: {RANDOM_STATE}")

    print("\n--- 2. Configuration Summary ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"  Output Dir: {OUTPUT_DIR}")
    # ... (Print other configs: Data, Training, Model, Checkpointing) ...
    print(f"  Model Name: {MODEL_NAME}")
    print(f"    Encoder Channels: {CLASSIFIER_CHANNELS}")
    print(f"    Encoder Strides: {CLASSIFIER_STRIDES}")
    print(f"    FC Hidden Dims: {CLASSIFIER_FC_DIMS}")
    print(f"    Dropout: {CLASSIFIER_DROPOUT}")
    print(f"    Output Classes: {NUM_CLASSES}")


    # ==============================================================================
    # 6. Main Execution Steps (Inside the Guard)
    # ==============================================================================
    print("\n--- 6. Starting Main Execution ---")

    # --- 6.1. Unzipping Data ---
    print("\n--- 6.1. Unzipping Data ---")
    # ... (Unzip logic remains the same) ...
    unzip_success_cancer = unzip_data(TAR_CANCER_PATH, CANCER_EXTRACT_PATH)
    unzip_success_normal = unzip_data(ZIP_NORMAL_PATH, NORMAL_EXTRACT_PATH)
    if not unzip_success_cancer and not unzip_success_normal: raise RuntimeError("Failed to extract datasets.")


    # --- 6.2. Preparing Data File List (** 변경: 분류용 함수 호출 **) ---
    print("\n--- 6.2. Preparing and Splitting Data Files (for Classification) ---")
    train_files, val_files = prepare_classification_files(
        cancer_extract_root=CANCER_EXTRACT_PATH,
        normal_extract_root=NORMAL_EXTRACT_PATH,
        cancer_expected_subfolder=CANCER_EXPECTED_SUBFOLDER,
        normal_expected_subfolder=NORMAL_EXPECTED_SUBFOLDER,
        image_key=IMAGE_KEY, label_key=LABEL_KEY,
        image_folder=IMAGE_SUBFOLDER,
        balance=BALANCE_DATA,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE
    )

    if not train_files: raise ValueError("Failed to prepare training files.")
    if not val_files and VALIDATION_SPLIT > 0: print("Warning: Validation files list is empty.")


    # --- 6.3. Creating MONAI Datasets and DataLoaders ---
    print("\n--- 6.3. Creating MONAI Datasets and DataLoaders ---")
    # ... (Dataset/DataLoader creation logic largely same, but uses updated transforms) ...
    train_ds, val_ds = None, None
    train_loader, val_loader = None, None
    try:
        if train_files:
            dataset_class = CacheDataset if CACHE_DATASET else Dataset
            cache_args = {'cache_rate': CACHE_RATE, 'num_workers': NUM_WORKERS} if CACHE_DATASET else {}
            print(f"Creating {dataset_class.__name__} for training...")
            train_ds = dataset_class(data=train_files, transform=train_transforms, **cache_args)
            loader_num_workers = 0 if (CACHE_DATASET and CACHE_RATE == 1.0) else NUM_WORKERS
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=loader_num_workers, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate) # Use MONAI collate
            print(f"Training DataLoader created. Samples: {len(train_ds)}, Batches/Epoch: {len(train_loader)}")

        if val_files:
            print("Creating regular Dataset for validation...")
            val_ds = Dataset(data=val_files, transform=val_transforms)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate)
            print(f"Validation DataLoader created. Samples: {len(val_ds)}, Batches/Epoch: {len(val_loader)}")
        else: print("No validation files, skipping validation dataset/loader creation.")
    except Exception as e: print(f"Error creating MONAI datasets/dataloaders: {e}"); traceback.print_exc(); raise


    # --- (Optional) Visualize a sample (** 수정: 레이블 확인 **) ---
    if train_loader and len(train_loader)>0 :
        print("\n--- Checking a sample from training dataloader ---")
        try:
            check_batch = first(train_loader) # Use MONAI's first utility
            if check_batch and IMAGE_KEY in check_batch and LABEL_KEY in check_batch:
                check_image = check_batch[IMAGE_KEY][0] # First image in batch
                check_label = check_batch[LABEL_KEY][0] # First label in batch
                print(f"  Sample image shape: {check_image.shape}, dtype: {check_image.dtype}")
                print(f"  Sample label tensor: {check_label}, shape: {check_label.shape}, dtype: {check_label.dtype}") # Should be tensor([0.]) or tensor([1.])
                # Visualize middle slice of the image
                if check_image.shape[1] > 0:
                    slice_idx = check_image.shape[1] // 2
                    img_slice_np = check_image[0, slice_idx, :, :].cpu().numpy()
                    plt.figure("Check Sample Image", (6, 6))
                    plt.title(f"Image Slice {slice_idx}, Label: {check_label.item():.0f}") # Show label value
                    plt.imshow(img_slice_np, cmap="gray"); plt.axis('off')
                    plt.show()
            else: print("Could not retrieve valid sample batch.")
        except Exception as e: print(f"Error checking/visualizing sample: {e}")


    # --- 6.4. Building Model and Setup (** 변경: 분류 모델 사용 **) ---
    print("\n--- 6.4. Building Model, Optimizer, and Scheduler ---")
    model = None
    gc.collect(); torch.cuda.empty_cache()
    try:
        if MODEL_NAME == "Simple3DClassifier":
            print(f"Building {MODEL_NAME}...")
            model = Simple3DClassifier(
                in_channels=1,
                num_classes=NUM_CLASSES,
                channels=CLASSIFIER_CHANNELS,
                strides=CLASSIFIER_STRIDES,
                fc_dims=CLASSIFIER_FC_DIMS,
                dropout=CLASSIFIER_DROPOUT
            )
        else:
            # You could potentially adapt the UNet encoder here, but Simple3DClassifier is clearer for this task
            raise ValueError(f"Unsupported MODEL_NAME for classification: '{MODEL_NAME}'. Use 'Simple3DClassifier'.")

        model = model.to(DEVICE)
        print(f"  Model '{MODEL_NAME}' built and moved to {DEVICE}.")

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', # ** 변경: AUC는 높을수록 좋으므로 'max' 모드
                                         factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE,
                                         verbose=True, min_lr=MIN_LR)
        print(f"  Optimizer: Adam (LR={LEARNING_RATE})")
        print(f"  LR Scheduler: ReduceLROnPlateau (Mode: max, Factor={LR_SCHEDULER_FACTOR}, Patience={LR_SCHEDULER_PATIENCE})")

        early_stopping_counter = 0
        best_val_metric = -1.0 # ** 변경: AUC 기준, 초기값 -1 또는 0
        best_val_epoch = -1

        print("Model, Optimizer, Scheduler initialized.")

        # --- Model Summary (Optional) ---
        if TORCHINFO_AVAILABLE:
            try:
                input_size = (BATCH_SIZE, 1, *TARGET_SPATIAL_SHAPE) # B, C, D, H, W
                print(f"\n--- Model Summary (Input size: {input_size}) ---")
                summary(model, input_size=input_size, device=str(DEVICE),
                        col_names=["input_size", "output_size", "num_params", "mult_adds"], verbose=0)
            except Exception as e:
                print(f"\nError generating model summary: {e}")
                print(model)
        else: print("\n'torchinfo' not installed. Skipping model summary."); print(model)

    except Exception as e: print(f"Error building model or setup: {e}"); traceback.print_exc(); raise


    # --- 6.5. Checkpoint Loading (** 변경: best_metric은 AUC 기준 **) ---
    start_epoch = 0
    if RESUME_CHECKPOINT:
        # ... (Checkpoint loading logic largely the same, but interpret best_metric as AUC) ...
        ckpt_path_to_load = None
        if RESUME_CHECKPOINT is True: ckpt_path_to_load = LATEST_CHECKPOINT_PATH if os.path.exists(LATEST_CHECKPOINT_PATH) else None
        elif isinstance(RESUME_CHECKPOINT, str): ckpt_path_to_load = RESUME_CHECKPOINT if os.path.exists(RESUME_CHECKPOINT) else None

        if ckpt_path_to_load:
            print(f"\n--- Resuming Training from Checkpoint: {ckpt_path_to_load} ---")
            try:
                checkpoint = torch.load(ckpt_path_to_load, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_metric = checkpoint.get('best_metric', -1.0) # Load previous best AUC
                best_val_epoch = checkpoint.get('best_val_epoch', -1)
                if 'scheduler_state_dict' in checkpoint and hasattr(lr_scheduler, 'load_state_dict'):
                    try: lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict']); print("  LR Scheduler state loaded.")
                    except: print("  Warning: Could not load LR scheduler state.")
                print(f"  Resuming from epoch {start_epoch}. Previous best validation AUC: {best_val_metric:.4f} at epoch {best_val_epoch+1}")
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0; best_val_metric = -1.0; best_val_epoch = -1; traceback.print_exc()
        else: print("\n--- Checkpoint not found. Starting from scratch. ---")
    else: print("\n--- Starting Training from Scratch (Epoch 0) ---")


    # --- 6.6. Training Model (** 변경: 분류 작업 루프 **) ---
    print("\n--- 6.6. Starting Model Training ---")
    history_log = []
    log_file = os.path.join(LOG_DIR, f"training_log_classification_{TIMESTAMP}.csv")
    try: # Write header for classification log
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,val_auc,val_accuracy,lr,time_epoch_sec\n")
        print(f"  Logging training progress to: {log_file}")
    except OSError as e: print(f"  Warning: Could not create log file '{log_file}': {e}")


    if model and train_loader:
        total_epochs = MAX_EPOCHS
        print(f"  Starting training from epoch {start_epoch} up to {total_epochs} epochs...")
        # ... (Print device, sample counts, batch size etc.) ...

        for epoch in range(start_epoch, total_epochs):
            epoch_start_time = time.time()
            print("-" * 60); print(f"Epoch {epoch + 1}/{total_epochs}")

            # --- Training Phase ---
            model.train(); epoch_loss = 0; train_step = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", unit="batch", leave=True)
            for batch_data in train_pbar:
                train_step += 1
                inputs = batch_data[IMAGE_KEY].to(DEVICE)
                # ** 변경: 클래스 레이블 로드 및 형태 조정 **
                # Label should be shape [Batch, 1] and float type for BCEWithLogitsLoss
                labels = batch_data[LABEL_KEY].to(DEVICE) # Already shape [B, 1], float from transform
                if labels.shape[-1] != 1: labels = labels.unsqueeze(1) # Ensure shape [B, 1]

                optimizer.zero_grad()
                outputs = model(inputs) # Shape [Batch, 1] (logits)

                try:
                    loss = loss_function(outputs, labels)
                except Exception as loss_err:
                     print(f"\nError during loss calculation step {train_step}: {loss_err}")
                     print(f"  Output shape: {outputs.shape}, dtype: {outputs.dtype}")
                     print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
                     raise loss_err

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{epoch_loss/train_step:.4f}"})

            avg_epoch_train_loss = epoch_loss / train_step
            print(f"Epoch {epoch + 1} Average Training Loss: {avg_epoch_train_loss:.4f}")

            # --- Validation Phase ---
            avg_val_loss = float('nan')
            current_val_auc = float('nan')
            current_val_acc = float('nan')

            if val_loader:
                model.eval()
                val_epoch_loss = 0; val_step = 0
                # ** 변경: AUC 및 정확도 계산 위한 변수 초기화 **
                auc_metric.reset()
                y_pred_trans = [] # Store transformed predictions
                y_true = [] # Store true labels

                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val  ", unit="batch", leave=True)
                with torch.no_grad():
                    for val_batch_data in val_pbar:
                        val_step += 1
                        val_images = val_batch_data[IMAGE_KEY].to(DEVICE)
                        val_labels = val_batch_data[LABEL_KEY].to(DEVICE) # Shape [B, 1], float
                        if val_labels.shape[-1] != 1: val_labels = val_labels.unsqueeze(1)

                        val_outputs = model(val_images) # Logits [Batch, 1]
                        v_loss = loss_function(val_outputs, val_labels)
                        val_epoch_loss += v_loss.item()

                        # ** 변경: AUC 계산 (MONAI metric 사용) **
                        # Need probabilities for AUC, so apply sigmoid
                        val_outputs_prob = torch.sigmoid(val_outputs)
                        auc_metric(val_outputs_prob, val_labels) # Accumulates state

                        # ** 변경: 정확도 계산 위한 값 저장 **
                        # Apply post-processing (sigmoid + threshold) for accuracy calculation later
                        val_outputs_binary = post_pred(val_outputs) # Shape [Batch, 1], values 0 or 1
                        y_pred_trans.extend(val_outputs_binary.cpu().numpy())
                        y_true.extend(val_labels.cpu().numpy())

                        val_pbar.set_postfix({"val_loss": f"{v_loss.item():.4f}"})

                avg_val_loss = val_epoch_loss / val_step
                try: # Aggregate AUC
                    current_val_auc = auc_metric.aggregate()
                    if isinstance(current_val_auc, torch.Tensor): current_val_auc = current_val_auc.item() # Ensure float
                except Exception as e: print(f"Could not aggregate AUC metric: {e}"); current_val_auc = float('nan')

                try: # Calculate Accuracy
                    y_true = np.array(y_true).flatten() # Flatten labels
                    y_pred_trans = np.array(y_pred_trans).flatten() # Flatten predictions
                    current_val_acc = accuracy_score(y_true, y_pred_trans)
                except Exception as e: print(f"Could not calculate Accuracy: {e}"); current_val_acc = float('nan')


                print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.4f}")
                print(f"Epoch {epoch + 1} Validation ROC AUC: {current_val_auc:.4f}")
                print(f"Epoch {epoch + 1} Validation Accuracy: {current_val_acc:.4f}")

                # --- Learning Rate Scheduling (** 변경: AUC 기준 **) ---
                lr_scheduler.step(current_val_auc if not np.isnan(current_val_auc) else -1.0) # Step based on AUC

                # --- Checkpointing & Early Stopping (** 변경: AUC 기준 **) ---
                if not np.isnan(current_val_auc) and current_val_auc > best_val_metric:
                    print(f"  New best validation AUC: {current_val_auc:.4f} (previous: {best_val_metric:.4f}). Saving model...")
                    best_val_metric = current_val_auc
                    best_val_epoch = epoch
                    try:
                         torch.save({
                            'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                            'best_metric': best_val_metric, # Save best AUC
                            'best_val_epoch': best_val_epoch,
                            'config': { # Save key config params for reference
                                'model_name': MODEL_NAME,
                                'num_classes': NUM_CLASSES,
                                'target_spatial_shape': TARGET_SPATIAL_SHAPE,
                                'target_spacing': TARGET_SPACING,
                                'classifier_channels': CLASSIFIER_CHANNELS,
                                'classifier_fc_dims': CLASSIFIER_FC_DIMS,
                            }}, BEST_MODEL_PATH)
                         print(f"  Best model saved to: {BEST_MODEL_PATH}")
                         early_stopping_counter = 0
                    except Exception as save_err: print(f"  Error saving best model: {save_err}")
                else:
                    early_stopping_counter += 1
                    print(f"  Validation AUC did not improve. Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

                if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1} after {EARLY_STOPPING_PATIENCE} epochs without AUC improvement.")
                    break
            else:
                 print("  No validation loader. Skipping validation, LR scheduling, and early stopping.")
                 # Optionally save based on training loss? Less meaningful for classification usually.

            # --- Save Latest Checkpoint ---
            # ... (Save latest checkpoint logic - same as before, stores best_metric which is now AUC) ...
            if (epoch + 1) % SAVE_CHECKPOINT_FREQ == 0 or (epoch + 1) == total_epochs:
                 try:
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                                'best_metric': best_val_metric, 'best_val_epoch': best_val_epoch}, LATEST_CHECKPOINT_PATH)
                    print(f"  Saved latest checkpoint at epoch {epoch + 1}")
                 except Exception as save_err: print(f"  Error saving latest checkpoint: {save_err}")


            # --- Log History ---
            epoch_time_sec = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            log_entry = {
                'epoch': epoch, 'train_loss': avg_epoch_train_loss, 'val_loss': avg_val_loss,
                'val_auc': current_val_auc, 'val_accuracy': current_val_acc, # Log classification metrics
                'lr': current_lr, 'time_epoch_sec': epoch_time_sec
            }
            history_log.append(log_entry)
            try: # Append to log file
                with open(log_file, 'a') as f:
                    f.write(f"{epoch},{avg_epoch_train_loss:.6f},{avg_val_loss:.6f},"
                            f"{current_val_auc:.6f},{current_val_acc:.6f}," # Write new metrics
                            f"{current_lr:.8f},{epoch_time_sec:.2f}\n")
            except IOError as e: print(f"  Warning: Could not write to log file: {e}")

            print(f"Epoch {epoch + 1} finished in {epoch_time_sec:.2f} seconds. Current LR: {current_lr:.8f}")
            gc.collect(); torch.cuda.empty_cache()
            # End of Epoch Loop

        print(f"\n--- Model Training Finished (Stopped at epoch {epoch + 1}) ---")
        if val_loader and best_val_epoch != -1:
            print(f"  Best Validation AUC: {best_val_metric:.4f} was achieved at epoch {best_val_epoch + 1}")
        elif not val_loader: print("  Training finished without validation.")

    else: print("Skipping training: Model not built or training dataloader is empty.")

    # --- 6.7. Visualizing Training History (** 변경: 분류 함수 호출 **) ---
    print("\n--- 6.7. Visualizing Training History ---")
    plot_classification_history(log_file, LOG_DIR)


    # --- 6.8. Evaluating Best Model (** 변경: 분류 평가 **) ---
    print("\n--- 6.8. Evaluating Best Model on Validation Set ---")
    if os.path.exists(BEST_MODEL_PATH):
        print(f"  Loading best model for evaluation from: {BEST_MODEL_PATH}")
        eval_model = None; gc.collect(); torch.cuda.empty_cache()
        try:
            checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            # --- Re-initialize model based on saved config or defaults ---
            saved_config = checkpoint.get('config', {})
            eval_model_name = saved_config.get('model_name', MODEL_NAME)
            eval_num_classes = saved_config.get('num_classes', NUM_CLASSES)
            eval_channels = saved_config.get('classifier_channels', CLASSIFIER_CHANNELS)
            eval_fc_dims = saved_config.get('classifier_fc_dims', CLASSIFIER_FC_DIMS)
            # Note: Strides, dropout not saved in example, using defaults
            print(f"  Re-initializing model '{eval_model_name}' for evaluation.")
            if eval_model_name == "Simple3DClassifier":
                 eval_model = Simple3DClassifier(in_channels=1, num_classes=eval_num_classes, channels=eval_channels, fc_dims=eval_fc_dims)
            else: raise ValueError(f"Unsupported model '{eval_model_name}' in checkpoint.")

            eval_model.load_state_dict(checkpoint['model_state_dict'])
            eval_model.to(DEVICE); eval_model.eval()
            print("  Best model loaded successfully.")

            best_epoch_eval = checkpoint.get('best_val_epoch', 'N/A')
            best_metric_eval = checkpoint.get('best_metric', np.nan) # Best AUC
            if best_epoch_eval != 'N/A': best_epoch_eval += 1
            print(f"  Best model saved at Epoch: {best_epoch_eval}, with Validation AUC: {best_metric_eval:.4f}")

            if val_loader:
                print("  Evaluating best model on the validation set...")
                eval_auc_metric = ROCAUCMetric()
                eval_loss_total = 0.0; eval_steps = 0
                eval_y_pred_trans = []
                eval_y_true = []

                eval_pbar = tqdm(val_loader, desc="Evaluating Best Model", unit="batch", leave=True)
                with torch.no_grad():
                    for eval_batch_data in eval_pbar:
                        eval_steps += 1
                        eval_images = eval_batch_data[IMAGE_KEY].to(DEVICE)
                        eval_labels = eval_batch_data[LABEL_KEY].to(DEVICE) # [B, 1] float
                        if eval_labels.shape[-1] != 1: eval_labels = eval_labels.unsqueeze(1)

                        eval_outputs = eval_model(eval_images) # Logits [B, 1]
                        eval_loss = loss_function(eval_outputs, eval_labels); eval_loss_total += eval_loss.item()

                        eval_outputs_prob = torch.sigmoid(eval_outputs)
                        eval_auc_metric(eval_outputs_prob, eval_labels) # Accumulate AUC state

                        eval_outputs_binary = post_pred(eval_outputs)
                        eval_y_pred_trans.extend(eval_outputs_binary.cpu().numpy())
                        eval_y_true.extend(eval_labels.cpu().numpy())
                        eval_pbar.set_postfix({"eval_loss": f"{eval_loss.item():.4f}"})

                final_avg_loss = eval_loss_total / eval_steps
                final_auc = eval_auc_metric.aggregate().item()
                final_acc = accuracy_score(np.array(eval_y_true).flatten(), np.array(eval_y_pred_trans).flatten())

                print("\n--- Validation Set Performance (Best Model) ---")
                print(f"  Average Loss: {final_avg_loss:.4f}")
                print(f"  ROC AUC: {final_auc:.4f}")
                print(f"  Accuracy: {final_acc:.4f}")

            else: print("  No validation data available to evaluate the best model.")
            del eval_model, checkpoint; gc.collect(); torch.cuda.empty_cache()
        except Exception as e: print(f"Error loading or evaluating best model: {e}"); traceback.print_exc()
    else: print("  Best model checkpoint not found. Skipping evaluation.")


    # --- 6.9. Predicting Sample (** 변경: 분류 결과 출력 **) ---
    print("\n--- 6.9. Predicting on a Validation Sample ---")
    if os.path.exists(BEST_MODEL_PATH) and val_loader:
        print(f"  Loading best model again for prediction from: {BEST_MODEL_PATH}")
        pred_model = None; gc.collect(); torch.cuda.empty_cache()
        try:
            checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            # --- Re-initialize model ---
            saved_config = checkpoint.get('config', {})
            pred_model_name = saved_config.get('model_name', MODEL_NAME)
            pred_num_classes = saved_config.get('num_classes', NUM_CLASSES)
            pred_channels = saved_config.get('classifier_channels', CLASSIFIER_CHANNELS)
            pred_fc_dims = saved_config.get('classifier_fc_dims', CLASSIFIER_FC_DIMS)
            print(f"  Re-initializing model '{pred_model_name}' for prediction.")
            if pred_model_name == "Simple3DClassifier":
                 pred_model = Simple3DClassifier(in_channels=1, num_classes=pred_num_classes, channels=pred_channels, fc_dims=pred_fc_dims)
            else: raise ValueError(f"Unsupported model '{pred_model_name}'")

            pred_model.load_state_dict(checkpoint['model_state_dict'])
            pred_model.to(DEVICE); pred_model.eval()
            print("  Best model reloaded successfully.")

            # --- Get a Sample Batch ---
            val_iterator = iter(val_loader)
            sample_batch = next(val_iterator)
            sample_image = sample_batch[IMAGE_KEY][0] # First image tensor
            true_label = sample_batch[LABEL_KEY][0].item() # Get scalar label
            try:
                image_path = sample_batch[IMAGE_KEY + "_meta_dict"]['filename_or_obj'][0]
                image_filename = os.path.basename(image_path)
            except KeyError:
                image_filename = "Unknown Sample (Metadata Missing)"
                print("  Warning: Could not retrieve image filename from metadata.")

            print(f"\n  Predicting on sample: {image_filename}")
            print(f"    True Label: {'Cancer' if true_label == 1 else 'Normal'} ({true_label:.0f})")
            print(f"    Input Image Shape: {sample_image.shape}")

            sample_image_batch = sample_image.unsqueeze(0).to(DEVICE) # Add batch dim

            # --- Perform Inference ---
            with torch.no_grad():
                pred_logits = pred_model(sample_image_batch) # Output: [1, 1] (logit)
                pred_prob = torch.sigmoid(pred_logits)     # Output: [1, 1] (probability)
                pred_class = post_pred(pred_logits).item() # Output: 0.0 or 1.0

            print(f"    Predicted Logit: {pred_logits.item():.4f}")
            print(f"    Predicted Probability (Cancer): {pred_prob.item():.4f}")
            print(f"    Predicted Label: {'Cancer' if pred_class == 1 else 'Normal'} ({pred_class:.0f})")

            # Optional: Visualize a slice of the image
            slice_to_show = sample_image.shape[1] // 2
            img_slice_np = sample_image[0, slice_to_show, :, :].cpu().numpy()
            plt.figure("Prediction Sample Image", (6, 6))
            plt.title(f"Sample: {image_filename}\nTrue: {'Cancer' if true_label==1 else 'Normal'}, Pred: {'Cancer' if pred_class==1 else 'Normal'} (Prob: {pred_prob.item():.2f})")
            plt.imshow(img_slice_np, cmap="gray"); plt.axis('off')
            plt.show()

            del pred_model, checkpoint, sample_batch, sample_image, sample_image_batch, pred_logits, pred_prob
            gc.collect(); torch.cuda.empty_cache()

        except StopIteration: print("  Validation dataset iterator exhausted.")
        except Exception as e: print(f"An error occurred during prediction: {e}"); traceback.print_exc()
        finally:
            if 'pred_model' in locals() and pred_model is not None: del pred_model
            gc.collect(); torch.cuda.empty_cache()
    else: print("  Skipping prediction: Best model not found or no validation data.")


    print("\n--- Script Finished ---")
    print(f"Output artifacts saved in: {OUTPUT_DIR}")
    print("="*60)

# <<< End of if __name__ == '__main__': block >>>