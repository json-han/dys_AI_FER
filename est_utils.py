# -*- coding: utf-8 -*-
import os
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ExifTags
from tqdm import tqdm
import matplotlib.font_manager as fm # Added for font loading
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for plotting
from sklearn.model_selection import train_test_split # Added for data splitting


# --- Configuration ---
EST_DATA_BASE_DIR = './Data/EST_data' # Renamed from BASE_DIR
IMG_SPLITS = ['img_train', 'img_val', 'img_test']
SEG_SPLITS = ['seg_train', 'seg_val', 'seg_test']
LABEL_FOLDERS = ['label_train', 'label_val', 'label_test'] # Renamed from LABEL_SPLITS
LABEL_TARGETS = ['anger', 'happy', 'panic', 'sadness'] # Renamed from CLASSES

# New constant for soft labeling
BACKBONE_CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
ANNOTATOR_KEYS = ['annot_A', 'annot_B', 'annot_C']
BOX_KEYS = ['minX', 'maxX', 'minY', 'maxY']

SEG_CLASS_MAP = {
    0: 'background', 1: 'hair', 2: 'body', 3: 'face', 4: 'cloth', 5: 'etc'
}

SEG_COLORS = {
    0: (0, 0, 0, 0),
    1: (255, 215, 0, 100),    # hair - gold
    2: (135, 206, 235, 100),  # body - skyblue
    3: (255, 0, 0, 100),      # face - red
    4: (0, 255, 0, 100),      # cloth - green
    5: (128, 0, 128, 100),    # etc - purple
}

# --- Path Finding and Listing ---

def find_image_path(filename):
    for split in IMG_SPLITS:
        for cls in CLASSES:
            path = os.path.join(BASE_DIR, split, cls, filename)
            if os.path.isfile(path):
                return path
    for split in IMG_SPLITS:
        cand = os.path.join(BASE_DIR, split, filename)
        if os.path.isfile(cand):
            return cand
    return None

def find_seg_path(filename, seg_split):
    img_path = find_image_path(filename)
    if not img_path:
        return None
    emotion_class = os.path.basename(os.path.dirname(img_path))
    split_prefix = seg_split.replace('seg_', '')
    seg_dir = os.path.join(BASE_DIR, seg_split)
    cand_path = os.path.join(seg_dir, f'{split_prefix}_{emotion_class}.npz')
    return cand_path if os.path.isfile(cand_path) else None

# --- Data Loading ---

def load_all_label_records(label_split):
    json_files = glob.glob(os.path.join(BASE_DIR, label_split, '*.json'))
    records = []
    for jf in json_files:
        with open(jf, 'r', encoding='euc-kr') as f:
            try:
                data = json.load(f)
                records.extend(data)
            except Exception as e:
                print(f'JSON load error: {jf} -> {e}')
                continue
    return records

def long_format_from_records(records, split_name):
    rows = []
    for item in records:
        common_info = {
            'split': split_name,
            'filename': item.get('filename'),
            'gender': item.get('gender'),
            'age': item.get('age'),
            'isProf': item.get('isProf'),
            'uploader_faceExp': item.get('faceExp_uploader'),
            'uploader_bg': item.get('bg_uploader'),
        }
        for annot_key in ANNOTATOR_KEYS:
            ann = item.get(annot_key, {})
            boxes = ann.get('boxes', {})
            row = {
                **common_info,
                'annotator': annot_key,
                'faceExp': ann.get('faceExp'),
                'bg': ann.get('bg'),
                'minX': boxes.get('minX'),
                'maxX': boxes.get('maxX'),
                'minY': boxes.get('minY'),
                'maxY': boxes.get('maxY'),
            }
            rows.append(row)
    return pd.DataFrame(rows)

def load_seg_mask(npz_path, filename):
    with np.load(npz_path) as npz:
        if filename in npz.files:
            return npz[filename]
        else:
            return None

# --- JSON Parsing Utility ---
def safe_json_loads(x):
    """
    Pandas DataFrame에서 JSON 문자열을 안전하게 로드합니다.
    유효하지 않거나 비어있는 문자열은 NaN으로 처리합니다.
    """
    if pd.isna(x) or not isinstance(x, str) or not x.strip():
        return np.nan
    try:
        return json.loads(x.replace("'", '"'))
    except json.JSONDecodeError:
        return np.nan

# --- Dataset Class ---
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, csv_file, processor, phase, backbone_classes):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['phase'] == phase].reset_index(drop=True)
        self.processor = processor
        self.backbone_classes = backbone_classes

        # JSON 문자열로 저장된 컬럼들을 딕셔너리 객체로 변환
        for col in ['annot_A', 'annot_B', 'annot_C']:
            self.df[col] = self.df[col].apply(safe_json_loads)
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['img_path']
        image = Image.open(image_path)
        
        # Use imported correct_image_orientation
        image = correct_image_orientation(image) # correct_image_orientation is defined later in this file
        image = image.convert("RGB")

        soft_label_vector = torch.tensor(
            [row[f"soft_{cls}"] for cls in self.backbone_classes],
            dtype=torch.float
        )
        
        # 바운딩 박스 좌표 평균 계산 및 이미지 크롭
        minX_coords, minY_coords, maxX_coords, maxY_coords = [], [], [], []

        for annotator in ['annot_A', 'annot_B', 'annot_C']:
            if pd.notna(row[annotator]) and 'boxes' in row[annotator]:
                boxes = row[annotator]['boxes']
                minX_coords.append(boxes['minX'])
                minY_coords.append(boxes['minY'])
                maxX_coords.append(boxes['maxX'])
                maxY_coords.append(boxes['maxY'])
        
        if minX_coords:
            avg_minX = np.mean(minX_coords)
            avg_minY = np.mean(minY_coords)
            avg_maxX = np.mean(maxX_coords)
            avg_maxY = np.mean(maxY_coords)
            cropped_image = image.crop((int(avg_minX), int(avg_minY), int(avg_maxX), int(avg_maxY)))
        else:
            cropped_image = image
        
        inputs = self.processor(images=cropped_image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].squeeze()

        return inputs['pixel_values'], soft_label_vector

# --- Image & Bbox Utilities ---

def correct_image_orientation(img):
    try:
        exif = img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag, None) == 'Orientation':
                    if value == 1:
                        pass
                    elif value == 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 3:
                        img = img.rotate(180, expand=True)
                    elif value == 4:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif value == 5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90, expand=True)
                    elif value == 6:
                        img = img.rotate(-90, expand=True)
                    elif value == 7:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
                    elif value == 8:
                        img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def bbox_size_and_ratio(minX, maxX, minY, maxY, img_w=None, img_h=None):
    try:
        w = max(0, float(maxX) - float(minX))
        h = max(0, float(maxY) - float(minY))
        area = w * h
        aspect = (w / h) if h > 0 else np.nan
        rel_area = (area / (img_w * img_h)) if (img_w and img_h and img_w*img_h>0) else np.nan
        return w, h, area, aspect, rel_area
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def draw_bbox_on_image(img, bbox, color=(255, 0, 0), width=3):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
    return img

def overlay_segmentation(img, seg_mask):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()
    w, h = img.size

    if seg_mask.shape[1] != w or seg_mask.shape[0] != h:
        seg_img = Image.fromarray(seg_mask.astype(np.uint8))
        seg_img = seg_img.resize((w, h), resample=Image.NEAREST)
        seg_mask = np.array(seg_img)

    for y in range(h):
        for x in range(w):
            cls_id = int(seg_mask[y, x])
            color = SEG_COLORS.get(cls_id, (255, 255, 255, 60))
            if cls_id != 0:
                overlay_pixels[x, y] = color
    blended = Image.alpha_composite(img, overlay)
    return blended

def compute_seg_ratios(seg_mask):
    total = seg_mask.size
    ratios = {}
    for k, name in SEG_CLASS_MAP.items():
        ratios[name] = float(np.sum(seg_mask == k)) / total if total > 0 else np.nan
    return ratios

# --- Plotting Utilities for Soft Labels ---

def plot_soft_label_boxplot(df, soft_label_cols, output_path, font_prop):
    """
    소프트 레이블 점수의 분포를 박스 플롯으로 시각화하고 저장합니다.

    Args:
        df (pd.DataFrame): 소프트 레이블 데이터프레임.
        soft_label_cols (list): 소프트 레이블 컬럼 이름 리스트.
        output_path (str): 박스 플롯을 저장할 파일 경로.
        font_prop (FontProperties): 한글 폰트 속성.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[soft_label_cols])
    plt.title('Distribution of Soft Label Scores Across All Categories', fontproperties=font_prop)
    plt.xlabel('Emotion', fontproperties=font_prop)
    plt.ylabel('Soft Label Score', fontproperties=font_prop)
    plt.xticks(rotation=45, ha='right', fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Box plot saved as '{output_path}'")

def plot_soft_label_histograms(df, soft_label_cols, output_path, font_prop):
    """
    각 소프트 레이블 컬럼에 대한 히스토그램을 생성하고 저장합니다.

    Args:
        df (pd.DataFrame): 소프트 레이블 데이터프레임.
        soft_label_cols (list): 소프트 레이블 컬럼 이름 리스트.
        output_path (str): 히스토그램을 저장할 파일 경로.
        font_prop (FontProperties): 한글 폰트 속성.
    """
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    fig.suptitle('Individual Distribution of Soft Label Scores with Frequencies', fontsize=18, fontproperties=font_prop)
    axes = axes.flatten()

    for i, col in enumerate(soft_label_cols):
        ax = axes[i]
        bins = np.linspace(0, 1, num=11)
        counts, _ = np.histogram(df[col], bins=bins)

        ax.hist(df[col], bins=bins, rwidth=0.8, color='skyblue', edgecolor='black')

        for count, rect in zip(counts, ax.patches):
            if count > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    f'{int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=12
                )

        ax.set_title(f'Distribution of {col}', fontsize=16, fontproperties=font_prop)
        ax.set_xlabel('Score', fontproperties=font_prop)
        ax.set_ylabel('Frequency', fontproperties=font_prop)
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{x:.1f}' for x in bins], rotation=45, ha='right', fontproperties=font_prop)
        ax.set_xlim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    print(f"Histograms with counts saved as '{output_path}'")

# New function: create_soft_label
def create_soft_label(annotations,
                      backbone_classes=BACKBONE_CLASSES,
                      label_map=None): # label_map will be passed from notebook
    """
    라벨러가 부여한 감정 레이블을 백본 클래스별 소프트 라벨로 변환하는 함수.

    Args:
        annotations (list): 라벨러가 부여한 감정 레이블(str) 리스트.
        backbone_classes (list): 백본 클래스 이름 리스트 (순서 엄수).
        label_map (dict): 라벨러 감정 → backbone 클래스별 soft dict mapping.

    Returns:
        list: 백본 클래스별 소프트 레이블 (확률 분포).
    """
    if label_map is None:
        raise ValueError("label_map must be provided for create_soft_label.")

    soft_label = np.zeros(len(backbone_classes), dtype=float)
    for ann in annotations:
        if ann in label_map:
            for class_name, weight in label_map[ann].items():
                try:
                    idx = backbone_classes.index(class_name)
                    soft_label[idx] += weight
                except ValueError:
                    print(f"Warning: Class '{class_name}' not found in BACKBONE_CLASSES. Skipping.")
    if len(annotations) > 0:
        # 정규화 (총합이 1이 되도록)
        total_weight = np.sum(soft_label)
        if total_weight > 0:
            soft_label /= total_weight
    return soft_label.tolist()

# --- Data Splitting Utilities ---

def get_stratify_key(row, soft_label_columns, backbone_classes=BACKBONE_CLASSES):
    """
    한 행(row)을 입력받아 soft label의 최대값에 해당하는 감정(들)을 찾아 키를 반환하는 함수.
    여러 감정이 동점일 경우 하이픈으로 연결하여 반환합니다.

    Args:
        row (pd.Series): soft label 컬럼을 포함하는 데이터프레임의 한 행.
        soft_label_columns (list): soft label 컬럼 이름 리스트.
        backbone_classes (list): 백본 클래스 이름 리스트 (순서 엄수).

    Returns:
        str: 계층적 분할을 위한 키 (예: 'happy', 'anger-fear').
    """
    soft_labels = row[soft_label_columns]
    max_value = soft_labels.max()

    # 최대값을 가진 컬럼(감정)들을 찾기
    max_labels = soft_labels[soft_labels == max_value].index.tolist()

    # 컬럼명에서 'soft_' 접두사 제거 및 알파벳순 정렬
    cleaned_labels = sorted([label.replace('soft_', '') for label in max_labels])

    # 하이픈(-)으로 연결하여 최종 키 생성
    return '-'.join(cleaned_labels)

def split_soft_label_data(df, soft_label_columns, test_size=0.2, val_size=0.5, random_state=42, rare_threshold=10):
    """
    Soft Label 분포를 기반으로 데이터를 Train, Validation, Test 세트로 재분할합니다.
    희귀 클래스는 Train 세트에 통합됩니다.

    Args:
        df (pd.DataFrame): 소프트 레이블 데이터프레임.
        soft_label_columns (list): 소프트 레이블 컬럼 이름 리스트.
        test_size (float): 최종 Test 세트의 비율 (전체 데이터 대비).
        val_size (float): Test 세트 분리 후 Validation 세트의 비율 (남은 데이터 대비).
        random_state (int): 재현성을 위한 랜덤 시드.
        rare_threshold (int): 계층적 분할 시 희귀 클래스로 간주할 최소 샘플 수.

    Returns:
        tuple: (train_df, val_df, test_df) 분할된 데이터프레임.
    """
    # 'stratify_key' 컬럼 생성
    df['stratify_key'] = df.apply(lambda row: get_stratify_key(row, soft_label_columns), axis=1)

    # Stratify Key 분포 확인
    key_distribution = df['stratify_key'].value_counts()

    # 희귀 클래스 식별 및 분리
    rare_keys_for_split_issue = key_distribution[key_distribution < rare_threshold].index.tolist()
    df_rare = df[df['stratify_key'].isin(rare_keys_for_split_issue)]
    df_main = df[~df['stratify_key'].isin(rare_keys_for_split_issue)]

    # 메인 데이터셋을 먼저 분리 (Train vs Temp)
    X_main = df_main.drop('stratify_key', axis=1)
    y_main = df_main['stratify_key']

    train_df_main, temp_df_main = train_test_split(
        df_main,
        test_size=test_size, # 전체 데이터 대비 test_size
        random_state=random_state,
        stratify=y_main
    )

    # 희귀 데이터셋을 train_df에 합침
    train_df = pd.concat([train_df_main, df_rare])

    # temp 세트를 val / test 세트로 분리
    X_temp = temp_df_main.drop('stratify_key', axis=1)
    y_temp = temp_df_main['stratify_key']

    val_df, test_df = train_test_split(
        temp_df_main,
        test_size=val_size, # 남은 데이터 대비 val_size (0.5면 50%)
        random_state=random_state,
        stratify=y_temp
    )

    # 원본 DataFrame에 새로운 phase 할당 (선택 사항, 필요에 따라)
    # df.loc[train_df.index, 'new_phase'] = 'train'
    # df.loc[val_df.index, 'new_phase'] = 'val'
    # df.loc[test_df.index, 'new_phase'] = 'test'
    # df['phase'] = df['new_phase']
    # df.drop(columns=['new_phase'], inplace=True)

    return train_df, val_df, test_df


# --- Master DataFrame Builder ---

def build_master_dataframe():
    """
    모든 라벨과 이미지 메타데이터를 결합하여 마스터 데이터프레임을 생성합니다.
    이 함수는 초기 실행 시 시간이 다소 걸릴 수 있습니다.
    """
    print("Building master dataframe... (This may take a while)")
    
    all_dfs = []
    for lbl_split in LABEL_SPLITS:
        recs = load_all_label_records(lbl_split)
        df_long = long_format_from_records(recs, lbl_split)
        all_dfs.append(df_long)
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.dropna(subset=['filename'], inplace=True)

    image_meta = {}
    unique_files = master_df['filename'].unique()
    
    print(f"Processing {len(unique_files)} unique images for metadata...")
    for fname in tqdm(unique_files):
        img_path = find_image_path(fname)
        if img_path:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                image_meta[fname] = {'img_w': w, 'img_h': h}
            except Exception as e:
                pass
    
    meta_df = pd.DataFrame.from_dict(image_meta, orient='index').reset_index().rename(columns={'index': 'filename'})
    master_df = pd.merge(master_df, meta_df, on='filename', how='left')
    
    print("Calculating bounding box statistics...")
    
    def calculate_bbox_stats(row):
        return bbox_size_and_ratio(
            row['minX'], row['maxX'], row['minY'], row['maxY'],
            row['img_w'], row['img_h']
        )

    bbox_stats_series = master_df.apply(calculate_bbox_stats, axis=1, result_type='expand')
    bbox_stats_series.columns = ['bbox_w', 'bbox_h', 'bbox_area', 'bbox_aspect', 'bbox_rel_area']
    master_df = pd.concat([master_df, bbox_stats_series], axis=1)
    
    print("Master dataframe built successfully!")
    return master_df

# --- General Validation Set Creation Utility ---

def create_general_validation_set(dataset_paths, backbone_classes, output_filename):
    """
    AffectNet 및 FER2013 test 폴더의 이미지들을 사용하여 일반 검증 데이터셋을 생성합니다.
    Hard Label을 One-hot Encoding 방식의 Soft Label로 변환하여 저장합니다.

    Args:
        dataset_paths (dict): 각 데이터셋의 test 폴더 경로를 담은 딕셔너리 (예: {'AffectNet': './Data/AffectNet/Test'}).
        backbone_classes (list): 백본 모델이 사용하는 클래스 리스트.
        output_filename (str): 생성될 CSV 파일의 이름.

    Returns:
        pd.DataFrame: 생성된 일반 검증 데이터셋 DataFrame.
    """
    image_data_list = []

    for dataset_name, path in dataset_paths.items():
        print(f"Processing {dataset_name}...")

        emotion_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        
        for emotion in tqdm(emotion_folders, desc=f'{dataset_name} Emotions'):
            cleaned_emotion = emotion.lower()
            
            # 'angry' -> 'anger', 'sad' -> 'sad' 등 매핑 (FER2013의 경우)
            # AffectNet의 'Contempt'는 backbone_classes에 있으므로 그대로 사용
            if cleaned_emotion == 'angry':
                cleaned_emotion = 'anger'
            elif cleaned_emotion == 'sad':
                cleaned_emotion = 'sad'
            elif cleaned_emotion == 'happy':
                cleaned_emotion = 'happy'
            elif cleaned_emotion == 'neutral':
                cleaned_emotion = 'neutral'
            elif cleaned_emotion == 'surprise':
                cleaned_emotion = 'surprise'
            elif cleaned_emotion == 'fear':
                cleaned_emotion = 'fear'
            elif cleaned_emotion == 'disgust':
                cleaned_emotion = 'disgust'
            elif cleaned_emotion == 'contempt':
                cleaned_emotion = 'contempt'
            
            if cleaned_emotion not in backbone_classes:
                print(f"[Warning] Skipping emotion '{emotion}' from {dataset_name} as it is not in backbone_classes.")
                continue
                
            image_files = glob.glob(os.path.join(path, emotion, '*.jpg')) + \
                          glob.glob(os.path.join(path, emotion, '*.png')) + \
                          glob.glob(os.path.join(path, emotion, '*.jpeg'))

            for img_path in image_files:
                soft_label_vector = {f'soft_{cls}': 0.0 for cls in backbone_classes}
                soft_label_vector[f'soft_{cleaned_emotion}'] = 1.0
                
                row = {
                    'phase': 'val',
                    'category': cleaned_emotion,
                    'filename': os.path.basename(img_path),
                    'img_path': img_path.replace('\\', '/'), # 경로 구분자 통일
                    'exists': True,
                    'annot_A': np.nan,
                    'annot_B': np.nan,
                    'annot_C': np.nan,
                }
                
                row.update(soft_label_vector)
                image_data_list.append(row)

    df_general_val = pd.DataFrame(image_data_list)
    
    # 컬럼 순서 정의 (기존 파일과 동일하게)
    final_columns = [
        'phase', 'category', 'filename', 'img_path', 'exists',
        'annot_A', 'annot_B', 'annot_C'
    ] + [f'soft_{cls}' for cls in backbone_classes]
    
    df_general_val = df_general_val[final_columns]

    print(f"총 {len(image_data_list)}개의 이미지 파일을 수집했습니다.")
    print(f"생성된 DataFrame 형태: {df_general_val.shape}")
    print("Category 분포:")
    print(df_general_val['category'].value_counts())
    
    df_general_val.to_csv(output_filename, index=False)
    print(f"{output_filename} 파일이 성공적으로 생성되었습니다.")
    
    return df_general_val

# New: Font setting for Korean
try:
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf" # 실제 파일경로
    FONT_PROP = fm.FontProperties(fname=FONT_PATH)
except FileNotFoundError:
    print(f"Warning: Font file not found at {FONT_PATH}. Using default font.")
    FONT_PROP = fm.FontProperties() # 기본 폰트 사용
except Exception as e:
    print(f"Error loading font: {e}. Using default font.")
    FONT_PROP = fm.FontProperties()
