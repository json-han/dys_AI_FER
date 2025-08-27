# -*- coding: utf-8 -*-
import os
import json
import glob
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ExifTags
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = 'workspace/AI/FER/Data/EST_data'
IMG_SPLITS = ['img_train', 'img_val', 'img_test']
SEG_SPLITS = ['seg_train', 'seg_val', 'seg_test']
LABEL_SPLITS = ['label_train', 'label_val', 'label_test']
CLASSES = ['anger', 'happy', 'panic', 'sadness']
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
