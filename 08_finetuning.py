import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ExifTags
import numpy as np
import json
import mlflow
from transformers import ViTImageProcessor, AutoModelForImageClassification
from collections import Counter
from tqdm.auto import tqdm 

# ==============================================================================
# 0. 필수 변수 및 함수 정의
# ==============================================================================

# MLflow 트래킹 서버 URI 설정, 실험 이름 설정
mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("FER Fine-tuning with Soft Labels")

# 백본 모델 클래스 순서
backbone_classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# -----------------------------------------------------------------------------
# EmotionDataset 클래스 (바운딩 박스 평균 크롭 로직 포함)
# -----------------------------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, csv_file, processor, phase, backbone_classes):
        self.df = pd.read_csv(csv_file)
        
        # 'phase' 컬럼을 기준으로 데이터 분리
        self.df = self.df[self.df['phase'] == phase].reset_index(drop=True)
        self.processor = processor
        self.backbone_classes = backbone_classes
        
        # JSON 문자열로 저장된 컬럼들을 딕셔너리 객체로 변환 (더욱 견고한 방식으로 수정)
        for col in ['annot_A', 'annot_B', 'annot_C']:
            
            def safe_json_loads(x):
                # 값이 문자열이 아니거나, 비어있거나, 공백만 있으면 NaN 처리
                if pd.isna(x) or not isinstance(x, str) or not x.strip():
                    return np.nan
                try:
                    # 따옴표를 변환하여 JSON으로 로드
                    return json.loads(x.replace("'", '"'))
                except json.JSONDecodeError:
                    # 변환 실패 시에도 NaN 처리
                    return np.nan
            
            self.df[col] = self.df[col].apply(safe_json_loads)
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['img_path']
        image = Image.open(image_path)
        image = self.correct_image_orientation(image)
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
    
    # orientation 보정 함수
    def correct_image_orientation(self, img):
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


# -----------------------------------------------------------------------------
# 메인 학습 스크립트
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ===== 모델 및 하이퍼파라미터 설정 =====
    model_name = "HardlyHumans/Facial-expression-detection"
    processor = ViTImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    labels = model.config.id2label
    backbone_classes = [labels[i] for i in range(len(labels))]
    num_classes = len(backbone_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ===== 데이터로더 준비 (이미 분리된 CSV 파일 사용) =====
    # '06_softlabel_dataset_resplit.csv' 파일명을 사용합니다.
    final_csv_path = '06_softlabel_dataset_resplit.csv'

    train_dataset = EmotionDataset(final_csv_path, processor, 'train', backbone_classes)
    val_dataset = EmotionDataset(final_csv_path, processor, 'val', backbone_classes)
    
    # 재앙적 망각 모니터링을 위한 일반 데이터셋 (별도 파일로 준비)
    # '07_general_val_set.csv' 파일에 FER2013/AffectNet 데이터가 있음
    general_val_dataset = EmotionDataset('07_general_val_set.csv', processor, 'val', backbone_classes)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    general_val_loader = DataLoader(general_val_dataset, batch_size=16, shuffle=False)

    learning_rate = 1e-5
    epochs = 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.KLDivLoss(reduction="batchmean") 
    softmax_layer = nn.LogSoftmax(dim=1)

    # ===== MLflow 실험 시작 및 학습 루프 =====
    with mlflow.start_run(run_name="fine_tuning_run_1"):
        # 파라미터 로깅
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("loss_function", "KLDivLoss")

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for images, soft_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                images = images.to(device)
                soft_labels = soft_labels.to(device)
                
                outputs = model(pixel_values=images).logits
                loss = loss_fn(softmax_layer(outputs), soft_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_val_loss = 0
            total_general_val_loss = 0
            
            with torch.no_grad():
                for images, soft_labels in val_loader:
                    images = images.to(device)
                    soft_labels = soft_labels.to(device)
                    outputs = model(pixel_values=images).logits
                    loss = loss_fn(softmax_layer(outputs), soft_labels)
                    total_val_loss += loss.item()
                
                for images, soft_labels in general_val_loader:
                    images = images.to(device)
                    soft_labels = soft_labels.to(device)
                    outputs = model(pixel_values=images).logits
                    loss = loss_fn(softmax_layer(outputs), soft_labels)
                    total_general_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            avg_general_val_loss = total_general_val_loss / len(general_val_loader)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | General Val Loss: {avg_general_val_loss:.4f}")

            # 메트릭 로깅
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("general_val_loss", avg_general_val_loss, step=epoch)
        
        # 모델 아티팩트 저장
        mlflow.pytorch.log_model(model, name="emotion_fine_tuned_model")
        print("Fine-tuning 완료 및 모델 저장")