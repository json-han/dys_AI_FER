import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import mlflow
from transformers import ViTImageProcessor, AutoModelForImageClassification
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Import from est_utils
from est_utils import correct_image_orientation, BACKBONE_CLASSES, FONT_PROP

# ==============================================================================
# 0. Configuration
# ==============================================================================
config = {
    "model_name": "HardlyHumans/Facial-expression-detection",
    "learning_rate": 5e-6,  # Lowered learning rate
    "epochs": 5,
    "batch_size": 16,
    "train_csv_path": '06_softlabel_dataset_resplit.csv',
    "general_val_csv_path": '07_general_val_set.csv',
    "mlflow_tracking_uri": "http://0.0.0.0:5000",
    "mlflow_experiment_name": "FER Fine-tuning with Soft Labels",
    "mlflow_run_name": "fine_tuning_run_2",
    "patience": 3,  # Early Stopping patience
    "min_delta": 0.001, # Minimum change to qualify as an improvement
}

# ==============================================================================
# 1. EmotionDataset Class
# ==============================================================================
class EmotionDataset(Dataset):
    def __init__(self, csv_file, processor, phase, backbone_classes):
        self.df = pd.read_csv(csv_file)
        
        # 'phase' 컬럼을 기준으로 데이터 분리
        self.df = self.df[self.df['phase'] == phase].reset_index(drop=True)
        self.processor = processor
        self.backbone_classes = backbone_classes

        # JSON 문자열로 저장된 컬럼들을 딕셔너리 객체로 변환
        for col in ['annot_A', 'annot_B', 'annot_C']:
            def safe_json_loads(x):
                if pd.isna(x) or not isinstance(x, str) or not x.strip():
                    return np.nan
                try:
                    return json.loads(x.replace("'", '"'))
                except json.JSONDecodeError:
                    return np.nan
            self.df[col] = self.df[col].apply(safe_json_loads)
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['img_path']
        image = Image.open(image_path)
        
        # Use imported correct_image_orientation
        image = correct_image_orientation(image)
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

# ==============================================================================
# 2. Helper Functions for Training and Evaluation
# ==============================================================================
def train_epoch(model, data_loader, optimizer, loss_fn, softmax_layer, device):
    model.train()
    total_train_loss = 0
    for images, soft_labels in tqdm(data_loader, desc="Training"):
        images = images.to(device)
        soft_labels = soft_labels.to(device)
        
        outputs = model(pixel_values=images).logits
        loss = loss_fn(softmax_layer(outputs), soft_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(data_loader)

def evaluate_model_performance(model, data_loader, loss_fn, softmax_layer, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    all_true_labels_hard = []
    all_predicted_labels_hard = []

    with torch.no_grad():
        for images, soft_labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            soft_labels = soft_labels.to(device)
            
            outputs = model(pixel_values=images).logits
            loss = loss_fn(softmax_layer(outputs), soft_labels)
            total_loss += loss.item()

            # Calculate accuracy for hard labels (argmax)
            predicted_labels = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(soft_labels, dim=1) # Assuming one-hot for general_val_set
            
            correct_predictions += (predicted_labels == true_labels).sum().item()
            total_samples += soft_labels.size(0)

            all_true_labels_hard.extend(true_labels.cpu().numpy())
            all_predicted_labels_hard.extend(predicted_labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy, all_true_labels_hard, all_predicted_labels_hard

# ==============================================================================
# 3. Main Fine-tuning Orchestration Function
# ==============================================================================
def run_fine_tuning(config):
    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["mlflow_experiment_name"])

    # Model and Processor Setup
    processor = ViTImageProcessor.from_pretrained(config["model_name"], use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(config["model_name"])
    
    # Ensure backbone_classes from est_utils is used for consistency
    # labels = model.config.id2label # Original line, now using BACKBONE_CLASSES
    # backbone_classes = [labels[i] for i in range(len(labels))] # Original line
    num_classes = len(BACKBONE_CLASSES) # Use imported BACKBONE_CLASSES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoaders Preparation
    train_dataset = EmotionDataset(config["train_csv_path"], processor, 'train', BACKBONE_CLASSES)
    # Note: Original script used 'val' phase for val_dataset, but 06_softlabel_dataset_resplit.csv
    # contains 'train', 'val', 'test' phases. Assuming 'val' is correct for validation.
    val_dataset = EmotionDataset(config["train_csv_path"], processor, 'val', BACKBONE_CLASSES)
    general_val_dataset = EmotionDataset(config["general_val_csv_path"], processor, 'val', BACKBONE_CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    general_val_loader = DataLoader(general_val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.KLDivLoss(reduction="batchmean") 
    softmax_layer = nn.LogSoftmax(dim=1)

    # MLflow Experiment Start and Training Loop
    with mlflow.start_run(run_name=config["mlflow_run_name"]):
        mlflow.log_params(config) # Log all config parameters

        # Pre-training General Validation Set Check
        avg_initial_general_val_loss, initial_general_val_accuracy, _, _ = evaluate_model_performance(
            model, general_val_loader, loss_fn, softmax_layer, device
        )
        mlflow.log_metric("initial_general_val_loss", avg_initial_general_val_loss)
        mlflow.log_metric("initial_general_val_accuracy", initial_general_val_accuracy)
        print(f"Initial General Val Loss: {avg_initial_general_val_loss:.4f} | Initial General Val Accuracy: {initial_general_val_accuracy:.4f}")

        # Early Stopping Initialization
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(config["epochs"]):
            # Training
            avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn, softmax_layer, device)

            # Validation
            avg_val_loss, _, _, _ = evaluate_model_performance(model, val_loader, loss_fn, softmax_layer, device)
            avg_general_val_loss, general_val_accuracy, _, _ = evaluate_model_performance(model, general_val_loader, loss_fn, softmax_layer, device)

            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | General Val Loss: {avg_general_val_loss:.4f} | General Val Accuracy: {general_val_accuracy:.4f}")

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("general_val_loss", avg_general_val_loss, step=epoch)
            mlflow.log_metric("general_val_accuracy", general_val_accuracy, step=epoch)

            # Early Stopping Check
            if avg_val_loss < best_val_loss - config['min_delta']:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # Optional: Save best model state here
                # torch.save(model.state_dict(), "best_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config['patience']:
                    print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss for {config['patience']} epochs.")
                    break
        
        mlflow.pytorch.log_model(model, "emotion_fine_tuned_model")
        print("Fine-tuning completed and model saved to MLflow.")

        # ==============================================================================
        # 4. Test Set Evaluation and Logging
        # ==============================================================================
        print("\n==== Starting Test Set Evaluation ====")
        test_dataset = EmotionDataset(config["train_csv_path"], processor, 'test', BACKBONE_CLASSES)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        avg_test_loss, test_accuracy, y_true_hard, y_pred_hard = evaluate_model_performance(
            model, test_loader, loss_fn, softmax_layer, device
        )

        # Calculate additional metrics
        f1_macro = f1_score(y_true_hard, y_pred_hard, average='macro', zero_division=0)
        precision_macro = precision_score(y_true_hard, y_pred_hard, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_hard, y_pred_hard, average='macro', zero_division=0)

        print(f"Test KL-Divergence Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Macro: {f1_macro:.4f}")
        print(f"Test Precision-Macro: {precision_macro:.4f}")
        print(f"Test Recall-Macro: {recall_macro:.4f}")

        # Log test metrics to MLflow
        mlflow.log_metric("test_kl_loss", avg_test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_precision_macro", precision_macro)
        mlflow.log_metric("test_recall_macro", recall_macro)

        # Generate and log Confusion Matrix
        cm = confusion_matrix(y_true_hard, y_pred_hard, labels=range(len(BACKBONE_CLASSES)))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Handle NaN values for annotation
        text_annotations = np.empty_like(cm_normalized, dtype=object)
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                if np.isnan(cm_normalized[i, j]):
                    text_annotations[i, j] = 'X'
                else:
                    text_annotations[i, j] = f"{cm_normalized[i, j]:.2f}"

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=text_annotations, fmt="", cmap='Blues',
                    xticklabels=BACKBONE_CLASSES, yticklabels=BACKBONE_CLASSES)
        plt.title('Normalized Confusion Matrix (Test Set)', fontproperties=FONT_PROP)
        plt.ylabel('True Label', fontproperties=FONT_PROP)
        plt.xlabel('Predicted Label', fontproperties=FONT_PROP)
        plt.tight_layout()
        
        cm_output_path = "08_test_confusion_matrix.png"
        plt.savefig(cm_output_path)
        mlflow.log_artifact(cm_output_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_output_path} and logged to MLflow.")

# ==============================================================================
# 4. Main Execution
# ==============================================================================
if __name__ == "__main__":
    run_fine_tuning(config)
