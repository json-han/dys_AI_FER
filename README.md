# 얼굴 감정 인식 (Facial Emotion Recognition) 프로젝트

이 프로젝트는 얼굴 표정 이미지를 분석하여 감정을 인식하는 모델을 개발하고 평가합니다. 데이터 전처리, 모델 학습, 추론, 결과 분석 등 다양한 단계의 코드를 포함하고 있습니다.

## 디렉토리 구조

```
/workspace/AI/FER/
├─── ... (Jupyter 노트북 및 Python 스크립트)
├───est_utils.py
├───README.md
├───__pycache__/
├───configs/
├───Data/
└───output/
```

*   **configs/**: `soft_label_map` 등 설정 파일을 저장하는 디렉토리입니다.
*   **Data/**: `AffectNet`, `FER2013` 등 학습 및 평가에 사용되는 데이터셋을 저장하는 디렉토리입니다.
*   **output/**: 모델 학습 결과물, 로그 등 생성된 파일을 저장하는 디렉토리입니다.

## 파일 설명

*   `01_EST_EDA_refactored.ipynb`: EST 데이터셋에 대한 탐색적 데이터 분석(EDA)을 수행합니다.
*   `02_predict_emotions_with_HardlyHumans.ipynb`: "HardlyHumans" 모델을 사용하여 감정을 예측합니다.
*   `03_analyze_prediction_results.ipynb`: 예측 결과를 분석합니다.
*   `04_soft_labeling_processor.ipynb`: 데이터에 소프트 라벨을 생성하는 프로세서입니다.
*   `05_soft_label_data_validator.ipynb`: 소프트 라벨링된 데이터의 유효성을 검사합니다.
*   `05_soft_label_eda_analyzer.ipynb`: 소프트 라벨링된 데이터에 대한 EDA 및 분석을 수행합니다.
*   `06_soft_label_data_splitter.ipynb`: 소프트 라벨링된 데이터를 학습/검증/테스트용으로 분할합니다.
*   `07_general_validation_set_creator.ipynb`: 일반적인 검증 데이터셋을 생성합니다.
*   `08_finetuning_refactored.py`: 모델을 미세 조정(fine-tuning)하는 스크립트입니다.
*   `09_model_analysis.ipynb`: 학습된 모델의 성능을 분석합니다.
*   `10_image_inference.ipynb`: 이미지를 입력하여 감정 추론을 수행합니다.
*   `est_utils.py`: 프로젝트 전반에서 사용되는 유틸리티 함수들을 포함합니다.
