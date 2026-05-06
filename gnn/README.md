# GNN

AML(자금세탁 탐지)을 위한 GNN 모델 실험 디렉토리.

## 디렉토리 구조

```
gnn/
└── baseline/               # Multi-PNA+EU baseline 모델 (IBM/Multi-GNN 기반)
    ├── main.py             # 실험 진입점 — 데이터 로드 후 학습 또는 추론 실행
    ├── models.py           # GNN 모델 구현 — GINe, GATe, PNA, RGCN
    ├── training.py         # 학습 루프 — homo/hetero 그래프 각각 train 함수 포함
    ├── train_util.py       # 학습 유틸 — EgoID transform, DataLoader, 평가 함수
    ├── data_loading.py     # 데이터 로딩 — CSV 읽기, 시간 기반 train/val/test 분할
    ├── data_util.py        # 그래프 데이터 객체 — GraphData, HeteroGraphData, port/time-delta 계산
    ├── inference.py        # 저장된 모델 불러와 추론 실행
    ├── util.py             # 공통 유틸 — argparser, logger, set_seed
    ├── format_kaggle_files.py  # 전처리 스크립트 — Kaggle raw CSV → formatted_transactions.csv
    ├── data_config.json    # 데이터/모델 경로 설정 (실행 전 경로 수정 필요)
    ├── model_settings.json # 모델별 하이퍼파라미터 (lr, hidden dim, dropout 등)
    └── env.yml             # baseline 전용 conda 환경 파일 (PyTorch, PyG 포함)
```

## 환경 설치

프로젝트 공통 `requirements.txt`에는 PyTorch/PyG가 없으므로 baseline 실행 시 아래 순서로 별도 설치 필요.

```bash
# 1. PyTorch (CUDA 11.8)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# 2. PyG extensions
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 3. 기타 의존성
pip install datatable wandb munch tqdm scikit-learn pandas
```

## 실행 방법

### 1. 데이터 전처리
```bash
python format_kaggle_files.py /path/to/HI-Small_Trans.csv
```

### 2. data_config.json 경로 설정
```json
{
  "paths": {
    "aml_data": "/path/to/aml_data"
  }
}
```

### 3. 학습 실행 (Multi-PNA+EU)
```bash
python main.py --data Small_HI --model pna --emlps --reverse_mp --ego --ports --tqdm --testing
```
