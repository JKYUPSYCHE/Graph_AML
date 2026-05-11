# GF Model

Multi-PNA+EU baseline에 계좌 수준 노드 피처(Graph Feature)를 추가한 GNN 모델.

## baseline과의 차이점

| 항목 | baseline | gf_model |
|------|----------|----------|
| 노드 피처 | placeholder (1) | 계좌 수준 6개 피처 (train-only) |
| 노드 피처 파일 | 없음 | `account_node_features.csv` |
| `--node_features` 플래그 | 없음 | 있음 |

### 노드 피처 목록 (`--node_features` 사용 시)

| 피처 | 설명 |
|------|------|
| `train_only_pagerank` | train 엣지 기반 PageRank(log1p) |
| `train_only_in_degree` | train 기준 수신 거래 수 |
| `train_only_out_degree` | train 기준 송신 거래 수 |
| `train_only_in_amount_sum` | train 기준 수신 총액(log1p)|
| `train_only_out_amount_sum` | train 기준 송신 총액(log1p) |
| `is_new_in_val_test` | val/test에만 등장한 신규 계좌 여부 |

val/test에만 등장한 계좌(leakage 방지)는 통계값이 0으로 채워진다.

## 디렉토리 구조

```
gnn/
└── gf_model/               # GF 모델 — Multi-PNA+EU + 계좌 수준 노드 피처
    ├── main.py             # 실험 진입점 — 데이터 로드 후 학습 또는 추론 실행
    ├── models.py           # GNN 모델 구현 — GINe, GATe, PNA, RGCN
    ├── training.py         # 학습 루프 — homo/hetero 그래프 각각 train 함수 포함
    ├── train_util.py       # 학습 유틸 — EgoID transform, DataLoader, 평가 함수
    ├── data_loading.py     # 데이터 로딩 — CSV 읽기, 노드 피처 로딩, 시간 기반 분할
    ├── data_util.py        # 그래프 데이터 객체 — GraphData, HeteroGraphData, port/time-delta 계산
    ├── inference.py        # 저장된 모델 불러와 추론 실행
    ├── util.py             # 공통 유틸 — argparser, logger, set_seed
    ├── format_kaggle_files.py  # 전처리 스크립트 — Kaggle raw CSV → formatted_transactions.csv
    ├── data_config.json    # 데이터/모델 경로 설정 (실행 전 경로 수정 필요)
    ├── model_settings.json # 모델별 하이퍼파라미터 (lr, hidden dim, dropout 등)
```

## 실행 방법

### 1. 데이터 전처리

```bash
# 엣지 파일 생성
python format_kaggle_files.py /path/to/HI-Small_Trans.csv

# 노드 피처 파일 생성 (--node_features 사용 시 필요)
python /path/to/make_account_node_features.py formatted_transactions.csv
```

`make_account_node_features.py` 실행 결과로 같은 폴더에 `account_node_features.csv`가 생성된다.

### 2. data_config.json 경로 설정

`gnn/gf_model/data_config.json`의 경로가 실제 환경에 맞는지 확인한다. 경로는 `gnn/gf_model/` 기준 상대 경로다.

```json
{
  "paths": {
    "aml_data": "../../data",
    "node_features": "../../data",
    "model_to_load": "../../experiments/gf_model/models",
    "model_to_save": "../../experiments/gf_model/models"
  }
}
```

`--data` 인자로 넘긴 값이 `aml_data` 및 `node_features` 경로 하위 폴더명이 된다.
예: `--data Small_LI` → `../../data/Small_LI/formatted_transactions.csv`, `../../data/Small_LI/account_node_features.csv`

### 3. TensorBoard 설치 (에폭별 지표 시각화)

학습 중 Train/Val/Test의 F1·Recall·Precision·AUPRC를 실시간으로 그래프로 확인하려면 아래 패키지가 필요하다.

```bash
pip install tensorboard "setuptools<70"
```

> `setuptools<70` 이 필요한 이유: tensorboard가 내부적으로 `pkg_resources`를 사용하는데, setuptools 70 이상에서는 해당 모듈이 제거되어 실행 오류가 발생한다.

학습 실행 후 (또는 실행 중) 별도 터미널에서 아래 명령어로 TensorBoard를 실행한다.

```bash
# gnn/gf_model/ 디렉토리에서 실행
tensorboard --logdir runs
```

브라우저에서 `http://localhost:6006` 으로 접속하면 에폭별 지표 그래프를 확인할 수 있다.
로그는 `runs/{데이터}_{모델}_{날짜시간}/` 폴더에 저장되며, 실험을 여러 번 돌렸을 때 TensorBoard에서 실험별로 비교할 수 있다.

### 4. 학습 실행

`gnn/gf_model/` 디렉토리에서 실행한다.

```bash
# GF 모델 (노드 피처 포함)
python main.py --data Small_LI --model pna --emlps --reverse_mp --ego --ports --tqdm --node_features

# 노드 피처 없이 실행 (placeholder 사용)
python main.py --data Small_LI --model pna --emlps --reverse_mp --ego --ports --tqdm

# 모델 저장 포함
python main.py --data Small_LI --model pna --emlps --reverse_mp --ego --ports --tqdm --node_features --save_model --unique_name gf_run1

# early stopping 적용
python main.py --data Small_LI --model pna --emlps --reverse_mp --ego --ports --tqdm --node_features --patience 10

# 저장된 모델로 추론만 실행
python main.py --data Small_LI --model pna --emlps --reverse_mp --ego --ports --node_features --inference --unique_name gf_run1
```

### 5. CLI 플래그 전체 목록

**필수 인자**

| 플래그 | 설명 |
|--------|------|
| `--data` | 데이터셋 폴더명. `aml_data` 경로 하위 폴더명과 일치해야 함 (예: `Small_LI`) |
| `--model` | 모델 아키텍처 선택: `gin`, `gat`, `pna`, `rgcn`, `mlp` |

**노드 피처 옵션**

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--node_features` | False | `account_node_features.csv`의 계좌 수준 노드 피처 6개 사용. 미지정 시 placeholder(1) 사용 |

**그래프 구성 옵션**

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--emlps` | False | Edge MLP(EMLP) 사용. 엣지 피처를 메시지 패싱에 통합 |
| `--reverse_mp` | False | Reverse Message Passing 사용. 동질 그래프 대신 `node→node` / `node→rev_to→node` 이질 그래프로 변환 |
| `--ports` | False | Port Numbering 엣지 피처 추가. 시간순 기준 이웃 순서를 피처로 인코딩 |
| `--tds` | False | Time Delta 엣지 피처 추가. 연속 거래 간 시간 차이를 피처로 인코딩 |
| `--ego` | False | Ego ID 사용. 배치 내 seed 엣지의 양 끝 노드에 1을 표시하는 피처 추가 |

**학습 하이퍼파라미터**

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--n_epochs` | 100 | 최대 학습 에폭 수 |
| `--batch_size` | 8192 | 미니배치 크기 (엣지 수 기준) |
| `--num_neighs` | [100, 100] | 홉별 샘플링 이웃 수. 2-hop이면 `--num_neighs 100 100` |
| `--patience` | None | Early stopping patience. 지정한 에폭 수만큼 val F1 미개선 시 조기 종료. 미지정 시 비활성화 |
| `--seed` | 42 | 랜덤 시드 |

**모델 저장 / 로드**

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--save_model` | False | Best val F1 epoch의 모델 파라미터 저장 (`model_to_save` 경로) |
| `--unique_name` | None | 저장/로드할 모델 파일명 식별자. `--save_model` 또는 `--finetune`, `--inference`와 함께 사용 |
| `--finetune` | False | 저장된 모델을 불러와 이어서 학습. `--unique_name`으로 로드할 모델 지정 필요 |
| `--inference` | False | 학습 없이 저장된 모델로 추론만 실행. `--unique_name`으로 로드할 모델 지정 필요 |

**기타**

| 플래그 | 기본값 | 설명 |
|--------|--------|------|
| `--tqdm` | False | 터미널 실행 시 배치 진행 바 표시. 비대화형 환경(서버 로그)에서는 끄는 것 권장 |

## 저장된 모델로 추론하기

`--save_model`로 학습하면 두 파일이 저장된다.

```
experiments/gf_model/models/
├── checkpoint_{unique_name}.tar        # 모델 가중치
└── checkpoint_{unique_name}_args.json  # 학습에 사용한 플래그 전체
```

`_args.json`에는 `--emlps`, `--ports`, `--node_features` 등 모델 구조에 영향을 주는 플래그가 모두 기록된다. 추론 시 이 플래그들을 동일하게 넘겨야 모델 구조가 일치하여 로드가 성공한다.

**추론 실행 예시**

```bash
# 1. args.json 확인
cat experiments/gf_model/models/checkpoint_gf_run1_args.json

# 2. json의 플래그 그대로 --inference 붙여서 실행
python main.py --data Small_LI --model pna --emlps --reverse_mp --ego --ports --node_features --inference --unique_name gf_run1
```
