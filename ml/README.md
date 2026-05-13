# ML 파트 README

`ml/`은 `data/preprocessing` 단계에서 생성된 ML-ready parquet를 입력으로 받아 XGBoost 학습, validation threshold 선택, final test 평가를 수행한다. 이 문서는 ML 파트의 단일 운영 문서이다.

## 핵심 기준

| 항목 | 기준 |
|---|---|
| feature 생성 | `data/preprocessing/03_ml_feature_process.ipynb` 담당 |
| ML 역할 | feature 생성 없음. ML-ready parquet를 읽어 학습, threshold 선택, 평가만 수행 |
| 기본 흐름 | `ml_io.py` 입력 검증 후 `ml_train.py` -> `ml_val.py` -> `ml_test.py` |
| 작업 구조 | 노트북이 실행 흐름을 관리하고, 세부 기능은 `ml_*.py` 모듈에 위임 |
| 실행 진입점 | `ml/ml-00/ml-00_smoke_test.ipynb` |
| sample 결과 | smoke/debug 전용. 성능 주장 금지 |
| final test | full model + full validation threshold 확정 후 1회 평가 |
| commit 제외 | parquet, model, output artifact |

## 작업 구조

ML 파트 작업은 `노트북 메인 + 모듈 세부 기능` 구조를 따른다.

| 구분 | 역할 |
|---|---|
| 노트북 | 실험 실행의 메인 진입점. 경로, sample 여부, overwrite 정책, XGBoost 파라미터, threshold 전략을 설정하고 모듈 함수를 순서대로 호출 |
| 모듈 파일 | 세부 기능 구현. 입력 검증, feature column 로딩, 모델 학습, threshold 선택, metric 계산, final test 안전장치를 담당 |
| 수정 기준 | 실행 조건이나 실험 설정은 노트북에서 조정하고, 재사용되는 로직이나 검증 규칙은 `ml_*.py` 모듈에서 수정 |
| 금지 기준 | 노트북 안에 긴 학습/검증 로직을 직접 누적하지 않음. 노트북은 orchestration과 결과 확인 중심으로 유지 |

현재 기준 메인 노트북은 `ml/ml-00/ml-00_smoke_test.ipynb`다. 세부 기능은 같은 폴더의 `ml_io.py`, `ml_train.py`, `ml_val.py`, `ml_test.py`, `ml_metrics.py`가 담당한다.

## 전체 데이터 흐름(예시)

```text
data/raw/
  원본 IBM AML 파일

data/preprocessing/02_preprocessing_clean_base.ipynb
  -> data/processed/step01_clean_base/clean_base.parquet
  -> data/processed/step01_clean_base/account_mapping.csv
  -> data/processed/step01_clean_base/label_verification_report.csv

data/preprocessing/03_ml_feature_process.ipynb
  -> ML-00, ML-00 no-time, ML-01용 train/val/test parquet
  -> ml_feature_columns.csv
  -> feature_catalog.csv
  -> leakage_check.csv
  -> split_summary.csv

ml/ml-00/ml_train.py
  -> model.pkl
  -> feature_columns.json
  -> train_summary.json

ml/ml-00/ml_val.py
  -> threshold.json
  -> metrics_val.json
  -> confusion_matrix_val.csv

ml/ml-00/ml_test.py
  -> metrics_test.json
  -> confusion_matrix_test.csv
```

## 모듈 역할

| 파일 | 역할 |
|---|---|
| `ml_io.py` | 경로 해석, 입력 파일 확인, feature catalog 로딩, parquet 로딩, forbidden feature 차단, split 검증, feature hash, label metadata |
| `ml_metrics.py` | validation F1 기준 threshold 선택, F1/Recall/Precision/AP/confusion matrix 계산 |
| `ml_train.py` | train/val 기반 XGBoost 학습, `model.pkl`, `feature_columns.json`, `train_summary.json` 저장 |
| `ml_val.py` | validation threshold 선택, `threshold.json`, `metrics_val.json`, `confusion_matrix_val.csv` 저장 |
| `ml_test.py` | final test 전용. `confirm_final_test=True`와 provenance 검증 없이는 실행 차단 |
| `ml_utils.py` | ML-00 기준 seed 고정과 프로젝트 경로 상수 제공 |
| `ml_search_spaces.py` | XGBoost random search preset 정의 |
| `ml_tune.py` | XGBoost random search trial 실행. final test는 실행하지 않음 |
| `ml-00_smoke_test.ipynb` | 현재 기준 노트북 실행 및 smoke test 진입점. 데이터 경로, overwrite, XGBoost 파라미터, threshold 정책을 직접 설정 |

## 현재 입력 예시

현재 전처리 산출물 예시는 아래와 같다. 파일명은 고정 계약이 아니며 노트북 입력값으로 바꿀 수 있다.단, 개별 산출물의 양식이나 산출물의 형태는 고정되어야 한다.

```text
data/processed/ml_features/
  ml_exp00_Xy_train.parquet
  ml_exp00_Xy_val.parquet
  ml_exp00_Xy_test.parquet
  ml_exp00_no_time_Xy_train.parquet
  ml_exp00_no_time_Xy_val.parquet
  ml_exp00_no_time_Xy_test.parquet
  ml_exp01_Xy_train.parquet
  ml_exp01_Xy_val.parquet
  ml_exp01_Xy_test.parquet
  ml_feature_columns.csv
```

## 실행 모드

같은 노트북을 사용하더라도 smoketest와 실제 분석은 설정과 해석 기준을 분리한다. 노트북 설정 셀의 `DATA_DIR`, 파일명, `SAMPLE_ROWS`, `RUN_SMOKETEST_CASE_CHECKS`, `RUN_TRAIN_AND_VALIDATION`, `RUN_FINAL_TEST`가 실행 모드를 결정한다.

| 모드 | 목적 | 데이터 | 주요 설정 | 성능 해석 |
|---|---|---|---|---|
| Smoketest fixture 검증 | 입력 계약, contract/bad case, 누수 방어 로직 확인 | `ml/ml-96_smoketest/` | `RUN_SMOKETEST_CASE_CHECKS=True`, `RUN_TRAIN_AND_VALIDATION=False`, `RUN_FINAL_TEST=False` | 성능 주장 금지 |
| 실제 데이터 sample dry-run | 실제 parquet 경로와 학습/validation 흐름 점검 | `data/processed/ml_features/` | `RUN_SMOKETEST_CASE_CHECKS=False`, `SAMPLE_ROWS=100_000`, `RUN_FINAL_TEST=False` | 성능 주장 금지 |
| 실제 데이터 full validation | 모델, feature, threshold 선택 | `data/processed/ml_features/` | `RUN_SMOKETEST_CASE_CHECKS=False`, `SAMPLE_ROWS=None`, `RUN_FINAL_TEST=False` | validation은 내부 선택 지표 |
| Final test | 최종 1회 test 평가 | full test split | `SAMPLE_ROWS=None`, `RUN_TRAIN_AND_VALIDATION=True`, `RUN_FINAL_TEST=True` | 최종 성능 근거로 사용 가능 |

Smoketest는 코드와 입력 계약 검증용이며 실제 분석 결과로 해석하지 않는다. 실제 분석은 먼저 sample dry-run으로 경로와 실행 흐름을 확인한 뒤, `SAMPLE_ROWS=None`으로 full train/validation을 실행한다. final test는 모델, feature, threshold 선택이 끝난 뒤 한 번만 실행한다.

## 공통 노트북 설정

`ml-00`은 폴더명에 하이픈이 있어 일반 패키지 import가 어렵다. 노트북이 `ml/ml-00`에서 실행되면 같은 폴더의 `ml_*.py` 파일을 바로 import할 수 있다. 프로젝트 루트에서 실행한다면 `ml/ml-00`을 `sys.path`에 추가한다.

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "ml" / "ml-00"))

import ml_io
import ml_train
import ml_val
import ml_test
```

현재 기준 실행 진입점은 `ml/ml-00/ml-00_smoke_test.ipynb`다. 이 노트북에서 아래 값을 직접 입력하고, 이후 함수들은 설정 셀에서 만든 값을 재사용한다.

| 설정 | 의미 |
|---|---|
| `EXPERIMENT_NAME` | output directory 이름. smoketest, sample, full 실행을 서로 다른 이름으로 분리 |
| `DATA_DIR` | train/val/test parquet와 feature catalog가 있는 폴더 |
| `TRAIN_FILE_NAME`, `VAL_FILE_NAME`, `TEST_FILE_NAME` | split parquet 파일명 |
| `FEATURE_COLUMNS_FILE_NAME` | feature catalog CSV 파일명 |
| `SAMPLE_ROWS` | smoke/debug용 앞부분 row 제한. full 실행은 `None` |
| `RUN_SMOKETEST_CASE_CHECKS` | fixture 전용 contract/bad case 검증 실행 여부 |
| `RUN_TRAIN_AND_VALIDATION` | train과 validation threshold 선택 실행 여부 |
| `RUN_FINAL_TEST` | final test 실행 여부. 최종 평가 전까지 `False` 유지 |
| `INPUT_PATHS` | `make_input_paths()`로 한 번만 만든 입력 경로 객체 |
| `OVERWRITE_POLICY` | train/val/test artifact 덮어쓰기 허용 여부 |
| `XGB_PARAMS` | 노트북에서 `XGBTrainConfig` 기본값을 덮어쓸 XGBoost 파라미터 |
| `THRESHOLD_STRATEGY`, `MANUAL_THRESHOLD` | validation threshold 선택 방식 |

경로 선택은 별도 자동 분기 변수에 의존하지 않고, 설정 셀에서 직접 입력한다. `INPUT_PATHS`는 한 번만 만들고 preview, smoketest check, train, validation, final test에서 재사용한다.

```python
INPUT_PATHS = ml_io.make_input_paths(
    data_dir=DATA_DIR,
    train_file_name=TRAIN_FILE_NAME,
    val_file_name=VAL_FILE_NAME,
    test_file_name=TEST_FILE_NAME,
    feature_columns_file_name=FEATURE_COLUMNS_FILE_NAME,
    project_root=PROJECT_ROOT,
)
```

## Smoketest 실행

Smoketest는 `ml/ml-96_smoketest/`의 작은 fixture로 입력 검증 로직과 실패 케이스를 확인하는 절차다. 이 결과는 모델 성능으로 해석하지 않는다.

```python
EXPERIMENT_NAME = "ml-00-smoketest"
RUN_SMOKETEST_CASE_CHECKS = True
RUN_TRAIN_AND_VALIDATION = False
RUN_FINAL_TEST = False
SAMPLE_ROWS = None

DATA_DIR = BASE_DIR / "ml" / "ml-96_smoketest" / "ml_features"
TRAIN_FILE_NAME = "ml_exp00_Xy_train_smoketest.parquet"
VAL_FILE_NAME = "ml_exp00_Xy_val_smoketest.parquet"
TEST_FILE_NAME = "ml_exp00_Xy_test_smoketest.parquet"
FEATURE_COLUMNS_FILE_NAME = "ml_feature_columns_smoketest.csv"
```

fixture 데이터로 train/validation 함수 호출까지 확인하려면 `RUN_TRAIN_AND_VALIDATION=True`로 바꿀 수 있다. 이 경우에도 출력 metric은 smoke/debug 결과이며 성능 주장에 사용하지 않는다.

`RUN_SMOKETEST_CASE_CHECKS=True`인 상태에서 `DATA_DIR`이 `ml/ml-96_smoketest/ml_features`가 아니면 노트북은 fixture 전용 검증과 실제 데이터가 섞이지 않도록 중단한다.

## 실제 분석 실행

실제 분석에서는 `RUN_SMOKETEST_CASE_CHECKS=False`를 유지한다. `DATA_DIR`과 파일명은 전처리 산출물로 직접 지정한다.

```python
DATA_DIR = PROCESSED_DIR / "ml_features"
TRAIN_FILE_NAME = "ml_exp00_Xy_train.parquet"
VAL_FILE_NAME = "ml_exp00_Xy_val.parquet"
TEST_FILE_NAME = "ml_exp00_Xy_test.parquet"
FEATURE_COLUMNS_FILE_NAME = "ml_feature_columns.csv"
RUN_SMOKETEST_CASE_CHECKS = False
```

### 1. Sample Dry-Run

실제 parquet 경로, feature catalog, 모델 학습 호출, validation threshold 저장 흐름을 빠르게 확인한다. sample metric은 성능 주장에 사용하지 않는다.

```python
EXPERIMENT_NAME = "ml-00-sample"
SAMPLE_ROWS = 100_000
RUN_TRAIN_AND_VALIDATION = True
RUN_FINAL_TEST = False
```

### 2. Full Train/Validation

sample dry-run이 통과하면 전체 train/validation split으로 모델과 threshold를 선택한다. validation metric은 내부 선택 지표이며 최종 성능 주장이 아니다.

```python
EXPERIMENT_NAME = "ml-00-full"
SAMPLE_ROWS = None
RUN_TRAIN_AND_VALIDATION = True
RUN_FINAL_TEST = False
```

`XGB_PARAMS`를 넘기면 노트북 값이 모듈 파일의 `XGBTrainConfig` 기본값을 덮어쓴다. 모듈 기본값을 쓰려면 train config 생성부의 `**XGB_PARAMS` 전달을 제거한다.

### 3. Final Test

final test는 full train/validation에서 선택한 모델, feature, threshold가 확정된 뒤 1회만 실행한다. `ml_test.py`는 sampled model, sampled threshold, sampled test를 차단한다.

```python
EXPERIMENT_NAME = "ml-00-full"
SAMPLE_ROWS = None
RUN_TRAIN_AND_VALIDATION = True
RUN_FINAL_TEST = True
```

현재 노트북 흐름에서는 train/validation 실행 후 같은 `OUT_DIR`의 `model.pkl`, `feature_columns.json`, `threshold.json`을 사용해 final test가 이어진다. final test만 별도 디렉터리에서 실행하면 필요한 artifact가 없어 실패할 수 있으므로, full train/validation 산출물이 있는 같은 `EXPERIMENT_NAME`과 `OUT_DIR`을 사용한다.

## Threshold와 Metric 기준

현재 구조는 3-way split을 유지한다.

```text
train: 모델 fit
val: early stopping + threshold selection
test: 최종 1회 평가
```

validation split은 early stopping과 threshold selection에 함께 사용된다. 따라서 validation metric은 최종 성능이 아니라 내부 model/threshold 선택 참고 지표로 해석한다. 최종 성능 주장은 full test 1회 평가 결과만 사용한다.

| `threshold_strategy` | 의미 | 사용 기준 |
|---|---|---|
| `max_f1` | validation F1이 최대가 되는 threshold 자동 선택 | 기본값 |
| `manual` | 노트북의 `manual_threshold` 값을 고정 threshold로 사용 | 운영 기준 비교나 민감도 분석 시 제한적으로 사용 |

`max_f1`의 F1은 `label=1`인 positive class 기준이다. 현재 코드 기준은 아래와 같다.

| 기준 | 내용 |
|---|---|
| positive class | `label=1`, AML/자금세탁 거래 |
| negative class | `label=0`, 정상 거래 |
| score | `model.predict_proba(x)[:, 1]`, class `1` 확률 |
| prediction | `probability >= threshold`이면 `1` |
| F1 계산 | `sklearn.metrics.f1_score()` 기본값. binary `pos_label=1` 기준 |

test split에서 threshold를 다시 고르면 데이터 누수 위험이 있으므로, `ml_test.py`는 `threshold.json`에 저장된 validation threshold만 사용한다.

## 출력 Artifact 계약

| 파일 | 생성 모듈 | 의미 |
|---|---|---|
| `model.pkl` | `ml_train.py` | 학습된 XGBoost 모델 |
| `feature_columns.json` | `ml_train.py` | 학습에 사용한 feature 순서와 hash |
| `train_summary.json` | `ml_train.py` | 입력 경로, sample 여부, label 분포, hyperparameter |
| `threshold.json` | `ml_val.py` | validation threshold와 provenance |
| `metrics_val.json` | `ml_val.py` | validation metric과 label 분포 |
| `confusion_matrix_val.csv` | `ml_val.py` | validation confusion matrix |
| `metrics_test.json` | `ml_test.py` | final test metric과 provenance |
| `confusion_matrix_test.csv` | `ml_test.py` | final test confusion matrix |

`threshold.json`에는 `threshold_strategy`, `manual_threshold`, `feature_columns_hash`, `train_sampled`, `train_sample_rows`가 함께 저장된다. 이 값은 final test provenance 검증과 사후 해석에 사용한다.

## 안전장치

| 위험 | 방어 |
|---|---|
| label leakage | `label`, `target`, `y`, `is_laundering` exact name 차단 |
| pattern leakage | `laundering`, `pattern`, `typology`, `attempt` substring 차단 |
| feature order 불일치 | order-sensitive `feature_columns_hash` 저장 및 검증 |
| split 파일 오입력 | `split` 컬럼이 있으면 train/val/test 값 검증 |
| sampled model final test 사용 | `train_summary.json`의 `sampled=True` 차단 |
| sampled threshold final test 사용 | `threshold.json`의 `sampled=True` 차단 |
| sampled test | `ml_test.py`에서 `sample_rows is not None` 차단 |
| final 결과 overwrite | `overwrite=False` 기본값으로 기존 test artifact 존재 시 실패 |
| threshold 재조정 누수 | `ml_val.py`가 저장한 `threshold.json`을 `ml_test.py`가 그대로 사용 |
| 설정 분산 | 노트북 설정 셀에서 `INPUT_PATHS`, `OVERWRITE_POLICY`, `XGB_PARAMS`, `THRESHOLD_STRATEGY`를 한 번만 지정 |

## 운영 원칙

| 원칙 | 내용 |
|---|---|
| 원본/전처리 산출물 | ML 파트에서 수정하지 않음 |
| 대용량 파일 | parquet, model, 결과 파일 commit 금지 |
| test 사용 | 모델/feature/threshold 선택 종료 후 1회 |
| overwrite | 노트북의 `OVERWRITE_POLICY`에서 train/val/test별로 명시 |
| seed | ML-00에서는 `ml_utils.set_seed()`를 기준으로 사용하고 `XGBTrainConfig.seed`에도 같은 값을 전달 |
| XGBoost 설정 | 노트북의 `XGB_PARAMS`가 `XGBTrainConfig` 기본값을 덮어씀. 모듈 기본값을 쓰려면 `**XGB_PARAMS` 전달을 제거 |
| 문서 갱신 | 입력 계약, 실행 방식, artifact 의미가 바뀌면 이 `README.md`를 갱신 |

## 검증

```text
ml/ml-00/ml-00_smoke_test.ipynb
```

검증은 위 노트북에서 수행한다. fixture 검증은 `RUN_SMOKETEST_CASE_CHECKS=True`, 실제 데이터 실행은 `RUN_SMOKETEST_CASE_CHECKS=False`로 두고 실행한다.
