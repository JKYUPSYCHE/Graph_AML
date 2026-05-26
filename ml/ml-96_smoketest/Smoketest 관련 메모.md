# ML-96 Smoketest Fixtures

이 폴더는 ML-00 이후 실험에서 계속 재사용할 수 있는 로컬 검증용 데이터셋을 관리한다.

목적은 모델 성능 향상이 아니라 다음을 빠르게 확인하는 것이다.

- 입력 파일 경로가 올바른지
- feature catalog가 모듈 규격에 맞는지
- 프로젝트형 Graph Feature 이름과 stage/group 메타데이터가 ML 입력 계약을 깨지 않는지
- label 누수, 컬럼 누락, split 오염을 조용히 통과시키지 않는지
- 정상 데이터는 train/validation 흐름을 실행할 수 있는지

## 현재 커밋 기준

현재 커밋 대상에는 fixture 생성 스크립트, 검증 스크립트, manifest, 생성된 parquet/csv 파일을 포함할 수 있다.

`ml/ml-96_smoketest/` 아래 파일은 synthetic contract fixture이며 원천 데이터나 실제 실험 성능 산출물이 아니다.

```text
ml/ml-96_smoketest/Smoketest 관련 메모.md
ml/ml-96_smoketest/generate_smoketest_fixtures.py
ml/ml-96_smoketest/run_smoketest_case_checks.py
ml/ml-96_smoketest/smoketest_manifest.json
ml/ml-96_smoketest/ml_features/.gitkeep
ml/ml-96_smoketest/contract_cases/.gitkeep
ml/ml-96_smoketest/bad_cases/.gitkeep
```

fresh clone 직후 fixture parquet/csv 파일이 없거나 오래된 경우 아래 명령으로 재생성한다.

```bash
python ml/ml-96_smoketest/generate_smoketest_fixtures.py --overwrite
```

`smoketest_manifest.json`은 fixture 목적과 금지 용도를 명시한다.

```json
{
  "purpose": "schema_contract_smoketest_only",
  "not_for": ["model_performance", "feature_importance", "business_validation"],
  "label_signal_injected": true
}
```

이 fixture에는 smoke 검증을 위해 label signal이 일부 feature에 주입되어 있다. 모델 성능, feature importance, 업무 검증 근거로 사용하지 않는다.

## 정상 데이터

```text
ml_features/
  ml_exp00_Xy_train_smoketest.parquet
  ml_exp00_Xy_val_smoketest.parquet
  ml_exp00_Xy_test_smoketest.parquet
  ml_feature_columns_smoketest.csv
```

`ml_feature_columns_smoketest.csv`의 `used_in_ml`은 `TRUE` / `FALSE` 문자열만 사용한다. 이는 `ml_io.parse_used_in_ml()`의 strict 정책과 맞추기 위한 계약이다.

현재 synthetic feature는 성능 주장용 `feature_00` 계열이 아니라 프로젝트 방향에 맞춘 AML Graph Feature 형태를 사용한다.

```text
tx_amount_log
tx_hour_sin
tx_hour_cos
sender_tx_count_1d
receiver_tx_count_1d
sender_out_amount_mean_30d
receiver_in_amount_mean_30d
sender_fanout_7d
receiver_fanin_7d
pair_tx_count_30d
pair_amount_sum_30d
pass_through_balance_1d
flow_burst_score_1h
cycle_proxy_count_30d
```

feature group은 smoke 검증용 메타데이터이며 실제 실험 성능 수치로 해석하지 않는다.

`tx_hour_sin`, `tx_hour_cos`는 0~23 hour를 생성한 뒤 실제 cyclic 값으로 계산한다. 따라서 값 범위는 `[-1, 1]`이다.

노트북에서는 fixture 여부를 자동 분기하지 않는다. fixture 파일이 준비된 경우에만 `ml/ml-00/ml-00_smoke_test.ipynb` 설정 셀에서 `DATA_DIR`과 파일명 4개를 위 파일로 직접 입력해 사용한다.

## Contract Cases

정상적으로 통과해야 하는 운영 상황

```text
contract_cases/
  val_shuffled_columns_smoketest_contract_cases.parquet
  val_extra_column_smoketest_contract_cases.parquet
  val_project_feature_names_smoketest_contract_cases.parquet
```

- `val_shuffled_columns...`: parquet 물리 컬럼 순서가 바뀌어도 feature catalog 순서로 읽히는지 확인
- `val_extra_column...`: 모델에 쓰지 않는 추가 컬럼이 있어도 조용히 실패하지 않고 정상 처리되는지 확인
- `val_project_feature_names...`: 프로젝트형 feature 이름을 사용하는 정상 fixture가 통과하는지 확인

## Bad Cases

명확한 에러로 실패해야 하는 데이터

```text
bad_cases/
  val_missing_feature_smoketest_bad_cases.parquet
  val_missing_multiple_features_smoketest_bad_cases.parquet
  val_missing_label_smoketest_bad_cases.parquet
  val_nan_label_smoketest_bad_cases.parquet
  val_non_binary_label_smoketest_bad_cases.parquet
  val_string_label_smoketest_bad_cases.parquet
  val_wrong_split_smoketest_bad_cases.parquet
  val_missing_split_smoketest_bad_cases.parquet
  val_null_split_smoketest_bad_cases.parquet
  val_blank_split_smoketest_bad_cases.parquet
  val_mixed_split_smoketest_bad_cases.parquet
  val_label_leak_catalog_smoketest_bad_cases.csv
  val_nan_feature_smoketest_bad_cases.parquet
  val_wrong_dtype_smoketest_bad_cases.parquet
  val_null_explosion_smoketest_bad_cases.parquet
  val_inf_feature_smoketest_bad_cases.parquet
  val_negative_inf_feature_smoketest_bad_cases.parquet
  val_empty_rows_smoketest_bad_cases.parquet
  val_single_class_smoketest_bad_cases.parquet
  val_duplicate_feature_catalog_smoketest_bad_cases.csv
  val_empty_feature_catalog_smoketest_bad_cases.csv
  val_invalid_used_in_ml_catalog_smoketest_bad_cases.csv
  val_forbidden_name_catalog_smoketest_bad_cases.csv
  val_target_name_catalog_smoketest_bad_cases.csv
  val_y_name_catalog_smoketest_bad_cases.csv
  val_pattern_name_catalog_smoketest_bad_cases.csv
  val_missing_column_name_catalog_smoketest_bad_cases.csv
  val_missing_used_in_ml_catalog_smoketest_bad_cases.csv
  val_blank_selected_feature_catalog_smoketest_bad_cases.csv
  val_missing_selected_feature_name_catalog_smoketest_bad_cases.csv
```

Bad case 검증 범위는 다음과 같다.

- schema: 필수 feature, label, split 컬럼 누락
- split contamination: wrong/null/blank/mixed split
- label integrity: NaN, non-binary, non-numeric string, single-class
- feature integrity: NaN, all-null, non-numeric dtype, `inf`, `-inf`
- catalog integrity: 중복 feature, 빈 feature set, `used_in_ml` 오류, 필수 CSV 컬럼 누락, blank/missing selected feature name
- leakage guard: `label`, `target`, `y`, `laundering`, `pattern` 계열 feature 선택 차단

## CLI 검증

노트북 없이 fixture 검증만 실행할 수 있다.

```bash
python ml/ml-96_smoketest/generate_smoketest_fixtures.py --dry-run
python ml/ml-96_smoketest/generate_smoketest_fixtures.py --overwrite
python ml/ml-96_smoketest/run_smoketest_case_checks.py --io-module ml-01
```

fixture generator는 다음을 보장한다.

- parquet 저장 시 `pyarrow` engine을 명시적으로 사용한다.
- `pyarrow`가 없으면 `pip install -r requirements.txt` 안내와 함께 실패한다.
- 생성 후 planned file 존재 여부를 확인한다.
- parquet 파일은 기대 row count와 실제 row count가 일치하는지 확인한다.
- manifest를 생성해 schema/contract smoketest 전용임을 명시한다.

ML-00 freeze 입력 로직과의 호환성을 확인하려면 아래 명령을 사용한다.

```bash
python ml/ml-96_smoketest/run_smoketest_case_checks.py --io-module ml-00
```

정상 종료 기준은 마지막 줄에 `[SMOKETEST CASE CHECKS PASS]`가 출력되는 것이다.

## 노트북 사용법

`ml/ml-00/ml-00_smoke_test.ipynb`에서 아래처럼 데이터 경로와 파일명을 직접 입력

```python
DATA_DIR = BASE_DIR / "ml" / "ml-96_smoketest" / "ml_features"
TRAIN_FILE_NAME = "ml_exp00_Xy_train_smoketest.parquet"
VAL_FILE_NAME = "ml_exp00_Xy_val_smoketest.parquet"
TEST_FILE_NAME = "ml_exp00_Xy_test_smoketest.parquet"
FEATURE_COLUMNS_FILE_NAME = "ml_feature_columns_smoketest.csv"
```

contract/bad case를 노트북에서 확인하려면 fixture 전용 검증 옵션을 True 로 설정 한다.

```python
RUN_SMOKETEST_CASE_CHECKS = True
```

이 검사는 XGBoost 학습 없이 `ml_io`의 입력/스키마/누수 방지 로직만 확인한다. fixture parquet/csv가 없거나 오래된 경우 `generate_smoketest_fixtures.py --overwrite`로 먼저 재생성한다.

실제 데이터로 전환하려면 `DATA_DIR`과 파일명 4개를 실제 전처리 산출물로 직접 바꾸고, fixture 전용 검증은 False 처리

```python
DATA_DIR = PROCESSED_DIR / "ml_features"
TRAIN_FILE_NAME = "ml_exp00_Xy_train.parquet"
VAL_FILE_NAME = "ml_exp00_Xy_val.parquet"
TEST_FILE_NAME = "ml_exp00_Xy_test.parquet"
FEATURE_COLUMNS_FILE_NAME = "ml_feature_columns.csv"
RUN_SMOKETEST_CASE_CHECKS = False
```

`RUN_SMOKETEST_CASE_CHECKS=True`인 상태에서 `DATA_DIR`이 `ml/ml-96_smoketest/ml_features`가 아니면 노트북은 명확한 `ValueError`로 중단한다. 실제 데이터와 fixture 검증이 섞이는 것을 막기 위한 안전장치다.
