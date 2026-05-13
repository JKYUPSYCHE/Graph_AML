# ML-96 Smoketest Fixtures

이 폴더는 ML-00 이후 실험에서 계속 재사용할 수 있는 로컬 검증용 데이터셋을 관리한다.

목적은 모델 성능 향상이 아니라 다음을 빠르게 확인하는 것이다.

- 입력 파일 경로가 올바른지
- feature catalog가 모듈 규격에 맞는지
- label 누수, 컬럼 누락, split 오염을 조용히 통과시키지 않는지
- 정상 데이터는 train/validation 흐름을 실행할 수 있는지

## 현재 커밋 기준

현재 커밋 대상에는 fixture 생성 스크립트와 생성된 parquet/csv 파일을 포함하지 않는다.

커밋 대상은 문서와 빈 디렉터리 구조 유지용 `.gitkeep` 파일만 포함한다.

```text
ml/ml-96_smoketest/Smoketest 관련 메모.md
ml/ml-96_smoketest/ml_features/.gitkeep
ml/ml-96_smoketest/contract_cases/.gitkeep
ml/ml-96_smoketest/bad_cases/.gitkeep
```

fresh clone 직후에는 아래 fixture parquet/csv 파일이 존재하지 않는다. 따라서 smoketest 실행 전에 fixture 파일을 별도로 준비해야 한다.

## 정상 데이터

```text
ml_features/
  ml_exp00_Xy_train_smoketest.parquet
  ml_exp00_Xy_val_smoketest.parquet
  ml_exp00_Xy_test_smoketest.parquet
  ml_feature_columns_smoketest.csv
```

노트북에서는 fixture 여부를 자동 분기하지 않는다. fixture 파일이 준비된 경우에만 `ml/ml-00/ml-00_smoke_test.ipynb` 설정 셀에서 `DATA_DIR`과 파일명 4개를 위 파일로 직접 입력해 사용한다.

## Contract Cases

정상적으로 통과해야 하는 운영 상황

```text
contract_cases/
  val_shuffled_columns_smoketest_contract_cases.parquet
  val_extra_column_smoketest_contract_cases.parquet
```

- `val_shuffled_columns...`: parquet 물리 컬럼 순서가 바뀌어도 feature catalog 순서로 읽히는지 확인
- `val_extra_column...`: 모델에 쓰지 않는 추가 컬럼이 있어도 조용히 실패하지 않고 정상 처리되는지 확인

## Bad Cases

명확한 에러로 실패해야 하는 데이터

```text
bad_cases/
  val_missing_feature_smoketest_bad_cases.parquet
  val_missing_label_smoketest_bad_cases.parquet
  val_wrong_split_smoketest_bad_cases.parquet
  val_label_leak_catalog_smoketest_bad_cases.csv
  val_nan_feature_smoketest_bad_cases.parquet
  val_wrong_dtype_smoketest_bad_cases.parquet
  val_null_explosion_smoketest_bad_cases.parquet
  val_inf_feature_smoketest_bad_cases.parquet
  val_single_class_smoketest_bad_cases.parquet
  val_duplicate_feature_catalog_smoketest_bad_cases.csv
  val_empty_feature_catalog_smoketest_bad_cases.csv
  val_invalid_used_in_ml_catalog_smoketest_bad_cases.csv
  val_forbidden_name_catalog_smoketest_bad_cases.csv
```

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

이 검사는 XGBoost 학습 없이 `ml_io`의 입력/스키마/누수 방지 로직만 확인한다. 단, 현재 커밋에는 fixture parquet/csv가 포함되지 않으므로 파일이 없는 상태에서는 실행할 수 없다.

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
