# ML 파트 README

`ml/`은 전처리와 feature build 단계에서 생성된 ML-ready parquet를 입력으로 받아 XGBoost 학습, validation threshold 선택, final test 평가를 수행하는 작업 영역이다. 현재 운영 기준에서는 `ML-00`을 파이프라인 구성을 위한 사전 실험으로 보고, 본격적인 실험 번호 체계와 입출력 양식은 `ML-01`부터 적용한다.

## 현재 기준 요약

| 항목 | 현재 기준 |
|---|---|
| `ML-00` 위치 | `ml/ml-00/`, `ml/ml-00_baseline/` |
| `ML-00` 해석 | 파이프라인, smoke test, 입력 검증, baseline 구성 방식을 확인하기 위한 사전 실험 |
| 본격 실험 시작점 | `ml/ml-01/` |
| 현재 실행 진입점 | `ml/ml-01/01_train_val_test/00_ml_01_ml_run_00.ipynb` |
| 현재 모델 | XGBoost binary classifier |
| 현재 실험 성격 | ML-01 Stage 0. 파이프라인 사전 실험에서 정리한 기준 입력에 시간 이력 feature를 추가한 비교 실험 |
| feature 생성 | train/validation/test 노트북에서 수행하지 않음 |
| feature 승인 | `fb_outputs` 검토 후 `ml_inputs/<RUN_ID>/`에 승인 배치된 파일만 학습 입력으로 사용 |
| feature 선택 기준 | 승인 CSV에서 `used_in_ml="TRUE"`인 컬럼 |
| threshold 선택 | validation set에서만 수행 |
| final test | 기본 잠금. full artifact 확정 후 사용자가 명시적으로 실행 |
| commit 제외 | parquet, model, output artifact |

과거 문서나 산출물에서 `ML-00 baseline`이라는 표현이 남아 있어도, 현재 `ml/` 코드 운영 기준의 정식 실험 입출력 계약은 `ML-01`부터 보는 것이 안전하다. `ML-00` 결과는 파이프라인 설계와 검증 이력으로만 해석하고, 성능 주장이나 최종 비교표에는 검증된 `ML-01` 이후 artifact를 기준으로 사용한다.

## 디렉터리 구조

```text
ml/
  README.md
  AGENTS.md
  260520_feature_catalog.csv

  ml-00/
    ml-00_smoke_test.ipynb
    ml_io.py, ml_train.py, ml_val.py, ml_test.py, ...
    # 파이프라인 사전 실험 및 smoke/contract 검증용

  ml-00_baseline/
    feature_build/
    # 완료된 기준선/feature build 이력 영역. 기본적으로 읽기 전용으로 취급

  ml-01/
    fb_inputs/
    fb_outputs/
    ml_inputs/
    ml_outputs/
    01_train_val_test/
      00_ml_01_ml_run_00.ipynb
      ml_01_ml_io.py
      ml_01_ml_preview.py
      ml_01_ml_train.py
      ml_01_ml_val.py
      ml_01_ml_test.py
      ml_01_ml_metrics.py
      ml_01_ml_resource.py
      ml_01_ml_tune.py
      ml_01_ml_search_spaces.py
      ml_01_ml_utils.py

  ml-02/
    # 후속 실험 자리

  ml-96_smoketest/
    # 작은 fixture와 실패 케이스 검증용

  ml-99_others/
    # 이전 메모, 설명 문서, 보조 자료
```

`ml_inputs/`, `ml_outputs/`, `fb_inputs/`, `fb_outputs/` 아래의 parquet, model, 결과 artifact는 재현성 확인에는 필요할 수 있지만 commit 대상으로 보지 않는다.

## ML-01 데이터 흐름

```text
feature build input
  -> ml/ml-01/fb_inputs/<RUN_ID>/

feature build output
  -> ml/ml-01/fb_outputs/<RUN_ID>/

사람 검토 및 승인
  -> ml/ml-01/ml_inputs/<RUN_ID>/

XGBoost train / validation / final test
  -> ml/ml-01/01_train_val_test/00_ml_01_ml_run_00.ipynb
  -> ml_01_ml_io.py
  -> ml_01_ml_train.py
  -> ml_01_ml_val.py
  -> ml_01_ml_test.py

실험 산출물
  -> ml/ml-01/ml_outputs/<RUN_ID>/
```

현재 ML-01 노트북은 feature 생성이나 `fb_outputs -> ml_inputs` 복사를 수행하지 않는다. `ml_inputs/<RUN_ID>/`에 이미 승인 배치된 입력 묶음만 읽는다.

## ML-01 입력 계약

기본 식별자는 노트북 설정 셀에서 관리한다.

| 변수 | 의미 | 현재 예시 |
|---|---|---|
| `EXPORT_EXPERIMENT_ID` | 실험 ID | `ml_01` |
| `RUN_ID` | 승인 입력 묶음 ID | `r00` |
| `MODEL_RUN_ID` | 같은 입력으로 반복하는 모델 run ID | `d00` |
| `ARTIFACT_PREFIX` | 입력 파일 prefix | `ml_01__r00` |
| `ML_ARTIFACT_PREFIX` | 모델 산출물 prefix | `ml_01__r00__d00` |

현재 ML-01 입력 파일 양식은 아래와 같다.

```text
ml/ml-01/ml_inputs/<RUN_ID>/
  <ARTIFACT_PREFIX>_Xy_train.parquet
  <ARTIFACT_PREFIX>_Xy_val.parquet
  <ARTIFACT_PREFIX>_Xy_test.parquet
  <ARTIFACT_PREFIX>_fb_output_feature_contract_approve.csv
  <ARTIFACT_PREFIX>_encoding_manifest.json
  <ARTIFACT_PREFIX>_split_summary.csv
```

핵심 기준은 승인 CSV다. 모델 입력 feature는 `<ARTIFACT_PREFIX>_fb_output_feature_contract_approve.csv`에서 `used_in_ml="TRUE"`인 `column_name`만 의미한다. `fb_outputs`의 후보 contract, manifest, feature info는 승인 전 상태일 수 있으므로 실제 학습 feature 판단은 승인 CSV, `feature_columns.json`, `train_summary.json`의 feature list와 hash를 기준으로 한다.

`encoding_manifest.json`은 native categorical dtype 복원과 검증에 사용한다. manifest를 사용하는 run에서는 train, validation, final test가 같은 manifest를 참조해야 한다.

## ML-01 출력 계약

현재 출력 위치는 아래 형태다.

```text
ml/ml-01/ml_outputs/<RUN_ID>/
  <ML_ARTIFACT_PREFIX>_model.pkl
  <ML_ARTIFACT_PREFIX>_feature_columns.json
  <ML_ARTIFACT_PREFIX>_train_summary.json
  <ML_ARTIFACT_PREFIX>_feature_importance.csv
  <ML_ARTIFACT_PREFIX>_threshold.json
  <ML_ARTIFACT_PREFIX>_metrics_val.json
  <ML_ARTIFACT_PREFIX>_confusion_matrix_val.csv
  <ML_ARTIFACT_PREFIX>_metrics_test.json
  <ML_ARTIFACT_PREFIX>_confusion_matrix_test.csv
```

`model.pkl`, `feature_columns.json`, `train_summary.json`, `feature_importance.csv`는 train 단계 산출물이다. `threshold.json`, `metrics_val.json`, `confusion_matrix_val.csv`는 validation 단계 산출물이다. `metrics_test.json`, `confusion_matrix_test.csv`는 final test 단계 산출물이며, 최종 설정 확정 전에는 생성하지 않는다.

같은 입력으로 재실행할 때는 기존 파일을 덮어쓰기보다 `MODEL_RUN_ID`를 새로 부여하는 방식을 우선한다. `OVERWRITE_OUTPUTS=False`가 기본값이다.

## 코드 아키텍처

| 파일 | 역할 |
|---|---|
| `00_ml_01_ml_run_00.ipynb` | ML-01 train/validation/final-test orchestration. 경로, run ID, XGBoost 파라미터, 실행 스위치를 설정 |
| `ml_01_ml_io.py` | 경로 해석, 입력 파일 검증, 승인 feature contract 파싱, forbidden feature 차단, parquet schema/split/label 검증, feature hash, JSON 저장/로드 |
| `ml_01_ml_preview.py` | 승인 feature와 split summary, parquet schema를 학습 전 미리 확인하는 preview helper. 모델 학습과 artifact 저장은 하지 않음 |
| `ml_01_ml_train.py` | train/val split 로드, `scale_pos_weight` 계산, XGBoost 학습, native categorical 적용, train artifact 저장 |
| `ml_01_ml_val.py` | 저장된 모델과 feature 순서로 validation probability 계산, threshold 선택, validation metric 저장 |
| `ml_01_ml_test.py` | final test 전용. `confirm_final_test=True`, full artifact, provenance 검증 없이는 실행 차단 |
| `ml_01_ml_metrics.py` | F1 기준 threshold 선택, F1/Recall/Precision/AP/confusion matrix 계산 |
| `ml_01_ml_resource.py` | runtime, CPU memory, 환경 정보, data profile, score profile, feature importance 등 진단 metadata 수집 |
| `ml_01_ml_tune.py` | XGBoost random search 실행. final test는 수행하지 않음 |
| `ml_01_ml_search_spaces.py` | tuning search space preset 정의 |
| `ml_01_ml_utils.py` | seed 고정 helper |

노트북은 실행 순서와 설정을 관리하고, 재사용 로직은 `ml_01_ml_*.py` 모듈에 둔다. 긴 학습/검증 로직을 노트북에 누적하지 않는 것이 현재 구조의 기준이다.

## 실행 모드

| 모드 | 설정 | 해석 |
|---|---|---|
| Sample/debug | `SAMPLE_ROWS`에 양의 정수 지정, `RUN_FINAL_TEST=False` | 경로, 입력 계약, 학습 호출 확인용. 성능 주장 금지 |
| Full train/validation | `SAMPLE_ROWS=None`, `RUN_TRAIN_AND_VALIDATION=True`, `RUN_FINAL_TEST=False` | 모델과 validation threshold 선택용. validation metric은 내부 선택 지표 |
| Final test | `SAMPLE_ROWS=None`, `RUN_TRAIN_AND_VALIDATION=False` 또는 필요 시 유지, `RUN_FINAL_TEST=True` | full model과 validation threshold 확정 후 최종 1회 평가 |

final test는 기본 잠금 상태다. `ml_01_ml_test.py`는 sampled model, sampled threshold, sampled test를 차단하고, `threshold.json`의 validation threshold만 사용한다.

## 안전장치와 운영 원칙

| 기준 | 내용 |
|---|---|
| feature 승인 | 승인 CSV의 `used_in_ml="TRUE"`만 학습 feature로 사용 |
| 누수 위험 컬럼 | `label`, `target`, `y`, `is_laundering` exact name과 `laundering`, `pattern`, `typology`, `attempt` substring 차단 |
| split 경계 | train은 fit, validation은 threshold 선택, test는 최종 평가로만 사용 |
| threshold | validation에서만 선택하고 test에서 재조정하지 않음 |
| feature 순서 | `feature_columns_hash`로 저장 및 검증 |
| label | binary `label=1`을 AML/positive class로 해석 |
| class imbalance | train label 기준 `scale_pos_weight` 계산 |
| overwrite | 기본 차단. 재실행은 새 `MODEL_RUN_ID` 권장 |
| sample 결과 | smoke/debug 전용. 성능 주장 금지 |
| final test | full artifact 확정 후 1회 실행 원칙 |
| seed | 기본 `42`, 노트북 시작부에서 `ml_01_ml_utils.set_seed(SEED)` 호출 |

## 현재 주의사항

| 항목 | 내용 |
|---|---|
| `ML-00` 표기 잔존 | 일부 오래된 주석, docstring, 에러 메시지에 `ml_00` 표기가 남아 있을 수 있다. 실행 기준은 현재 파일 경로와 `ml_01_ml_*` 모듈명, `ML_ARTIFACT_PREFIX`를 우선한다. |
| `ml-00_baseline/` | 완료된 기준선/feature build 이력 영역이다. 신규 수정, 재학습, 산출물 재생성은 사용자 승인 후에만 수행한다. |
| artifact 상태 | 로컬에 parquet/model/result 파일이 존재할 수 있으나 README는 파일 존재를 성능 근거로 해석하지 않는다. |

## 문서 갱신 기준

다음이 바뀌면 이 README를 함께 갱신한다.

| 변경 | 갱신할 내용 |
|---|---|
| 신규 실험 폴더 추가 | 실행 진입점, 입력/출력 prefix, run ID 정책 |
| feature 승인 양식 변경 | 승인 CSV 파일명, 필수 컬럼, `used_in_ml` 정책 |
| artifact 파일명 변경 | 출력 계약과 final test provenance 기준 |
| train/validation/test 역할 변경 | 누수 방지 원칙과 실행 모드 |
| 자동화 테스트 추가 | 검증 명령과 실패 처리 기준 |