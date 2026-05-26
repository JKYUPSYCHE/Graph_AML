# ML 파트 README

`ml/`은 전처리와 feature build 단계에서 생성된 ML-ready parquet를 입력으로 받아 XGBoost 학습, validation threshold 선택, 대표 후보 test 평가를 수행하는 작업 영역이다. 현재 PART 2 운영 기준에서는 `ML-00`을 baseline freeze로 두고, `ML-01~ML-02`는 일반 집계 피처, `ML-03~ML-05`는 Graph Feature 추가 실험으로 순차 누적한다.

## 현재 기준 요약

| 항목 | 현재 기준 |
|---|---|
| `ML-00` 위치 | `ml/ml-00_baseline-freeze/` |
| `ML-00` 해석 | 현재 거래 row 기반 baseline freeze. 후속 실험 비교 기준 |
| 본격 누적 실험 시작점 | `ml/ml-01/` |
| 현재 실행 진입점 | `ml/ml-01/01_train_val_test/00_ml_01_ml_run_00.ipynb` |
| 현재 모델 | XGBoost binary classifier |
| 현재 실험 성격 | Stage 0~5 순차 누적. `ML-01~ML-02`는 aggregate-only, `ML-03~ML-05`는 aggregate + graph feature |
| feature 생성 | train/validation/test 노트북에서 수행하지 않음 |
| feature 승인 | `fb_outputs` 검토 후 `ml_inputs/<RUN_ID>/`에 승인 배치된 파일만 학습 입력으로 사용 |
| feature 선택 기준 | 승인 CSV에서 `used_in_ml="TRUE"`인 컬럼 |
| threshold 선택 | validation set에서만 수행 |
| test 평가 | 기본 잠금. `ML-02 tuned`, `ML-05 tuned`, `reduced/final tuned` 대표 후보에 한정해 사용자가 명시적으로 실행 |
| commit 제외 | parquet, model, output artifact |

## 실험 매트릭스

| 실험 ID | 피처 구성 | 분류 | 평가 원칙 |
|---|---|---|---|
| `ML-00` | 현재 거래 row 기반 baseline | baseline freeze | 기준 성능 고정 |
| `ML-01` | `ML-00` + Stage 0 시간 이력 피처 | 일반 집계 피처 | 고정 파라미터 validation 비교 |
| `ML-02` | `ML-01` + Stage 1 계좌별 통계 피처 | 일반 집계 피처 대표 후보 | 고정 파라미터 validation 후 tuning/test 대상 |
| `ML-03` | `ML-02` + Stage 2 fan-in/fan-out | Graph Feature 시작 | 고정 파라미터 validation 비교 |
| `ML-04` | `ML-03` + Stage 3 sender-receiver 관계 및 bank-pair corridor 반복성 | Graph Feature | 고정 파라미터 validation 비교 |
| `ML-05` | `ML-04` + Stage 4 pass-through/flow-balance | Graph Feature 대표 후보 | 고정 파라미터 validation 후 tuning/test 대상 |
| `ML-06` | Stage 0~4 품질/누수/중복/중요도 기반 축소 피처셋 | reduced 후보 | validation 비교 |
| `ML-07` | 최소 ablation 기반 practical/final candidate | ablation 후보 | validation 비교 |
| `ML-08` | 최종 피처셋 + tuning | reduced/final tuned 후보 | validation 기준 확정 후 test 대상 |
| `ML-09` | Stage 5 gather-scatter/2-hop/topology 선택 실험 | high-cost optional graph | 핵심 일정 이후 선택 실행 |

Stage 0~5는 계속 순차 누적한다. 바뀐 점은 Stage 1을 계좌별 통계로 두고 Stage 2부터 fan-in/fan-out Graph Feature를 시작한다는 점이다. 모든 Stage에서 test를 반복하지 않고, 대표 후보인 `ML-02`, `ML-05`, `reduced/final`만 tuning 후 test를 수행한다.

## 디렉터리 구조

```text
ml/
  README.md
  AGENTS.md
  260521_feature_catalog.csv

  ml-00_baseline-freeze/
    feature_build/
    train_val_test/
    # ML-00 baseline freeze 영역. 기본적으로 읽기 전용으로 취급

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
    # Stage 1 계좌별 통계 피처 실험

  ml-03/
    # Stage 2 fan-in/fan-out Graph Feature 실험

  ml-04/
    # Stage 3 sender-receiver 관계 및 bank-pair corridor 반복성 피처 실험

  ml-05/
    # Stage 4 pass-through/flow-balance 피처 실험

  ml-06/
  ml-07/
  ml-08/
    # 축소, ablation, final tuned 후보 실험

  ml-96_smoketest/
    # 작은 fixture와 실패 케이스 검증용
```

`ml_inputs/`, `ml_outputs/`, `fb_inputs/`, `fb_outputs/` 아래의 parquet, model, 결과 artifact는 재현성 확인에는 필요할 수 있지만 commit 대상으로 보지 않는다.

## 공통 데이터 흐름

```text
feature build input
  -> ml/<EXPERIMENT_ID>/fb_inputs/<RUN_ID>/

feature build output
  -> ml/<EXPERIMENT_ID>/fb_outputs/<RUN_ID>/

사람 검토 및 승인
  -> ml/<EXPERIMENT_ID>/ml_inputs/<RUN_ID>/

XGBoost train / validation / 대표 후보 test
  -> ml/ml-01/01_train_val_test/00_ml_01_ml_run_00.ipynb
  -> ml_01_ml_io.py
  -> ml_01_ml_train.py
  -> ml_01_ml_val.py
  -> ml_01_ml_test.py

실험 산출물
  -> ml/<EXPERIMENT_ID>/ml_outputs/<RUN_ID>/
```

현재 학습/평가 노트북은 feature 생성이나 `fb_outputs -> ml_inputs` 복사를 수행하지 않는다. `ml_inputs/<RUN_ID>/`에 이미 승인 배치된 입력 묶음만 읽는다. 아래 입력/출력 계약은 현재 구현된 `ML-01` 기준 예시이며, 후속 실험도 같은 원칙을 따른다.

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

`encoding_manifest.json`은 native categorical dtype 복원과 검증에 사용한다. manifest를 사용하는 run에서는 train, validation, 대표 후보 test가 같은 manifest를 참조해야 한다.

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

`model.pkl`, `feature_columns.json`, `train_summary.json`, `feature_importance.csv`는 train 단계 산출물이다. `threshold.json`, `metrics_val.json`, `confusion_matrix_val.csv`는 validation 단계 산출물이다. `metrics_test.json`, `confusion_matrix_test.csv`는 대표 후보 test 단계 산출물이며, 최종 설정 확정 전에는 생성하지 않는다.

같은 입력으로 재실행할 때는 기존 파일을 덮어쓰기보다 `MODEL_RUN_ID`를 새로 부여하는 방식을 우선한다. `OVERWRITE_OUTPUTS=False`가 기본값이다.

## 코드 아키텍처

| 파일 | 역할 |
|---|---|
| `00_ml_01_ml_run_00.ipynb` | ML-01 train/validation/final-test orchestration. 경로, run ID, XGBoost 파라미터, 실행 스위치를 설정 |
| `ml_01_ml_io.py` | 경로 해석, 입력 파일 검증, 승인 feature contract 파싱, forbidden feature 차단, parquet schema/split/label 검증, feature hash, JSON 저장/로드 |
| `ml_01_ml_preview.py` | 승인 feature와 split summary, parquet schema를 학습 전 미리 확인하는 preview helper. 모델 학습과 artifact 저장은 하지 않음 |
| `ml_01_ml_train.py` | train/val split 로드, `scale_pos_weight` 계산, XGBoost 학습, native categorical 적용, train artifact 저장 |
| `ml_01_ml_val.py` | 저장된 모델과 feature 순서로 validation probability 계산, threshold 선택, validation metric 저장 |
| `ml_01_ml_test.py` | 대표 후보 test 전용. `confirm_final_test=True`, full artifact, provenance 검증 없이는 실행 차단 |
| `ml_01_ml_metrics.py` | F1 기준 threshold 선택, F1/Recall/Precision/AP/confusion matrix 계산 |
| `ml_01_ml_resource.py` | runtime, CPU memory, 환경 정보, data profile, score profile, feature importance 등 진단 metadata 수집 |
| `ml_01_ml_tune.py` | XGBoost random search 실행. test는 수행하지 않음 |
| `ml_01_ml_search_spaces.py` | tuning search space preset 정의 |
| `ml_01_ml_utils.py` | seed 고정 helper |

노트북은 실행 순서와 설정을 관리하고, 재사용 로직은 `ml_01_ml_*.py` 모듈에 둔다. 긴 학습/검증 로직을 노트북에 누적하지 않는 것이 현재 구조의 기준이다.

## 실행 모드

| 모드 | 설정 | 해석 |
|---|---|---|
| Sample/debug | `SAMPLE_ROWS`에 양의 정수 지정, `RUN_FINAL_TEST=False` | 경로, 입력 계약, 학습 호출 확인용. 성능 주장 금지 |
| Full train/validation | `SAMPLE_ROWS=None`, `RUN_TRAIN_AND_VALIDATION=True`, `RUN_FINAL_TEST=False` | 모델과 validation threshold 선택용. validation metric은 내부 선택 지표 |
| Representative test | `SAMPLE_ROWS=None`, `RUN_TRAIN_AND_VALIDATION=False` 또는 필요 시 유지, `RUN_FINAL_TEST=True` | `ML-02 tuned`, `ML-05 tuned`, `reduced/final tuned` 대표 후보에서만 평가 |

test는 기본 잠금 상태다. `ml_01_ml_test.py`는 sampled model, sampled threshold, sampled test를 차단하고, `threshold.json`의 validation threshold만 사용한다. 모든 Stage에서 test를 반복하지 않는다.

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
| test 평가 | `ML-02 tuned`, `ML-05 tuned`, `reduced/final tuned` 대표 후보에 한정해 실행 |
| seed | 기본 `42`, 노트북 시작부에서 `ml_01_ml_utils.set_seed(SEED)` 호출 |

## 현재 주의사항

| 항목 | 내용 |
|---|---|
| `ML-00` 기준 | `ML-00`은 baseline freeze로 해석한다. 후속 실험 비교표의 기준 성능이다. |
| `ml-00_baseline-freeze/` | 완료된 기준선/feature build 이력 영역이다. 신규 수정, 재학습, 산출물 재생성은 사용자 승인 후에만 수행한다. |
| artifact 상태 | 로컬에 parquet/model/result 파일이 존재할 수 있으나 README는 파일 존재를 성능 근거로 해석하지 않는다. |

## 문서 갱신 기준

다음이 바뀌면 이 README를 함께 갱신한다.

| 변경 | 갱신할 내용 |
|---|---|
| 신규 실험 폴더 추가 | 실행 진입점, 입력/출력 prefix, run ID 정책 |
| feature 승인 양식 변경 | 승인 CSV 파일명, 필수 컬럼, `used_in_ml` 정책 |
| artifact 파일명 변경 | 출력 계약과 test provenance 기준 |
| train/validation/test 역할 변경 | 누수 방지 원칙과 실행 모드 |
| 자동화 테스트 추가 | 검증 명령과 실패 처리 기준 |
