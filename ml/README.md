# ML 파트 README

`ml/`은 전처리와 feature build 단계에서 생성된 ML-ready parquet를 입력으로 받아 XGBoost 학습, validation threshold 선택, 대표 후보 test 평가를 수행하는 작업 영역이다.

현재 문서의 목적은 **PART 2 Graph Feature + ML 운영 기준과 실험 현황을 한눈에 확인**하는 것이다. 아래 성능 수치는 로컬 validation artifact 기준이며, final test 성능이 아니다.

## 현재 기준 요약

| 항목 | 현재 기준 |
|---|---|
| `ML-00` | 현재 거래 row 기반 baseline freeze. 후속 실험 비교 기준 |
| `ML-01~ML-02` | aggregate/history 계열. validation 기준 대표 후보는 현재 `ML-02` |
| `ML-03~ML-05` | graph feature 누적 실험. 현재 fixed-param validation에서는 성능 개선으로 채택 금지 |
| `ML-06~ML-08` | reduced, ablation, final tuned 후보. 현재 후속 작업 영역 |
| 현재 모델 | XGBoost binary classifier |
| feature 생성 | train/validation/test 노트북에서 수행하지 않음 |
| feature 승인 | `fb_outputs` 검토 후 `ml_inputs/<RUN_ID>/`에 승인 배치된 파일만 학습 입력으로 사용 |
| feature 선택 기준 | 승인 CSV에서 `used_in_ml="TRUE"`인 컬럼 |
| threshold 선택 | validation set에서만 수행 |
| test 평가 | 기본 잠금. 대표 후보에서만 사용자 승인 후 실행 |
| commit 제외 | parquet, model, output artifact |

## 실험 매트릭스

| 실험 ID | 피처 구성 | 현재 상태 | 해석 | 다음 액션 |
|---|---|---|---|---|
| `ML-00` | 현재 거래 row 기반 baseline | baseline freeze | 후속 실험 비교 기준 | 신규 수정/재학습은 사용자 승인 후 수행 |
| `ML-01` | `ML-00` + Stage 0 시간 이력 피처 | validation 완료 | recall은 높지만 precision/AP 한계 | `ML-02`와 비교 기준으로 유지 |
| `ML-02` | `ML-01` + Stage 1 계좌별 통계 피처 | validation 기준 대표 후보 | 현재 확인된 validation 지표가 가장 좋음 | 대표 후보 tuning/test 여부를 별도 결정 |
| `ML-03` | `ML-02` + Stage 2 fan-in/fan-out | validation 완료 | graph feature 시작점이나 현 설정에서는 `ML-02`보다 낮음 | feature 중복/품질 점검 |
| `ML-04` | `ML-03` + Stage 3 sender-receiver 관계 및 bank-pair corridor 반복성 | 주의/원인분리 대상 | pair feature와 AUPRC early stopping 불안정. 개선 사례로 채택 금지 | pair feature ablation, 보수적 XGBoost 설정 비교 |
| `ML-05` | `ML-04` + Stage 4 pass-through/flow-balance | 주의/원인분리 대상 | feature 수와 비용은 증가했지만 validation 성능은 하락 | `ML-04` 원인분리 후 채택 여부 재검토 |
| `ML-06` | Stage 0~4 품질/누수/중복/중요도 기반 축소 피처셋 | scaffold/후속 작업 | reduced 후보 | selected feature list 구성 |
| `ML-07` | 최소 ablation 기반 practical/final candidate | scaffold/후속 작업 | ablation 후보 | full/practical/light 후보 비교 |
| `ML-08` | 최종 피처셋 + tuning | scaffold/후속 작업 | final tuned 후보 | validation 기준 확정 후 test 대상 |

## 현재 Validation 현황

주의: 아래 표는 validation artifact 기준이다. final test 성능, 배포 성능, 일반화 성능으로 단정하지 않는다.

| 실험 | run | feature count | F1 | Recall | Precision | AP/AUPRC | 현재 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| `ML-01` | `r02 / d00-fixparam` | 73 | 0.2284 | 0.4294 | 0.1556 | 0.1362 | recall은 높지만 false positive 부담 큼 |
| `ML-02` | `r01 / d00-optuna` | 101 | 0.3965 | 0.3795 | 0.4152 | 0.3465 | 현재 validation 기준 대표 후보 |
| `ML-03` | `r00 / d00-fixparam` | 146 | 0.1770 | 0.2475 | 0.1377 | 0.0899 | graph feature 추가 후 현 설정에서는 하락 |
| `ML-04` | `r01 / d00-fixparam` | 188 | 0.0858 | 0.1939 | 0.0551 | 0.0577 | pair feature/early stopping 원인분리 필요 |
| `ML-05` | `r00 / d00-fixparam` | 236 | 0.0838 | 0.1911 | 0.0537 | 0.0594 | feature 수 증가 대비 성능 하락 |

수치 출처는 각 실험의 `ml_outputs/<RUN_ID>/*_metrics_val.json`이다. `ML-02`는 계획상 대표 후보로 쓰는 `ml/ml-02/ml_outputs/r01/ml_02__r01__d00-optuna_metrics_val.json` 기준이다.

## 디렉터리 구조

```text
ml/
  README.md
  ml_leaderboard_representatives.json
  260521_feature_catalog.csv

  ml-00_baseline-freeze/
    feature_build/
    train_val_test/
    # ML-00 baseline freeze 영역. 기본적으로 읽기 전용으로 취급

  ml-01/
  ml-02/
  ml-03/
  ml-04/
  ml-05/
    fb_inputs/
    fb_outputs/
    ml_inputs/
    ml_outputs/
    00_feature_build/
    01_train_val_test/

  ml-06/
  ml-07/
  ml-08/
    # reduced, ablation, final tuned 후보 실험

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
  -> ml/<EXPERIMENT_ID>/01_train_val_test/

실험 산출물
  -> ml/<EXPERIMENT_ID>/ml_outputs/<RUN_ID>/
```

현재 학습/평가 노트북은 feature 생성이나 `fb_outputs -> ml_inputs` 복사를 수행하지 않는다. `ml_inputs/<RUN_ID>/`에 이미 승인 배치된 입력 묶음만 읽는다.

## 입력 계약

아래는 현재 구현 기준 예시다. 후속 실험도 같은 원칙을 따른다.

| 변수 | 의미 | 예시 |
|---|---|---|
| `EXPORT_EXPERIMENT_ID` | 실험 ID | `ml_01` |
| `RUN_ID` | 승인 입력 묶음 ID | `r00` |
| `MODEL_RUN_ID` | 같은 입력으로 반복하는 모델 run ID | `d00` |
| `ARTIFACT_PREFIX` | 입력 파일 prefix | `ml_01__r00` |
| `ML_ARTIFACT_PREFIX` | 모델 산출물 prefix | `ml_01__r00__d00` |

입력 파일 양식은 아래와 같다.

```text
ml/<EXPERIMENT_ID>/ml_inputs/<RUN_ID>/
  <ARTIFACT_PREFIX>_Xy_train.parquet
  <ARTIFACT_PREFIX>_Xy_val.parquet
  <ARTIFACT_PREFIX>_Xy_test.parquet
  <ARTIFACT_PREFIX>_fb_output_feature_contract_approve.csv
  <ARTIFACT_PREFIX>_encoding_manifest.json
  <ARTIFACT_PREFIX>_split_summary.csv
```

핵심 기준은 승인 CSV다. 모델 입력 feature는 `<ARTIFACT_PREFIX>_fb_output_feature_contract_approve.csv`에서 `used_in_ml="TRUE"`인 `column_name`만 의미한다. `fb_outputs`의 후보 contract, manifest, feature info는 승인 전 상태일 수 있으므로 실제 학습 feature 판단은 승인 CSV, `feature_columns.json`, `train_summary.json`의 feature list와 hash를 기준으로 한다.

`encoding_manifest.json`은 native categorical dtype 복원과 검증에 사용한다. manifest를 사용하는 run에서는 train, validation, 대표 후보 test가 같은 manifest를 참조해야 한다.

## 출력 계약

현재 출력 위치는 아래 형태다.

```text
ml/<EXPERIMENT_ID>/ml_outputs/<RUN_ID>/
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

## 실행 모드

| 모드 | 설정 | 해석 |
|---|---|---|
| Sample/debug | `SAMPLE_ROWS`에 양의 정수 지정, `RUN_FINAL_TEST=False` | 경로, 입력 계약, 학습 호출 확인용. 성능 주장 금지 |
| Full train/validation | `SAMPLE_ROWS=None`, `RUN_TRAIN_AND_VALIDATION=True`, `RUN_FINAL_TEST=False` | 모델과 validation threshold 선택용. validation metric은 내부 선택 지표 |
| Representative test | `SAMPLE_ROWS=None`, `RUN_FINAL_TEST=True` | `ML-02 tuned`, `ML-05 tuned`, `reduced/final tuned` 대표 후보에서만 사용자 승인 후 평가 |

test는 기본 잠금 상태다. validation threshold가 확정된 대표 후보에서만 실행하며, test set에서 threshold나 feature를 다시 선택하지 않는다.

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
| test 평가 | 대표 후보에서만 사용자 승인 후 실행 |
| seed | 기본 `42`, 노트북 시작부에서 seed 고정 helper 호출 |
| commit 제외 | parquet, model, prediction score, metrics artifact |

## 현재 리스크와 다음 작업

| 항목 | 현재 리스크 | 다음 작업 |
|---|---|---|
| `ML-04` | pair-history 피처가 초반 tree를 강하게 지배했고, AUPRC 기준 early stopping이 불안정했다. | pair feature ablation, 보수적 XGBoost 설정, 러닝 커브 재확인 |
| `ML-05` | Stage 4 pass-through/flow-balance 추가 후 feature count는 236개로 증가했지만 validation 성능은 낮다. | `ML-04` 원인분리 후 Stage 4 피처 채택 여부 재검토 |
| `ML-06` | reduced feature set이 아직 확정되지 않았다. | 누수, 품질, 중복, 중요도 기준 selected feature list 구성 |
| `ML-07` | practical/final candidate ablation이 아직 확정되지 않았다. | full/practical/light 후보 비교 |
| `ML-08` | final tuned 후보와 test 대상이 아직 확정되지 않았다. | validation 기준 최종 설정 확정 후 사용자 승인 하에 test 1회 실행 |

관련 분석 문서:

| 문서 | 내용 |
|---|---|
| `docs/ml04_learning_curve_diagnosis_2026-05-28.md` | `ML-04` AUPRC early stopping 및 pair feature 지배 현상 분석 |
| `docs/ml04_feature_outlier_review_2026-05-28.md` | `ML-04` 학습 피처 결측, 무한대, 이상치, ratio 안정성 검토 |
| `docs/ml_feature_catalog_by_expid_summary.md` | exp-id별 feature catalog 요약 |

## 문서 갱신 기준

다음이 바뀌면 이 README를 함께 갱신한다.

| 변경 | 갱신할 내용 |
|---|---|
| 신규 실험 폴더 추가 | 실행 진입점, 입력/출력 prefix, run ID 정책 |
| validation 대표 후보 변경 | 현재 validation 현황 표와 실험 매트릭스 |
| feature 승인 양식 변경 | 승인 CSV 파일명, 필수 컬럼, `used_in_ml` 정책 |
| artifact 파일명 변경 | 출력 계약과 test provenance 기준 |
| train/validation/test 역할 변경 | 누수 방지 원칙과 실행 모드 |
| final test 실행 | validation 표와 구분되는 별도 test 결과 섹션 |
| 자동화 테스트 추가 | 검증 명령과 실패 처리 기준 |
