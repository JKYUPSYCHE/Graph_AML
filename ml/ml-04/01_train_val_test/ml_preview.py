"""
1. 코드 전체 요약
- 이 코드는 노트북에서 ML 학습 입력값을 실행 전에 검증하고, 사람이 보기 좋은 미리보기 테이블로 정리하는 헬퍼 모듈이다.
- 핵심 진입점은 preview_ml_inputs()이며, feature 목록 CSV, split summary CSV, train/val/test parquet schema를 검증한 뒤 MLInputPreview 객체로 반환한다.
- 모델 학습, AML 룰 적용, threshold tuning, 성능 평가, 결과 파일 저장은 이 코드에 없다.

2. 데이터 흐름 요약
- 입력: ml_io.InputPaths, split_summary_path, feature_columns_path
- Feature CSV: check_feature_columns_file() 검증 → used_in_ml 파싱 → 선택 feature 목록 확정
- Split summary CSV: split/row/label count/positive rate 검증 → parquet metadata row count와 비교
- Parquet schema: train/val/test parquet에 feature column과 필수 non-model column 존재 여부 확인
- 출력: 검증된 feature 목록, hash, 표시용 DataFrame, split summary, schema summary를 담은 MLInputPreview

3. 변경 시 주의점
- feature_columns의 순서가 바뀌면 feature_columns_hash도 바뀌므로 실험 재현성에 영향이 있다.
- label_col 기본값 "label"을 바꾸면 feature 검증과 parquet 필수 컬럼 검증 모두 영향을 받는다.
- show_test_label_distribution=False가 기본값이므로 test label 분포는 노트북 표시에서 숨겨진다.
- validate_split_summary()는 parquet 전체를 로드하지 않고 metadata row count만 확인한다.
- 이 모듈은 파일을 저장하지 않는다. 결과 저장은 호출하는 노트북 또는 다른 파이프라인에서 처리하는 것으로 보인다. 확인 필요.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ML 입력 경로 해석, feature CSV 검증, parquet schema 확인 등 공통 입출력 검증 규칙은 ml_io 모듈에 위임한다.
# 확인 필요: ml_io 내부에서 project_root, feature mask, label_col을 어떤 기준으로 검증하는지는 이 코드만으로는 확정할 수 없다.
import ml_io

# 노트북에서 선택 feature 목록을 보여줄 때 사용할 컬럼 순서다.
# 실제 모델 입력 컬럼을 정의하는 값은 아니고, display용 컬럼 필터 역할을 한다.
# FEATURE_DISPLAY_COLUMNS에 없는 컬럼은 selected_features 원본에는 남을 수 있지만 selected_features_display에서는 제외된다.
FEATURE_DISPLAY_COLUMNS = [
    "feature_order",
    "column_name",
    "used_in_ml",
    "feature_group",
    "dtype",
    "leakage_risk",
    "selection_note",
    "target_experiment",
    "used_in_experiments",
    "excluded_reason",
]


@dataclass(frozen=True)
class MLInputPreview:
    """Validated preview payload for the train/validation/test notebook."""

    feature_columns: list[str]               # 모델에 실제 투입될 feature column 이름 목록이다.
    feature_columns_hash: str                # feature_columns 순서를 포함해 계산된 식별값이다.
    selected_features: pd.DataFrame          # used_in_ml 기준으로 선택된 feature 행 전체다.
    selected_features_display: pd.DataFrame  # 노트북에 간결하게 보여주기 위해 FEATURE_DISPLAY_COLUMNS 기준으로 줄인 표다.
    split_summary: pd.DataFrame              # split_summary CSV를 검증한 원본 요약 테이블이다. train/val/test별 rows, label count, positive_rate가 포함된다.
    split_summary_display: pd.DataFrame      # 노트북 표시용 split summary다. 기본 설정에서는 test label 분포가 "[hidden]"으로 가려진다.
    schema_summary: pd.DataFrame             # train/val/test parquet schema 검증 결과다.  split별 column 수, optional trace column 누락 여부, parquet path를 담는다.
    required_non_model_columns: list[str]    # 모델 feature는 아니지만 ML 입력 데이터에 반드시 있어야 하는 추적/라벨 컬럼이다.
    optional_trace_columns: list[str]        # 없어도 학습 입력 검증은 통과하지만, 추적과 해석에 유용한 선택 컬럼이다.


def _resolve_file(
    path: str | Path,
    label: str,
    project_root: str | Path | None = None,
) -> Path:
    """
    Resolve a required file path with the shared project-root policy.
    """
    # 경로 해석 정책을 이 함수에서 직접 구현하지 않고 ml_io에 위임한다.
    resolved = ml_io.resolve_project_path(path, project_root)

    # 이후 pd.read_csv 또는 ParquetFile 호출 전에 파일 존재 여부를 확인힌다.
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")

    # 디렉터리를 파일로 잘못 넘긴 경우도 별도 메시지로 막는다. 입력 계약이 깨진 상태에서 노트북이 계속 진행되는 것을 방지한다.
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def parquet_row_count(path: str | Path) -> int:
    """Return parquet row count from metadata without loading the table."""

    # parquet row 수만 확인, AML 데이터는 대용량일 수 있어 metadata 기반 검증이 중요하다.
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        # pyarrow가 없으면 parquet metadata를 읽을 수 없으므로 명시적으로 실패시킨다.
        raise ImportError(
            "pyarrow is required to inspect parquet metadata. "
            "Install pyarrow in the environment used by this notebook."
        ) from exc

    # 입력 parquet path도 공통 파일 검증 함수를 거친다.
    parquet_path = _resolve_file(path, "split_parquet")

    # ParquetFile(...).metadata.num_rows는 실제 컬럼 데이터를 로드하지 않는다.
    # split_summary의 rows 값과 parquet 실제 row 수를 비교하는 데 사용된다.
    return int(pq.ParquetFile(parquet_path).metadata.num_rows)


def load_selected_feature_rows(
    feature_columns_path: str | Path,
    *,
    project_root: str | Path | None = None,
    label_col: str = "label",
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Load selected feature rows for display after standard feature validation."""
    # feature CSV의 기본 계약 검증은 ml_io.check_feature_columns_file에 위임한다.
    feature_check = ml_io.check_feature_columns_file(
        feature_columns_path,
        project_root=project_root,
        label_col=label_col,
        strict=True,   # strict=True이므로 학습 입력으로 승인되지 않은 상태나 schema 오류를 강하게 막는 의도로 보인다.
    )

    # feature CSV를 다시 읽어 노트북 표시용 DataFrame을 만든다.
    # used_in_ml을 string dtype으로 읽어서 같은 mask  parse_used_in_ml에서 일관되게 처리하도록 한다.
    feature_table = pd.read_csv(
        feature_check.path,
        encoding="utf-8-sig",
        dtype={"used_in_ml": "string"},
    )

    # used_in_ml 컬럼을 실제 boolean mask로 변환한다.
    selected_mask = ml_io.parse_used_in_ml(feature_table["used_in_ml"])

    # 모델에 사용하도록 선택된 used_in_ml = True feature 행만 남긴다.
    # copy()를 사용해 이후 column_name strip, feature_order 재삽입이 원본 view 경고 없이 수행된다.
    selected_features = feature_table.loc[selected_mask].copy()

    # column_name 앞뒤 공백을 제거한다.
    selected_features["column_name"] = (
        selected_features["column_name"].astype(str).str.strip()
    )

    # 화면에 표시할 feature 순서와 ml_io가 검증한 selected_columns 순서가 같은지 확인한다.
    # 이 검증이 실패하면 노트북 표시와 실제 학습 입력 기준이 달라질 수 있으므로 즉시 중단한다.
    displayed_columns = selected_features["column_name"].tolist()
    if displayed_columns != feature_check.selected_columns:
        raise ValueError(
            "Displayed feature columns do not match ml_io selected columns. "
            "Check the feature mask CSV before training."
        )

    # 기존 feature_order가 CSV에 있더라도 선택된 feature만 기준으로 1부터 다시 부여한다.
    # 이 값은 display 순서 추적용이며, 모델 입력 컬럼 자체는 feature_check.selected_columns가 기준이다.
    selected_features = selected_features.drop(columns=["feature_order"], errors="ignore")
    selected_features.insert(0, "feature_order", range(1, len(selected_features) + 1))

    # 표시용 DataFrame에는 FEATURE_DISPLAY_COLUMNS 중 실제 CSV에 존재하는 컬럼만 포함한다.
    # 따라서 feature catalog에 일부 메타 컬럼이 없어도 display 생성은 실패하지 않는다.
    display_columns = [
        column for column in FEATURE_DISPLAY_COLUMNS if column in selected_features.columns
    ]

    return (
        feature_check.selected_columns,            # 반환값 1: 모델 입력 feature column 목록
        selected_features,                         # 반환값 2: 선택 feature 행 전체
        selected_features[display_columns].copy(), # 반환값 3: 노트북 표시용으로 컬럼을 제한한 선택 feature 표
    )


def validate_split_summary(
    split_summary_path: str | Path,
    paths: ml_io.InputPaths,
    *,
    project_root: str | Path | None = None,
    show_test_label_distribution: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate split summary rows and return raw/display versions."""

    # split summary CSV 경로를 해석하고 실제 파일인지 확인한다.
    # 이 파일은 train/val/test의 row 수와 label 분포가 기록된 입력 계약 문서 역할을 한다.
    summary_path = _resolve_file(
        split_summary_path,
        "ML_INPUT_SPLIT_SUMMARY_PATH",
        project_root,
    )
    split_summary = pd.read_csv(summary_path, encoding="utf-8-sig")


    # split summary가 최소한 갖춰야 하는 컬럼이다. positive_rate는 label_1_count / rows와 일치해야 한다.
    required_columns = {"split", "rows", "label_0_count", "label_1_count", "positive_rate"}
    missing_columns = required_columns - set(split_summary.columns)
    if missing_columns:
        raise ValueError(f"split_summary CSV is missing columns: {sorted(missing_columns)}")

    # split 값은 공백 제거 후 소문자로 정규화한다.
    split_summary = split_summary.copy()
    split_summary["split"] = split_summary["split"].astype("string").str.strip().str.lower()

    # 이 검증 함수는 train/val/test 세 split이 정확히 모두 있어야 한다.
    # 누락 split이나 예상 외 split 이름이 있으면 이후 학습/검증/최종평가 경계가 깨질 수 있다.
    expected_splits = {"train", "val", "test"}
    observed_splits = set(split_summary["split"].tolist())
    if observed_splits != expected_splits:
        raise ValueError(
            f"split_summary split mismatch. expected={sorted(expected_splits)}, "
            f"observed={sorted(observed_splits)}"
        )

    # split별 row는 하나씩만 있어야 한다. 중복 split이 있으면 set 비교만으로는 잡히지 않을 수 있어 별도로 검사한다.
    if split_summary["split"].duplicated().any():
        duplicated = split_summary.loc[split_summary["split"].duplicated(), "split"].tolist()
        raise ValueError(f"split_summary contains duplicated split rows: {duplicated}")

    # row 수, label count, positive_rate를 숫자로 강제 변환한다. 숫자가 아닌 값이 들어오면 즉시 실패한다.
    numeric_columns = ["rows", "label_0_count", "label_1_count", "positive_rate"]
    for column in numeric_columns:
        split_summary[column] = pd.to_numeric(split_summary[column], errors="raise")

    # row 수와 label count는 음수가 될 수 없으므로 입력 오류를 막는다.
    count_columns = ["rows", "label_0_count", "label_1_count"]
    if (split_summary[count_columns] < 0).any().any():
        raise ValueError("split_summary contains negative row/label counts.")

    # rows 값이 label_0_count + label_1_count와 일치하는지 확인한다.
    # 불일치하면 라벨 분포 요약과 실제 split 크기 중 하나가 잘못된 것이다.
    row_sum = split_summary["label_0_count"] + split_summary["label_1_count"]
    mismatch = split_summary[row_sum != split_summary["rows"]]
    if not mismatch.empty:
        raise ValueError(
            "split_summary row count mismatch: rows != label_0_count + label_1_count. "
            f"mismatch={mismatch.to_dict(orient='records')}"
        )

    # positive_rate가 label_1_count / rows와 일치하는지 검증한다.
    # tolerance 1e-12는 CSV 저장/부동소수점 표현 차이를 아주 작게 허용하는 값이다.
    expected_rate = split_summary["label_1_count"] / split_summary["rows"]
    rate_diff = (split_summary["positive_rate"] - expected_rate).abs()
    if (rate_diff > 1e-12).any():
        mismatch = split_summary.loc[
            rate_diff > 1e-12,
            ["split", "positive_rate", "label_1_count", "rows"],
        ]
        raise ValueError(
            "split_summary positive_rate mismatch. "
            f"mismatch={mismatch.to_dict(orient='records')}"
        )

    # InputPaths에 담긴 train/val/test parquet 경로와 split_summary row 수를 연결한다.
    # 여기서는 label 분포까지 parquet에서 다시 계산하지 않고 row count만 metadata로 비교한다.
    split_path_map = {
        "train": paths.train_path,
        "val": paths.val_path,
        "test": paths.test_path,
    }
    summary_rows = split_summary.set_index("split")["rows"].astype(int).to_dict()

    # split_summary의 rows와 실제 parquet metadata row 수가 같은지 확인한다.
    # 잘못된 split_summary 파일, 오래된 parquet, 경로 착오를 점검하는 목적이 있다.
    for split_name, split_path in split_path_map.items():
        if split_path is None:
            raise ValueError(f"{split_name} parquet path is missing.")
        parquet_rows = parquet_row_count(split_path)
        if parquet_rows != summary_rows[split_name]:
            raise ValueError(
                f"{split_name} row count mismatch between parquet and split_summary. "
                f"parquet_rows={parquet_rows}, split_summary_rows={summary_rows[split_name]}, "
                f"path={split_path}"
            )

    # raw summary는 그대로 보존하고, display summary는 노트북 표시 정책에 맞게 가공한다.
    display_summary = split_summary.copy()

    # test label 분포는 기본적으로 숨긴다. 최종 평가셋 정보를 노트북에서 불필요하게 노출하지 않기 위한 안전장치 이다.
    # show_test_label_distribution=True일 때만 test label count와 positive_rate가 표시된다.
    if not show_test_label_distribution:
        hidden_columns = ["label_0_count", "label_1_count", "positive_rate"]
        display_summary[hidden_columns] = display_summary[hidden_columns].astype("object")
        display_summary.loc[display_summary["split"] == "test", hidden_columns] = "[hidden]"

    # 반환값 1: 검증된 원본 split summary
    # 반환값 2: 노트북 표시용 split summary
    return split_summary, display_summary


def validate_split_schemas(
    paths: ml_io.InputPaths,
    feature_columns: list[str],
    *,
    label_col: str = "label",
    required_non_model_columns: set[str] | None = None,
    optional_trace_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Validate required columns in train/val/test parquet schemas."""
    # 모델 feature는 아니지만 학습/검증/추적에 필요한 기본 컬럼이다. label_col을 바꾸면 이 필수 컬럼 목록에도 반영된다.
    if required_non_model_columns is None:
        required_non_model_columns = {"tx_id", "timestamp", "split", label_col}
    else:
        required_non_model_columns = set(required_non_model_columns)

    # sender/receiver는 없어도 실패시키지 않는 trace용 컬럼이다. 계좌 흐름 해석이나 디버깅에는 유용하지만, 필수 모델 입력은 아닌 것으로 처리한다.
    if optional_trace_columns is None:
        optional_trace_columns = {"sender_account", "receiver_account"}
    else:
        optional_trace_columns = set(optional_trace_columns)

    # parquet에 반드시 존재해야 하는 컬럼은 선택 feature와 필수 non-model 컬럼의 합집합이다.
    # feature_columns 중 하나라도 parquet에 없으면 학습 시점에 실패하므로 여기서 먼저 막는다.
    required_columns = set(feature_columns) | required_non_model_columns

    # train/val/test 각각의 parquet schema를 검사한다.
    split_paths = {
        "train": paths.train_path,
        "val": paths.val_path,
        "test": paths.test_path,
    }

    rows: list[dict[str, object]] = []

    for split_name, split_path in split_paths.items():
        # 이 함수만 단독 호출하면 None split path는 건너뛴다.
        # 단, preview_ml_inputs()에서는 require_input_files(require_test=True)를 먼저 호출하므로 일반 경로에서는 train/val/test가 모두 존재해야 한다.
        if split_path is None:
            continue
        schema_names = set(ml_io.get_parquet_columns(split_path)) # parquet 컬럼명만 읽어 schema 검증을 수행한다.

        # 선택 feature 또는 필수 non-model 컬럼이 하나라도 없으면 학습 입력 계약 위반이다.
        # missing 목록은 최대 30개만 메시지에 보여주고 전체 개수도 함께 제공한다.
        missing_required = sorted(required_columns - schema_names)
        if missing_required:
            raise ValueError(
                f"{split_name} parquet is missing required ML input columns. "
                f"missing={missing_required[:30]}, "
                f"missing_count={len(missing_required)}, path={split_path}"
            )

        # schema_summary에 split별 검증 결과를 누적한다. missing_optional은 실패 조건이 아니라 추적 컬럼 누락 여부를 알려주는 정보다.
        rows.append(
            {
                "split": split_name,
                "column_count": len(schema_names),
                "missing_optional": sorted(optional_trace_columns - schema_names),
                "path": str(split_path),
            }
        )
    # 호출자는 이 DataFrame을 노트북에서 표시해 각 split의 schema 상태를 확인할 수 있다.
    return pd.DataFrame(rows)


def preview_ml_inputs(
    paths: ml_io.InputPaths,
    *,
    split_summary_path: str | Path,
    project_root: str | Path | None = None,
    label_col: str = "label",
    show_test_label_distribution: bool = False,
) -> MLInputPreview:
    """Validate approved ML inputs and prepare compact notebook display payloads."""
    # 이 함수가 모듈의 핵심 진입점이다.
    # 모델 학습 전에 feature CSV, split summary, parquet schema가 서로 맞는지 한 번에 검증한다.
    # train/val/test 및 feature_columns_path 같은 필수 입력 파일 존재 여부를 먼저 검사한다.
    # require_test=True이므로 test parquet도 필수로 요구한다.
    # 확인 필요: InputPaths가 정확히 어떤 필드를 가지는지는 ml_io.InputPaths 정의 확인이 필요하다.
    ml_io.require_input_files(paths, require_test=True)

    # feature catalog에서 used_in_ml로 선택된 feature 목록과 display용 표를 만든다.
    # 여기서 반환된 feature_columns가 이후 schema 검증과 hash 계산의 기준이 된다.
    feature_columns, selected_features, selected_features_display = load_selected_feature_rows(
        paths.feature_columns_path,
        project_root=project_root,
        label_col=label_col,
    )

    # split summary CSV의 row/label/rate 계약을 검증하고,
    # parquet metadata row count와 summary rows가 일치하는지 확인한다.
    split_summary, split_summary_display = validate_split_summary(
        split_summary_path,
        paths,
        project_root=project_root,
        show_test_label_distribution=show_test_label_distribution,
    )
    # train/val/test parquet가 선택 feature와 필수 추적/라벨 컬럼을 모두 갖고 있는지 확인한다.
    schema_summary = validate_split_schemas(paths, feature_columns, label_col=label_col)

    # 이 함수는 파일 저장이나 모델 학습을 하지 않는다.
    # 노트북에서 바로 표시하거나 이후 학습 코드에 넘길 수 있는 검증 결과 객체만 반환한다.
    return MLInputPreview(
        feature_columns=feature_columns,
        feature_columns_hash=ml_io.feature_columns_hash(feature_columns),
        selected_features=selected_features,
        selected_features_display=selected_features_display,
        split_summary=split_summary,
        split_summary_display=split_summary_display,
        schema_summary=schema_summary,
        required_non_model_columns=sorted({"tx_id", "timestamp", "split", label_col}),
        optional_trace_columns=sorted({"sender_account", "receiver_account"}),
    )
