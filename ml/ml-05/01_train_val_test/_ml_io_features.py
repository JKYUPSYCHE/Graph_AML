"""Feature catalog, leakage guard, and encoding manifest helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

FORBIDDEN_EXACT_NAMES = {"label", "target", "y", "is_laundering"}
FORBIDDEN_SUBSTRINGS = {"laundering", "pattern", "typology", "attempt"}
UNKNOWN_CATEGORY = "__UNKNOWN__"


def _resolve_feature_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve feature-catalog related paths without importing _ml_io_inputs."""

    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved.resolve()
    if project_root is None:
        raise ValueError(
            "Relative paths require project_root. "
            "Pass PROJECT_ROOT from Notebook Bootstrap or use an absolute path."
        )
    return (Path(project_root).expanduser().resolve() / resolved).resolve()

@dataclass(frozen=True)
class FeatureColumnsCheckResult:
    """ml_feature_columns.csv 검증 또는 정규화 처리 결과."""
    ok: bool                     # ok: 검증/정규화가 성공했는지 여부.
    processed: bool              # processed: 실제 파일 검증 또는 정규화 처리가 수행되었는지 여부.
    path: Path                   # path: 검증하거나 정규화한 ml_feature_columns.csv 경로.
    total_rows: int              # total_rows: CSV 전체 row 수. feature 후보 전체 개수를 추적할 때 사용된다.
    selected_count: int          # selected_count: used_in_ml="TRUE"로 선택된 feature 개수.
    selected_columns: list[str]  # selected_columns: 모델 입력으로 사용할 최종 feature 컬럼명 목록.
    # error_type/error_message: strict=False일 때 예외를 직접 raise하지 않고 결과 객체에 실패 원인을 담기 위한 필드.
    error_type: str | None = None
    error_message: str | None = None


def parse_used_in_ml(series: pd.Series) -> pd.Series:
    """
    used_in_ml 컬럼을 strict boolean mask로 변환한다.
    사용 목적
    --------
    - ML 학습에 사용할 feature row만 선택하기 위한 boolean mask를 만든다.
    - used_in_ml 값은 CSV에서 문자값 "TRUE" 또는 "FALSE"로 고정한다.
    - "TRUE"인 row의 column_name만 모델 입력 feature로 사용한다.
    """
    # used_in_ml이 비어 있으면 즉시 실패시킨다.
    if series.isna().any():
        missing_rows = (series[series.isna()].index + 2).tolist()   # index + 2: pandas index - 실제 CSV row 번호와 맞추기 위한 보정이다.
        raise ValueError(f"used_in_ml contains missing values. csv_rows={missing_rows[:30]}")
    allowed_values = {"TRUE", "FALSE"}   # 학습 feature 선택 정책을 "TRUE"/"FALSE" 두 문자열로 고정한다.
    as_text = series.astype(str)
    # 허용되지 않은 값이 하나라도 있으면 중단한다.
    invalid = as_text[~as_text.isin(allowed_values)]
    if not invalid.empty:
        invalid_rows = (invalid.index + 2).tolist()
        raise ValueError(
            "used_in_ml contains unsupported values. "
            'allowed_values=["TRUE", "FALSE"], '
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, "
            f"csv_rows={invalid_rows[:30]}"
        )
    return as_text == "TRUE" # 최종적으로 모델에 사용할 row만 True인 boolean mask를 반환한다.


def used_in_ml_mask(series: pd.Series) -> pd.Series:
    """하위 호환용 wrapper. 실제 파싱 정책은 parse_used_in_ml()에만 둔다."""
    # 기존 코드가 used_in_ml_mask()를 호출하고 있어도 동일한 strict 정책을 적용하기 위한 wrapper 함수다.
    return parse_used_in_ml(series)


def normalize_used_in_ml_values(series: pd.Series) -> pd.Series:
    """
    legacy bool 표기를 표준 문자값 "TRUE" / "FALSE"로 변환한다.
    이 함수는 export 사본 정규화용이다. 학습 입력 검증은 parse_used_in_ml()이 수행하며,
    최종 CSV에는 반드시 "TRUE" / "FALSE"만 남아야 한다.
    """
    # 정규화 단계에서도 누락값은 허용하지 않는다. 결측시 중단
    if series.isna().any():
        missing_rows = (series[series.isna()].index + 2).tolist()
        raise ValueError(f"used_in_ml contains missing values. csv_rows={missing_rows[:30]}")

    # pandas가 bool dtype으로 읽은 경우 True/False를 표준 문자열로 변환한다.
    if pd.api.types.is_bool_dtype(series):
        return series.map(lambda value: "TRUE" if bool(value) else "FALSE")

    # 과거 산출물에서 "True"/"False"처럼 대소문자가 다른 표기를 허용해 표준값으로 바꾼다.
    mapping = {
        "TRUE": "TRUE",
        "FALSE": "FALSE",
        "True": "TRUE",
        "False": "FALSE",
    }
    as_text = series.astype(str).str.strip()

    # 정규화 가능한 값인지 먼저 검사한다.
    invalid = as_text[~as_text.isin(mapping)]
    if not invalid.empty:
        invalid_rows = (invalid.index + 2).tolist()
        raise ValueError(
            "used_in_ml contains unsupported values for normalization. "
            'allowed_values=["TRUE", "FALSE", "True", "False"], '
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, "
            f"csv_rows={invalid_rows[:30]}"
        )

    return as_text.map(mapping) # 표준 문자열 값으로 변환된 Series를 반환한다.


def check_feature_columns_file(
    path: str | Path,                       # 파일명 또는 경로
    *,
    project_root: str | Path | None = None, # path가 상대경로일 때 해석 기준. None 처리 방식은 resolve_project_path() 정책을 따름
    label_col: str = "label",               # label 컬럼명
    strict: bool = False,                   # True이면 오류를 결과 객체로 감싸지 않고 원래 예외를 그대로 발생시킨다.
) -> FeatureColumnsCheckResult:
    """
    ml_feature_columns.csv 파일을 검증하고 처리 결과를 반환한다.
    반환
    ----
    - ok=True: 검증 성공
    - ok=False: 검증 실패. error_type, error_message에 원인 기록
    """
    # 예외 발생 전 path를 결과 객체에 담기 위해 초기값을 만든다.
    feature_columns_path = Path(path)

    try:
        # 입력 path를 실제 파일 경로로 변환한다.
        # 확인 필요: 상대경로 기준과 프로젝트 밖 경로 허용 여부는 resolve_project_path() 정의를 확인해야 한다.
        feature_columns_path = _resolve_feature_path(path, project_root)

        # feature 정의 CSV가 없으면 중단한다.
        if not feature_columns_path.exists():
            raise FileNotFoundError(f"feature columns file not found: {feature_columns_path}")

        # used_in_ml은 문자열 규칙을 엄격히 검증해야 하므로 dtype을 string으로 지정한다.
        feature_table = pd.read_csv(feature_columns_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})

        # column_name은 실제 parquet feature 컬럼명, used_in_ml은 모델 사용 여부다.
        required_columns = {"column_name", "used_in_ml"}
        missing_columns = required_columns - set(feature_table.columns)
        if missing_columns:
            raise ValueError(f"Feature columns CSV is missing columns: {sorted(missing_columns)}")

        # used_in_ml을 strict boolean mask로 변환한다.TRUE인 row만 이후 모델 입력 후보가 된다.
        mask = parse_used_in_ml(feature_table["used_in_ml"])
        selected_names = feature_table.loc[mask, "column_name"]

        # TRUE로 선택된 row인데 column_name이 비어 있으면 중단시킨다.
        missing_names = selected_names[selected_names.isna()]
        if not missing_names.empty:
            missing_rows = (missing_names.index + 2).tolist()
            raise ValueError(f"Selected feature rows contain missing column_name. csv_rows={missing_rows[:30]}")

        # 공백을 제거해 실제 feature 컬럼명 목록을 만든다. 이 list 순서가 모델 입력 X의 컬럼 순서가 된다.
        feature_columns = selected_names.astype(str).str.strip().tolist()

        # column_name이 공백 문자열인 경우도 명시적으로 차단한다.
        blank_rows = [int(row_index) + 2 for row_index, column in zip(selected_names.index, feature_columns) if not column]
        if blank_rows:
            raise ValueError(f"Selected feature rows contain blank column_name. csv_rows={blank_rows[:30]}")

        # 같은 feature가 중복 선택되면 중단시킨다.
        duplicated = sorted({column for column in feature_columns if feature_columns.count(column) > 1})
        if duplicated:
            raise ValueError(f"Duplicated selected feature columns: {duplicated}")

        # label_col 및 금지 이름/금지 문자열 목록에 해당하는 누수 위험 feature가 포함되었는지 검사한다.
        # 확인 필요: 실제 금지어 범위는 FORBIDDEN_EXACT_NAMES, FORBIDDEN_SUBSTRINGS 정의를 따른다.
        validate_no_forbidden_features(feature_columns, label_col=label_col)

        # TRUE로 선택된 feature가 하나도 없으면 중단한다.
        if not feature_columns:
            raise ValueError(f"No usable feature columns found. path={feature_columns_path}")

        # 검증 성공 시 feature 목록과 카운트를 함께 반환한다.
        return FeatureColumnsCheckResult(
            ok=True,
            processed=True,
            path=feature_columns_path,
            total_rows=int(len(feature_table)),
            selected_count=int(len(feature_columns)),
            selected_columns=feature_columns,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        # strict=True는 호출자에게 원래 예외를 그대로 전달한다.
        # strict=False는 배치/노트북에서 실패 원인을 결과 객체로 보고 싶을 때 사용한다.
        if strict:
            raise
        return FeatureColumnsCheckResult(
            ok=False,
            processed=False,
            path=feature_columns_path,
            total_rows=0,
            selected_count=0,
            selected_columns=[],
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def normalize_feature_columns_file(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    project_root: str | Path | None = None,
    label_col: str = "label",
    overwrite: bool = False,
    strict: bool = False,
) -> FeatureColumnsCheckResult:
    """
    ml_feature_columns.csv의 used_in_ml 값을 "TRUE" / "FALSE" 문자값으로 저장한다.
    기본값은 path 파일을 제자리에서 정규화한다. 원본 산출물을 보존해야 하는 경우,
    먼저 export 사본을 만든 뒤 그 사본 경로를 path로 넘긴다.
    """
    # source_path: 읽을 CSV
    # target_path: 정규화 결과를 저장할 CSV
    # output_path가 없으면 원본 파일을 제자리에서 덮어쓴다.
    source_path = Path(path)
    target_path = Path(output_path) if output_path is not None else Path(path)

    try:
        # 경로 해석은 resolve_project_path()에 위임한다.
        source_path = _resolve_feature_path(path, project_root)
        target_path = source_path if output_path is None else _resolve_feature_path(output_path, project_root)

        # 원본 feature column CSV가 없으면 정규화할 수 없다.
        if not source_path.exists():
            raise FileNotFoundError(f"feature columns file not found: {source_path}")

        # output_path가 별도 파일이고 이미 존재하면 overwrite=True 없이는 덮어쓰지 않는다.
        # 실험 산출물이나 기존 feature 정의 파일을 의도치 않게 덮어쓰는 것을 막는다.
        if target_path.exists() and target_path != source_path and not overwrite:
            raise FileExistsError(f"normalized feature columns output already exists: {target_path}")

        # used_in_ml 값을 정규화하기 위해 CSV를 읽는다.
        feature_table = pd.read_csv(source_path, encoding="utf-8-sig", dtype={"used_in_ml": "string"})

        # 정규화 전에도 필수 컬럼 구조는 유지되어야 한다.
        required_columns = {"column_name", "used_in_ml"}
        missing_columns = required_columns - set(feature_table.columns)
        if missing_columns:
            raise ValueError(f"Feature columns CSV is missing columns: {sorted(missing_columns)}")

        # 원본 DataFrame을 직접 건드리지 않고 copy 후 used_in_ml만 표준 문자열로 변환한다.
        feature_table = feature_table.copy()
        feature_table["used_in_ml"] = normalize_used_in_ml_values(feature_table["used_in_ml"])

        # 정규화된 CSV를 저장한다. output_path가 없으면 source_path를 덮어쓴다.
        target_path.parent.mkdir(parents=True, exist_ok=True)
        feature_table.to_csv(target_path, index=False, encoding="utf-8-sig")

        # 저장 직후 다시 strict 검증을 수행해 최종 파일이 학습 입력 규칙을 만족하는지 확인한다.
        return check_feature_columns_file(
            target_path,
            project_root=None,
            label_col=label_col,
            strict=True,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        # 정규화 실패도 check 함수와 동일하게 strict 여부에 따라 예외 또는 결과 객체로 처리한다.
        if strict:
            raise
        return FeatureColumnsCheckResult(
            ok=False,
            processed=False,
            path=target_path,
            total_rows=0,
            selected_count=0,
            selected_columns=[],
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def load_feature_columns(
    path: str | Path,
    label_col: str = "label",
    project_root: str | Path | None = None,
) -> list[str]:
    """
    ml_feature_columns.csv에서 모델에 사용할 feature column 목록을 read하고 검증하여 반환
    필수 컬럼
    - column_name: 실제 parquet에 존재해야 하는 feature 컬럼명
    - used_in_ml: 모델 사용 여부

    동작 의도
    1. CSV를 read하여 DataFrame으로 로드
    2. 필수 컬럼이 있는지 확인
    3. used_in_ml="TRUE"인 행만 CSV 순서 그대로 선택
    4. 빈 column_name과 중복 feature를 차단
    5. label/target/누수 의심 컬럼이 feature에 들어갔는지 검사
    6. 최종 feature column list를 반환
    이 함수는 stage, feature_group, 파일명에서 feature 조합을 추론하지 않음
    """
    # 외부 호출자가 가장 많이 사용할 진입점이다.
    # 내부적으로 check_feature_columns_file(strict=True)를 호출하므로 실패 시 즉시 예외가 발생한다.
    result = check_feature_columns_file(
        path,
        project_root=project_root,
        label_col=label_col,
        strict=True,
    )

    # 모델 학습/평가에 사용할 feature 컬럼명 list만 반환한다.
    return result.selected_columns


def load_encoding_manifest(path: str | Path | None) -> dict[str, Any] | None:
    """encoding_manifest.json을 읽고 native categorical 메타데이터를 검증한다."""

    if path is None:
        return None
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"encoding manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"encoding manifest must be a JSON object: {manifest_path}")
    feature_types = manifest.get("feature_types")
    if not isinstance(feature_types, dict):
        raise ValueError(f"encoding manifest is missing feature_types object: {manifest_path}")
    category_values = manifest.get("category_values", {})
    if category_values is not None and not isinstance(category_values, dict):
        raise ValueError(f"encoding manifest category_values must be an object: {manifest_path}")
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def categorical_columns_from_manifest(
    encoding_manifest: dict[str, Any] | None,
    feature_columns: list[str],
) -> list[str]:
    """feature_columns 중 XGBoost native categorical 컬럼만 반환한다."""

    if encoding_manifest is None:
        return []
    feature_types = encoding_manifest.get("feature_types", {})
    return [column for column in feature_columns if feature_types.get(column) == "c"]


def apply_encoding_manifest(
    x: pd.DataFrame,
    encoding_manifest: dict[str, Any] | None,
    feature_columns: list[str],
) -> pd.DataFrame:
    """manifest 기준으로 native categorical 컬럼 dtype을 복원한다."""

    categorical_columns = categorical_columns_from_manifest(encoding_manifest, feature_columns)
    if not categorical_columns:
        return x

    category_values = encoding_manifest.get("category_values", {}) if encoding_manifest is not None else {}
    missing_categories = [column for column in categorical_columns if column not in category_values]
    if missing_categories:
        raise ValueError(
            "encoding manifest is missing category values for categorical columns. "
            f"missing={missing_categories}"
        )

    converted = x.copy()
    for column in categorical_columns:
        categories = [str(value) for value in category_values[column]]
        values = converted[column].astype("string")
        known_mask = values.isna() | values.isin(categories)
        if not bool(known_mask.all()):
            if UNKNOWN_CATEGORY not in categories:
                unknown_values = sorted(values[~known_mask].dropna().unique().tolist())
                raise ValueError(
                    "encoding manifest is missing the unknown category sentinel. "
                    f"column={column!r}, sentinel={UNKNOWN_CATEGORY!r}, "
                    f"unknown_values={unknown_values[:30]}, unknown_count={int((~known_mask).sum())}"
                )
            values = values.where(known_mask, UNKNOWN_CATEGORY)
        converted[column] = pd.Categorical(values, categories=categories)
    return converted


def validate_no_forbidden_features(feature_columns: list[str], label_col: str = "label") -> None:
    """
    feature column 목록에 정답 누수 위험이 있는 이름이 들어갔는지 검사
    차단 기준
    - 정확히 금지 이름과 일치: label, target, y, is_laundering 등
    - 금지 문자열 포함: laundering, pattern, typology, attempt 등

    확인 필요: 금지어 목록이 과도하면 정상 feature가 차단될 수 있고, 부족하면 label leakage를 놓칠 수 있다.
    """

    # 정확히 일치하면 금지할 컬럼명을 소문자로 정규화한다.
    forbidden_exact = {name.lower() for name in FORBIDDEN_EXACT_NAMES}

    # 호출자가 지정한 label_col도 금지 목록에 추가한다..
    forbidden_exact.add(str(label_col).strip().lower())
    violations: list[str] = []

    # feature 이름을 하나씩 검사한다.
    for column in feature_columns:
        normalized = str(column).strip().lower()
        # label, target, y 등 정확히 금지 이름과 일치하는 경우 차단한다.
        if normalized in forbidden_exact:
            violations.append(column)
            continue
        # laundering, pattern 등 특정 문자열을 포함하면 누수 가능성이 있으므로 차단한다.
        if any(pattern in normalized for pattern in FORBIDDEN_SUBSTRINGS):
            violations.append(column)

    # 누수 의심 컬럼이 하나라도 있으면 학습을 중단한다.
    if violations:
        raise ValueError(
            "Data leakage risk: forbidden feature names were selected. "
            f"violations={violations[:30]}, violation_count={len(violations)}"
        )


# -----------------------------------------------------------------------------
# 4. feature column 재현성 저장/로드 함수
# -----------------------------------------------------------------------------
def feature_columns_hash(feature_columns: list[str]) -> str:
    """
    feature column 목록에 대한 SHA256 hash를 생성
    중요한 점
    - 이 hash는 순서를 포함하여 feature column 목록 전체에 대한 fingerprint 역할
    - ["a", "b"]와 ["b", "a"]는 서로 다른 hash
    모델 작동 관점
    - 대부분의 ML 모델은 입력 feature 순서를 그대로 사용
    - 학습 때와 예측 때 feature 순서가 달라지면 모델은 완전히 다른 의미의 값을 입력받게 됨
    """
    # JSON 문자열로 직렬화해 list 구조와 순서를 그대로 반영한다.
    # separators를 고정해 같은 feature 목록이면 항상 같은 문자열이 만들어지도록 한다.
    payload = json.dumps(feature_columns, ensure_ascii=False, separators=(",", ":"))

    # feature 목록의 fingerprint를 만든다.
    # 이 값은 학습 시점과 평가/추론 시점의 feature 순서 일치 여부를 확인하는 데 사용된다.
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def save_feature_columns(feature_columns: list[str], path: str | Path) -> None:
    """
    모델이 사용한 feature column 목록과 hash를 JSON 파일로 저장
    사용 시점
    - 모델 학습 직후, 실험 기록 저장 시, 추론/평가 단계에서 동일한 feature 순서를 재사용해야 할 때
    """
    # 학습에 사용한 feature 순서를 결과 디렉터리에 저장하기 위한 함수다.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # feature_columns와 hash를 함께 저장한다.
    # 이후 load_saved_feature_columns()에서 hash mismatch를 감지할 수 있다.
    payload = {
        "feature_columns": feature_columns,
        "feature_columns_hash": feature_columns_hash(feature_columns),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_saved_feature_columns(path: str | Path) -> list[str]:
    """
    저장된 feature column JSON을 읽고 hash가 맞는지 확인

    동작 의도
    - 저장된 feature_columns를 read하여 리스트로 반환
    - 리스트가 비어 있거나 형식이 맞지 않으면 에러
    - 저장된 hash와 현재 계산한 hash가 다르면 파일이 수정되었을 가능성이 있으므로 에러
    """
    # 학습 시 저장한 feature 목록을 다시 읽는다.
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    # feature_columns 필드는 반드시 비어 있지 않은 list여야 한다.
    columns = payload.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Invalid feature_columns file: {path}")

    # JSON에서 읽은 값을 문자열 list로 정규화한다.
    feature_columns = [str(col) for col in columns]

    # 저장된 hash가 있으면 현재 feature_columns로 다시 계산한 hash와 비교한다.
    # 다르면 파일이 수동 수정되었거나 순서가 바뀌었을 가능성이 있다.
    saved_hash = payload.get("feature_columns_hash")
    if saved_hash is not None and saved_hash != feature_columns_hash(feature_columns):
        raise ValueError(
            f"feature_columns hash mismatch in saved file: {path}. "
            "The feature order may have been modified."
        )

    return feature_columns  # 검증된 feature 컬럼 순서를 반환한다.


# -----------------------------------------------------------------------------
