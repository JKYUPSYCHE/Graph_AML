"""
ML-ready parquet 학습 데이터를 안전하게 불러오기 위한 입출력 유틸리티 모듈

모델 학습/검증/평가 직전에 필요한 입력 데이터 X, y를 안전하게 준비

전체 흐름
1. 사용자가 입력한 데이터 경로를 절대경로 or PROJECT_ROOT 기준 경로로 해석
2. ml_feature_columns.csv에서 used_in_ml="TRUE"인 feature column 목록 선택
3. label/target 계열 컬럼이나 누수 가능성이 큰 컬럼명이 feature에 들어가지 않도록 차단
4. parquet 파일에서 필요한 컬럼만 read
5. X는 숫자형인지, NaN/inf가 없는지 확인
6. y는 이진분류용 0/1 label인지, 두 클래스가 모두 존재하는지 확인
7. feature column 순서와 hash를 저장하여 재현성을 확보

중요한 전제
- Python 3.10 이상 권장: `str | Path`, `Path | None` 같은 union type 문법 사용
- parquet 처리를 위해 pyarrow 필요
- 이 모듈은 “데이터 로딩 및 검증” 모듈, scikit-learn, XGBoost, LightGBM 같은 모델 객체는 여기서 생성하지 않음
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. 데이터 누수 방지용 금지 feature 이름 규칙
# -----------------------------------------------------------------------------
# 제외 컬럼, 누락 컬럼, 또는 이름에 특정 패턴이 포함된 컬럼은 모델에서 사용하지 않도록 안전장치 역할
FORBIDDEN_EXACT_NAMES = {"label", "target", "y", "is_laundering"}
FORBIDDEN_SUBSTRINGS = {"laundering", "pattern", "typology", "attempt"}
UNKNOWN_CATEGORY = "__UNKNOWN__"


# -----------------------------------------------------------------------------
# 2. 경로 처리 함수
# -----------------------------------------------------------------------------
def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """
    입력받은 경로를 실제 사용할 Path 객체로 변환

    동작 의도
    - 절대경로가 들어오면 그대로 정규화해서 반환
    - 상대경로가 들어오면 project_root 기준으로 붙여서 절대경로 생성
    - 상대경로인데 project_root가 없으면, 노트북 실행 위치에 따라 파일을 잘못 찾을 수 있으므로 즉시 에러

    예시
    - path="/home/user/data/train.parquet" -> 그대로 사용
    - path="data/train.parquet", project_root="/home/user/project" -> /home/user/project/data/train.parquet
    """
    resolved = Path(path).expanduser() # 입력값을 Path 객체로 통일하고, "~" 같은 사용자 홈 경로를 실제 경로로 확장

    # 절대경로는 project_root를 붙이지 않고 그대로 정규화해서 반환
    if resolved.is_absolute():
        return resolved.resolve()

    # 상대경로는 노트북 실행 위치에 따라 달라질 수 있으므로 명시적 project_root를 요구
    if project_root is None:
        raise ValueError(
            "Relative paths require project_root. "
            "Pass PROJECT_ROOT from Notebook Bootstrap or use an absolute path."
        )
    # project_root를 절대경로로 정규화한 뒤, 상대경로를 붙여 최종 경로 확정
    return (Path(project_root).expanduser().resolve() / resolved).resolve()


@dataclass(frozen=True)
class InputPaths:
    """
    모델 학습/검증/평가에 필요한 입력 파일 경로를 하나로 묶는 설정 객체.
    필드 설명
    - train_path: 학습 데이터 parquet 경로
    - val_path: validation 데이터 parquet 경로
    - test_path: test 데이터 parquet 경로. test를 쓰지 않는 단계에서는 None 가능
    - feature_columns_path: 모델 입력 피처 컬럼 목록 CSV 경로
    """
    train_path: Path
    val_path: Path
    # test 데이터가 항상 필요한 것은 아니므로 None을 허용한다.
    # 예: validation threshold tuning 단계에서는 test를 아직 사용하지 않을 수 있다.
    test_path: Path | None
    feature_columns_path: Path
    encoding_manifest_path: Path | None = None


def print_input_paths(paths: InputPaths) -> None:
    """
    현재 설정된 입력 파일 경로를 출력한다.
    """
    print("train_path          :", paths.train_path)
    print("val_path            :", paths.val_path)
    print("test_path           :", paths.test_path)
    print("feature_columns_path:", paths.feature_columns_path)
    if paths.encoding_manifest_path is not None:
        print("encoding_manifest_path:", paths.encoding_manifest_path)


def require_input_files(paths: InputPaths, require_test: bool = False) -> None:
    """
    필수 입력 파일이 실제로 존재하는지 확인한다.
    동작 방식
    - train, val, feature_columns 파일은 항상 검사한다.
    - require_test=True이면 test_path도 필수로 검사한다.
    - require_test=True인데 test_path가 None이면 ValueError를 발생시킨다.
    - 존재하지 않는 파일이 하나라도 있으면 FileNotFoundError를 발생시킨다.
    주의
    - 현재 구현은 path.exists()만 검사한다.
    - 즉, 경로가 존재하면 통과하며, 그것이 실제 파일인지 디렉터리인지는 검사하지 않는다.
    - 파일 여부까지 엄격히 보려면 exists() 대신 is_file() 사용을 검토할 수 있다.
    """
    # 기본적으로 학습과 검증에 필요한 파일은 항상 필수 입력으로 본다.
    required = {
        "train": paths.train_path,
        "val": paths.val_path,
        "feature_columns": paths.feature_columns_path,
    }
    if paths.encoding_manifest_path is not None:
        required["encoding_manifest"] = paths.encoding_manifest_path

    # test 평가는 최종 평가 단계에서만 필요할 수 있으므로 옵션으로 검사한다.
    if require_test:
        # require_test=True이면 test_path가 반드시 지정되어 있어야 한다.
        # None 상태로 넘어가면 어떤 test 파일을 검사해야 할지 알 수 없으므로 즉시 실패시킨다.
        if paths.test_path is None:
            raise ValueError("test_file_name is required when require_test=True.")
        required["test"] = paths.test_path

    # required에 들어 있는 경로 중 실제로 존재하지 않는 항목만 모은다.
    # dict 형태로 만들면 어떤 역할의 파일이 어떤 경로에서 누락됐는지 바로 확인할 수 있다.
    missing = {
        name: str(path)
        for name, path in required.items()
        if not path.exists()
    }

    # 누락된 파일이 하나라도 있으면 조용히 넘어가지 않고 명시적으로 실패시킨다.
    # AML 실험에서는 잘못된 입력 파일로 학습하는 것보다 초기에 실패하는 편이 안전하다.
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")


# -----------------------------------------------------------------------------
# 3. ml_feature_columns.csv 처리 함수
#  1. 코드 전체 요약
#      ml_feature_columns.csv를 기준으로 사용할 feature 컬럼을 검증하고,
#      parquet split 파일에서 X, y를 안전하게 읽어 모델 입력 형태로 만드는 코드다.
#  핵심 책임
#      - used_in_ml == "TRUE"인 feature만 선택
#      - label, laundering, pattern 등 정답 누수 위험 컬럼 차단
#      - feature 컬럼 순서와 hash 저장으로 학습/평가 재현성 확보
#      - parquet 전체를 읽기 전에 schema로 필요한 컬럼 존재 여부 확인
#      - X는 숫자형 feature matrix, y는 0/1 binary label로 검증
#      - split 컬럼이 기대한 train, val, test와 맞는지 확인
#      - 실험 설정, 평가 결과 등을 JSON으로 저장/로드

#  2. 데이터 흐름 요약
#      1. ml_feature_columns.csv 입력
#      2. column_name, used_in_ml 필수 컬럼 검증
#      3. used_in_ml == "TRUE"인 row만 선택
#      4. 선택된 column_name 목록에서 빈 값, 중복, 누수 위험 이름 제거
#      5. 최종 feature_columns 생성
#      6. parquet split 파일 schema 확인
#      7. 필요한 컬럼만 읽음: feature_columns + label_col + split
#      8. split 값 검증
#      9. label을 숫자형으로 변환 후 0/1 검증
#      10. X = df[feature_columns], y = label
#      11. X의 숫자형, NaN, inf 검증
#      12. 모델 학습/평가에서 사용할 X, y 반환
# -----------------------------------------------------------------------------

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
        feature_columns_path = resolve_project_path(path, project_root)

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
        source_path = resolve_project_path(path, project_root)
        target_path = source_path if output_path is None else resolve_project_path(output_path, project_root)

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


def file_sha256(path: str | Path) -> str:
    """파일 내용의 SHA256 hash를 계산한다."""

    # 파일 자체가 변했는지 추적하기 위한 일반 hash 함수다.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found for sha256: {path}")
    digest = hashlib.sha256()

    # 큰 파일도 한 번에 메모리에 올리지 않고 1MB 단위로 읽는다.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
# 5. parquet 읽기 함수
# -----------------------------------------------------------------------------
def _get_pyarrow_parquet_module():
    """ pyarrow.parquet 모듈 import """
    # pyarrow는 parquet schema 확인과 batch 단위 읽기에 필요.
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to inspect/read parquet files. "
            "Install pyarrow in the environment used by this notebook."
        ) from exc
    return pq


def get_parquet_columns(path: str | Path) -> list[str]:
    """
    parquet 파일 전체를 메모리에 올리지 않고 schema의 컬럼명만 read하여 반환
    동작 의도
    - load_split에서 필요한 컬럼이 실제 parquet에 있는지 먼저 확인하기 위해 사용
    - 데이터가 매우 커도 schema만 읽으므로 상대적으로 가벼움
    """
    path = Path(path)

    # 이 유틸리티는 parquet만 지원한다.
    # CSV 등을 허용하면 아래 pyarrow parquet schema 로직이 맞지 않는다.
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported. path={path}")
    pq = _get_pyarrow_parquet_module()

    # parquet 전체 데이터를 읽지 않고 schema metadata에서 컬럼명만 가져온다.
    return pq.ParquetFile(path).schema_arrow.names


def read_parquet_columns(
    path: str | Path,                 # parquet 파일 경로
    columns: list[str],               # 읽을 컬럼명 목록
    sample_rows: int | None = None,   # None이면 전체 row, 정수이면 앞에서 해당 개수만 읽음
) -> pd.DataFrame:
    """
    parquet에서 지정한 컬럼만 read, 필요시 앞에서 N개 row만 read 하여 DataFrame으로 반환
    동작 의도
    - 전체 컬럼을 읽지 않고 feature + label + split 정도만 읽어 메모리 사용절감
    - sample_rows는 빠른 테스트용, (데이터가 불균형하면 앞부분 샘플에 한 클래스만 들어갈 수 있음)
    주의
    - sample_rows=None일 때는 pandas.read_parquet을 직접 사용
    - sample_rows가 지정되면 pyarrow의 batch iterator로 필요한 row 수만큼 읽음
    """
    path = Path(path)

    # sample_rows가 없으면 지정 컬럼 전체를 읽는다.
    if sample_rows is None:
        return pd.read_parquet(path, columns=columns)

    # sample_rows는 빠른 검증용이다. 0 이하 값은 의미가 없으므로 명시적으로 차단한다.
    if sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer.")
    pq = _get_pyarrow_parquet_module()
    parquet_file = pq.ParquetFile(path)

    # remaining: 앞으로 더 읽어야 하는 row 수.
    remaining = int(sample_rows)

    # batch별 pandas DataFrame을 임시 저장한 뒤 마지막에 concat한다.
    frames: list[pd.DataFrame] = []

    # 너무 큰 batch를 만들지 않도록 최대 65,536 row 단위로 제한한다.
    batch_size = min(remaining, 65_536)

    # parquet에서 필요한 컬럼만 batch 단위로 읽는다.
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        # 마지막 batch가 요청 row 수보다 많으면 필요한 만큼만 자른다.
        if len(frame) > remaining:
            frame = frame.iloc[:remaining]
        frames.append(frame)
        remaining -= len(frame)
        # 요청한 sample_rows만큼 읽었으면 더 이상 parquet를 읽지 않는다.
        if remaining <= 0:
            break

    # parquet가 비어 있거나 읽힌 batch가 없으면 지정 컬럼 구조만 가진 빈 DataFrame을 반환한다.
    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True) # batch별 DataFrame을 하나로 합쳐 호출자에게 반환한다.


def _require_existing_file(path: str | Path, label: str) -> Path:
    """입력 경로가 실제 파일인지 확인하고 Path로 반환한다."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def validate_parquet_split_values(path: str | Path, expected_split: str) -> None:
    """parquet의 split 컬럼이 기대 split 하나로만 구성되는지 batch 단위로 검증한다."""

    parquet_path = Path(path)
    expected = str(expected_split).strip().lower()
    if not expected:
        raise ValueError("expected_split must be a non-empty string.")

    pq = _get_pyarrow_parquet_module()
    parquet_file = pq.ParquetFile(parquet_path)
    observed: set[str] = set()
    null_count = 0
    blank_count = 0

    for batch in parquet_file.iter_batches(batch_size=65_536, columns=["split"]):
        normalized = batch.column(0).to_pandas().astype("string").str.strip().str.lower()
        null_count += int(normalized.isna().sum())
        non_null = normalized.dropna()
        blank_count += int((non_null == "").sum())
        observed.update(value for value in non_null.unique().tolist() if value != "")

        unexpected = sorted(value for value in observed if value != expected)
        if unexpected:
            raise ValueError(
                f"Unexpected split values. source={parquet_path}, "
                f"expected={expected!r}, values={sorted(observed)}"
            )

    if null_count:
        raise ValueError(
            f"Split column contains null values. source={parquet_path}, "
            f"expected_split={expected!r}, null_count={null_count}"
        )
    if blank_count:
        raise ValueError(
            f"Split column contains blank values. source={parquet_path}, "
            f"expected_split={expected!r}, blank_count={blank_count}"
        )
    if observed != {expected}:
        raise ValueError(
            f"Unexpected split values. source={parquet_path}, "
            f"expected={expected!r}, values={sorted(observed)}"
        )


def preflight_ml_inputs(
    paths: InputPaths,
    *,
    label_col: str = "label",
    require_test: bool = False,
    check_split_values: bool = True,
) -> dict[str, Any]:
    """학습/튜닝 실행 전에 승인 입력, schema, native categorical manifest 정합성을 검증한다."""

    required_files = {
        "train": paths.train_path,
        "val": paths.val_path,
        "feature_columns": paths.feature_columns_path,
    }
    split_paths: dict[str, Path] = {
        "train": Path(paths.train_path),
        "val": Path(paths.val_path),
    }

    if paths.encoding_manifest_path is not None:
        required_files["encoding_manifest"] = paths.encoding_manifest_path

    if require_test:
        if paths.test_path is None:
            raise ValueError("test_path is required when require_test=True.")
        required_files["test"] = paths.test_path
        split_paths["test"] = Path(paths.test_path)

    for label, file_path in required_files.items():
        _require_existing_file(file_path, label)

    feature_columns = load_feature_columns(paths.feature_columns_path, label_col=label_col)
    encoding_manifest = load_encoding_manifest(paths.encoding_manifest_path)
    categorical_columns = categorical_columns_from_manifest(encoding_manifest, feature_columns)
    native_categorical_like = [
        column
        for column in feature_columns
        if column.startswith("cat__") or column.endswith("__xgb_cat")
    ]

    if native_categorical_like and encoding_manifest is None:
        raise ValueError(
            "encoding_manifest_path is required for native categorical features. "
            f"categorical_like_features={native_categorical_like[:30]}, "
            f"count={len(native_categorical_like)}"
        )

    if encoding_manifest is not None:
        feature_types = encoding_manifest.get("feature_types", {})
        missing_feature_types = [column for column in feature_columns if column not in feature_types]
        if missing_feature_types:
            raise ValueError(
                "encoding manifest is missing feature_types for selected features. "
                f"missing={missing_feature_types[:30]}, missing_count={len(missing_feature_types)}"
            )

        unmarked_categorical = [column for column in native_categorical_like if column not in categorical_columns]
        if unmarked_categorical:
            raise ValueError(
                "native categorical-like features must be marked as categorical in encoding manifest. "
                f"unmarked={unmarked_categorical[:30]}, count={len(unmarked_categorical)}"
            )

        category_values = encoding_manifest.get("category_values", {})
        missing_category_values = [column for column in categorical_columns if column not in category_values]
        if missing_category_values:
            raise ValueError(
                "encoding manifest is missing category_values for categorical features. "
                f"missing={missing_category_values[:30]}, missing_count={len(missing_category_values)}"
            )

    required_columns = set(feature_columns) | {label_col, "split"}
    for split_name, split_path in split_paths.items():
        available_columns = set(get_parquet_columns(split_path))
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            raise ValueError(
                f"{split_name} parquet is missing required ML input columns. "
                f"missing={missing_columns[:30]}, missing_count={len(missing_columns)}, path={split_path}"
            )
        if check_split_values:
            validate_parquet_split_values(split_path, split_name)

    return {
        "feature_count": len(feature_columns),
        "feature_columns_hash": feature_columns_hash(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "checked_splits": sorted(split_paths),
        "encoding_manifest_path": None if paths.encoding_manifest_path is None else str(paths.encoding_manifest_path),
    }


# -----------------------------------------------------------------------------
# 6. X, y 검증 함수
# -----------------------------------------------------------------------------
def validate_features(
    x: pd.DataFrame,
    source_path: str | Path,
    allow_nan: bool = False,
    categorical_columns: list[str] | None = None,
) -> None:
    """
    모델 입력 feature matrix X가 학습 가능한 형태인지 검사
    검사 항목
    1. 비어 있지 않은가?
    2. 모든 컬럼이 숫자형인가?
    3. allow_nan=False일 때 NaN이 없는가?
    4. 무한대 inf, -inf 값이 없는가?

    모델 작동 관점
    - 많은 모델은 문자열/object feature를 직접 처리하지 못하므로 숫자형인지 확인
    - NaN은 모델 종류에 따라 허용 여부가 다름
    - inf 값은 대부분의 모델에서 학습 오류 또는 비정상적인 분기 원인이 됨
    """
    # feature matrix가 비어 있으면 학습/평가가 불가능하다.
    if x.empty:
        raise ValueError(f"Feature matrix is empty. source={source_path}")

    categorical_set = set(categorical_columns or [])
    unknown_categorical = sorted(categorical_set - set(x.columns))
    if unknown_categorical:
        raise ValueError(f"categorical columns are missing from X. source={source_path}, missing={unknown_categorical}")

    # 기본은 숫자형만 허용한다. native categorical manifest가 있는 컬럼만 pandas category dtype을 허용한다.
    non_numeric = [
        column
        for column in x.columns
        if not pd.api.types.is_numeric_dtype(x[column])
        and not (column in categorical_set and pd.api.types.is_categorical_dtype(x[column]))
    ]
    if non_numeric:
        raise ValueError(f"All features must be numeric. source={source_path}, non_numeric={non_numeric}")

    # NaN 허용 여부는 모델에 따라 다르다. allow_nan=False가 기본값이므로, 결측치가 있으면 학습 전에 명시적으로 실패한다.
    if not allow_nan:
        nan_checked = x.drop(columns=list(categorical_set), errors="ignore")
        missing_counts = nan_checked.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            raise ValueError(
                f"Feature matrix contains NaN values. source={source_path}, "
                f"missing_counts={missing_counts.head(30).to_dict()}"
            )

    # inf/-inf는 대부분의 ML 모델에서 오류나 비정상 학습을 유발한다.
    # category 컬럼은 숫자 변환하지 않고, numeric feature만 검사한다.
    numeric_x = x.drop(columns=list(categorical_set), errors="ignore")
    infinite_counts = 0 if numeric_x.empty else np.isinf(numeric_x.to_numpy(dtype="float64", copy=False)).sum()
    if infinite_counts:
        raise ValueError(f"Feature matrix contains infinite values. source={source_path}, count={int(infinite_counts)}")


def validate_labels(
    y: pd.Series,
    source_path: str | Path,
    label_col: str = "label",
    sample_rows: int | None = None,
) -> None:
    """
   label y가 0/1 형식인지 검사
    검사 항목
    1. 비어 있지 않은가?
    2. NaN이 없는가?
    3. 값이 0 또는 1로만 구성되어 있는가?
    4. 두 클래스가 모두 존재하는가?
    주의
    - load_split에서는 label을 숫자형으로 해석한 뒤 int8 변환 전에 이 함수를 호출
    - 따라서 0.5 같은 값이 int8 변환 과정에서 0으로 잘리는 문제를 차단
    """

    # label vector가 비어 있으면 binary classification 학습/평가가 불가능하다.
    if y.empty:
        raise ValueError(f"Label vector is empty. source={source_path}")

    # label 결측치는 정답이 없는 row를 의미하므로 차단한다.
    if y.isna().any():
        raise ValueError(f"Labels contain NaN. source={source_path}, label_col={label_col}")

    # label은 0/1 binary만 허용한다.
    # 이 검사를 int8 변환 전에 수행해야 0.5 같은 값이 0으로 잘리는 문제를 막을 수 있다.
    values = set(y.unique().tolist())
    if not values <= {0, 1}:
        raise ValueError(f"Labels must be binary 0/1. source={source_path}, label_col={label_col}, values={sorted(values)}")

    # 학습/평가 split에 양성/음성 중 하나만 있으면 F1, recall 등 평가가 왜곡되거나 불가능하다.
    # sample_rows 사용 시 앞부분만 읽어서 한 클래스만 나올 수 있으므로 힌트를 추가한다.
    if y.nunique() < 2:
        sample_hint = ""
        if sample_rows is not None:
            sample_hint = f" sample_rows={sample_rows}; increase sample_rows or use the full split."
        raise ValueError(
            "Both classes are required. "
            f"source={source_path}, label_counts={y.value_counts().to_dict()}.{sample_hint}"
        )


def label_summary(y: pd.Series) -> dict[str, Any]:
    """
    label 분포 요약 정보를 dict로 반환
    반환 정보
    - label_counts: 0/1별 개수
    - positive_count: label=1 개수
    - negative_count: label=0 개수
    - positive_ratio: 전체 중 positive 비율
    사용 목적
    - 실험 로그에 저장, 데이터 불균형 확인, train/val/test split 간 label 비율 비교
    """
    # label 값별 개수를 정렬된 dict로 만든다.
    counts = y.value_counts().sort_index().to_dict()
    total = int(len(y))
    positive_count = int(counts.get(1, 0))

    # JSON 저장에 적합하도록 key와 count를 기본 Python 타입으로 변환한다.
    return {
        "label_counts": {str(int(label)): int(count) for label, count in counts.items()},
        "positive_count": positive_count,
        "negative_count": int(counts.get(0, 0)),
        "positive_ratio": None if total == 0 else float(positive_count / total),
    }


def validate_split_column(df: pd.DataFrame, expected_split: str | None, source_path: str | Path) -> None:
    """
    parquet에 split 컬럼이 있을 경우, 기대한 split 값인지 확인
    예시
    - train 파일인데 split 컬럼 값이 모두 "train"이어야 함
    - val 파일인데 split 컬럼 값이 모두 "val"이어야 함
    동작 의도
    - 파일명은 train.parquet인데 내부 split이 val/test이면 데이터 누수나 평가 오류 발생
    - expected_split이 None이면 검사하지 않음
    - split 컬럼이 없으면 검사하지 않음
    """
    # expected_split을 지정하지 않으면 split 검증을 수행하지 않는다.
    if expected_split is None:
        return

    # expected_split이 지정된 경우 split 컬럼은 필수다.
    # 파일명만 믿지 않고 parquet 내부 split 값을 확인해 train/val/test 혼동을 방지한다.
    if "split" not in df.columns:
        raise ValueError(
            "Input parquet is missing required split column. "
            f"source={source_path}, expected_split={expected_split!r}"
        )
    raw_split = df["split"]

    # split 값 누락은 어떤 split에 속한 row인지 알 수 없으므로 차단한다.
    if raw_split.isna().any():
        raise ValueError(
            "Split column contains missing values. "
            f"source={source_path}, expected_split={expected_split!r}, "
            f"missing_count={int(raw_split.isna().sum())}"
        )

    # 문자열화 후 공백 제거, 소문자 변환으로 비교 기준을 맞춘다.
    normalized = raw_split.astype("string").str.strip().str.lower()

    # 공백 문자열도 유효한 split 값이 아니므로 차단한다.
    blank_mask = normalized == ""
    if blank_mask.any():
        raise ValueError(
            "Split column contains blank values. "
            f"source={source_path}, expected_split={expected_split!r}, blank_count={int(blank_mask.sum())}"
        )

    # split 컬럼의 실제 unique 값이 기대값 하나만으로 구성되어야 한다.
    values = set(normalized.unique().tolist())
    expected = expected_split.lower()
    if values != {expected}:
        raise ValueError(
            f"Unexpected split values. source={source_path}, expected={expected!r}, values={sorted(values)}"
        )


def load_split(
    path: str | Path,
    feature_columns: list[str],
    label_col: str = "label",
    sample_rows: int | None = None,
    allow_nan: bool = False,
    expected_split: str | None = None,
    encoding_manifest: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    하나의 ML split 파일을 읽어 X, y로 반환
    ★★★ 가장 중요한 함수 ★★★
        전체 동작 순서
    1. label_col이 feature_columns에 들어갔는지 검사
    2. feature_columns 중복 여부 검사
    3. parquet schema에서 필요한 컬럼이 모두 있는지 확인
    4. feature + label + 필요 시 split 컬럼만 read하여 DataFrame으로 로드
    5. split 컬럼이 기대값과 일치하는지 검사
    6. 원본 label을 숫자형으로 해석하고 0/1인지 검증
    7. y를 int8로 변환
    8. X를 feature_columns 순서대로 추출
    9. X를 검증
    10. 최종적으로 모델 학습/평가에 넣을 수 있는 X, y를 반환
    모델 작동 의도
    - 모델은 이 함수가 반환하는 X의 컬럼 순서 그대로 학습 및 예측을 수행
    - y는 binary classification target
    - 이 함수는 모델 성능 계산 전에 데이터 형식 문제와 정답 누수 문제를 최대한 조기에 막는 역할
    """
    path = Path(path)

    # label 컬럼이 feature 목록에 들어가면  먼저 차단한다.
    if label_col in feature_columns:
        raise ValueError(
            f"Data leakage risk: label_col={label_col!r} is included in feature_columns."
        )

    # 중복 feature는 입력 차원과 해석을 혼란스럽게 하므로 차단한다.
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns contains duplicated names.")

    # parquet에서 반드시 읽어야 하는 컬럼은 feature들과 label이다.
    required_columns = list(feature_columns) + [label_col]

    # 전체 데이터를 읽기 전에 schema만 확인한다.
    # 대용량 AML parquet에서 불필요한 메모리 사용을 줄이는 핵심 지점이다.
    available_columns = get_parquet_columns(path)

    # feature_columns 또는 label_col이 parquet에 없으면 학습 데이터를 만들 수 없다.
    missing = [column for column in required_columns if column not in available_columns]
    if missing:
        raise ValueError(
            f"Input parquet is missing required columns. path={path}, "
            f"missing={missing[:30]}, missing_count={len(missing)}"
        )

    # 실제로 읽을 컬럼 목록을 구성한다.
    # feature + label만 읽고, expected_split이 있으면 split도 추가한다.
    read_columns = list(required_columns)
    if expected_split is not None:
        if "split" not in available_columns:
            raise ValueError(
                "Input parquet is missing required split column. "
                f"path={path}, expected_split={expected_split!r}"
            )
        if "split" not in read_columns:
            read_columns.append("split")

    # parquet에서 필요한 컬럼만 읽는다. sample_rows가 있으면 앞부분 일부만 읽어 빠른 검증에 사용한다.
    df = read_parquet_columns(path, read_columns, sample_rows)

    # train/val/test 파일 혼동을 막기 위해 split 컬럼 값이 기대값과 맞는지 확인한다.
    validate_split_column(df, expected_split=expected_split, source_path=path)

    # int8 변환 전에 원본 label 값을 검증해 0.5 같은 값이 0으로 잘리는 문제를 차단한다.
    raw_y = pd.to_numeric(df[label_col], errors="raise")

    # y가 비어 있지 않고, NaN이 없고, 0/1만 있으며, 양쪽 클래스가 모두 있는지 확인한다.
    validate_labels(raw_y, path, label_col=label_col, sample_rows=sample_rows)

    # 모델 target으로 사용할 y를 메모리 효율적인 int8로 변환한다.
    y = raw_y.astype("int8")

    # feature_columns 순서를 그대로 유지하는 것이 중요, 모델 입력이 이 순서를 기준으로 해석되기 때문
    # 예: 학습 때 ["amount", "degree"]였는데 평가 때 ["degree", "amount"]가 되면 모델 입력 의미가 완전히 바뀐다.
    x = df[feature_columns].copy()
    x = apply_encoding_manifest(x, encoding_manifest, feature_columns)
    categorical_columns = categorical_columns_from_manifest(encoding_manifest, feature_columns)

    # X가 숫자형이고, 필요한 경우 NaN이 없고, inf/-inf가 없는지 최종 검증한다.
    validate_features(x, path, allow_nan=allow_nan, categorical_columns=categorical_columns)

    return x, y # 모델 학습/평가 함수가 바로 사용할 수 있는 X, y를 반환한다.
# -----------------------------------------------------------------------------
# 7. 일반 JSON 저장/로드 함수
# -----------------------------------------------------------------------------

def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """
    dict payload를 UTF-8 JSON 파일로 저장
    사용 예
    - 실험 설정 저장
    - label_summary 결과 저장
    - 모델 평가 metric 저장
    """
    # 실험 설정, label 분포, 평가 metric 등 작은 메타데이터를 저장하는 공통 함수다.
    path = Path(path)

    # 결과 디렉터리가 없으면 생성한다.
    path.parent.mkdir(parents=True, exist_ok=True)

    # ensure_ascii=False로 한글 key/value도 읽기 쉬운 형태로 저장한다.
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """
    UTF-8 JSON 파일을 읽어 Python 객체로 반환합니다.
    주의
    - 타입 힌트는 dict[str, Any]이지만, JSON 파일 내용이 list이면 실제 반환값도 list가 됨
    - 반드시 dict만 허용해야 하는 상황이라면 isinstance(result, dict) 검사를 추가하는 것이 안전
    """
    # JSON 파일을 읽어 Python 객체로 역직렬화한다.
    # 확인 필요: 이 함수는 dict만 검증하지 않으므로, 호출부에서 타입 검증이 필요한지 확인해야 한다.
    return json.loads(Path(path).read_text(encoding="utf-8"))
