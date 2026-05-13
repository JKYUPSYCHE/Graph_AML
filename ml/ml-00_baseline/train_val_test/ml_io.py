"""
ML-ready parquet 학습 데이터를 안전하게 불러오기 위한 입출력 유틸리티 모듈

모델 학습/검증/평가 직전에 필요한 입력 데이터 X, y를 안전하게 준비

전체 흐름
1. 사용자가 입력한 데이터 경로를 절대경로 or PROJECT_ROOT 기준 경로로 해석
2. feature catalog CSV에서 used_in_ml=True인 feature column 목록 선택
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
class InputPaths:
    """
    train/val/test parquet 와 feature catalog 경로를 한 묶음으로 보관하는 자료구조
    frozen=True -> 객체 생성 후 train_path, val_path 같은 값을 실수로 바꾸지 못하게 고정 
    """
    train_path: Path
    val_path: Path
    test_path: Path | None # 이 변수(test)에는 Path 타입 경로 정보가 들어올 수도 있고, None이 들어올 수도 있음
    feature_columns_path: Path


def make_input_paths(
    data_dir: str | Path,
    train_file_name: str,
    val_file_name: str,
    feature_columns_file_name: str,
    test_file_name: str | None = None,
    project_root: str | Path | None = None,
) -> InputPaths:
    """
    데이터 디렉터리와 파일명을 조합해 InputPaths 객체 생성

    동작 의도
    - 노트북에서 사용자가 data_dir, train 파일명, val 파일명 등을 따로 입력-> 이 함수가 조립
    - test_file_name은 선택값, 테스트셋이 아직 없거나 검증까지만 할 때 None으로 둘 수 있음
    """

    base = resolve_project_path(data_dir, project_root=project_root)
    return InputPaths(
        train_path=base / train_file_name,
        val_path=base / val_file_name,
        test_path=None if test_file_name is None else base / test_file_name,
        feature_columns_path=base / feature_columns_file_name,
    )


def print_input_paths(paths: InputPaths) -> None:
    """
    노트북에서 경로를 눈으로 확인할 수 있게 출력
    """
    print("train_path          :", paths.train_path)
    print("val_path            :", paths.val_path)
    print("test_path           :", paths.test_path)
    print("feature_columns_path:", paths.feature_columns_path)


def require_input_files(paths: InputPaths, require_test: bool = False) -> None:
    """
    필수 입력 파일이 존재하는지 확인
    
    동작 의도
    - 파일이 없으면 학습 중간이 아니라 시작 단계에서 바로 실패하도록 구성 
    - require_test=True이면 test_path도 검사
    - test_file_name이 None인데 require_test=True이면 ValueError
    """

    required = {
        "train": paths.train_path,
        "val": paths.val_path,
        "feature_columns": paths.feature_columns_path,
    }
    if require_test:
        if paths.test_path is None:
            raise ValueError("test_file_name is required when require_test=True.")
        required["test"] = paths.test_path

    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")




# -----------------------------------------------------------------------------
# 3. feature catalog 처리 함수
# -----------------------------------------------------------------------------
def used_in_ml_mask(series: pd.Series) -> pd.Series:
    """
    feature catalog의 used_in_ml 값을 strict boolean mask로 해석

    허용되는 true 표현: True, "true", "1", "yes", "y"
    허용되는 false 표현: False, "false", "0", "no", "n"

    주의
    - 허용 범위 밖의 값은 조용히 제외하지 않고 즉시 ValueError를 발생시킴
    - catalog 작성 규칙은 위 표현 중 하나로 통일해야 함
    """

    if series.isna().any():
        missing_rows = (series[series.isna()].index + 2).tolist()
        raise ValueError(f"used_in_ml contains missing values. csv_rows={missing_rows[:30]}")

    if pd.api.types.is_bool_dtype(series):
        return series

    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}
    allowed_values = true_values | false_values
    normalized = series.astype(str).str.strip().str.lower()
    invalid = normalized[~normalized.isin(allowed_values)]
    if not invalid.empty:
        invalid_rows = (invalid.index + 2).tolist()
        raise ValueError(
            "used_in_ml contains unsupported values. "
            f"allowed_true={sorted(true_values)}, allowed_false={sorted(false_values)}, "
            f"invalid_values={sorted(invalid.unique().tolist())[:30]}, csv_rows={invalid_rows[:30]}"
        )
    return normalized.isin(true_values)


def load_feature_columns(
    path: str | Path,
    label_col: str = "label",
) -> list[str]:
    """
    feature catalog CSV에서 모델에 사용할 feature column 목록을 read하고 검증하여 반환

    feature catalog 필수 컬럼
    - column_name: 실제 parquet에 존재해야 하는 feature 컬럼명
    - used_in_ml: 모델 사용 여부
    
    동작 의도
    1. CSV를 read하여 DataFrame으로 로드
    2. 필수 컬럼이 있는지 확인
    3. used_in_ml=True인 행만 CSV 순서 그대로 선택
    4. 빈 column_name과 중복 feature를 차단
    5. label/target/누수 의심 컬럼이 feature에 들어갔는지 검사
    6. 최종 feature column list를 반환

    이 함수는 stage, feature_group, 파일명에서 feature 조합을 추론하지 않음
    """

    path = Path(path)
    feature_table = pd.read_csv(path, encoding="utf-8-sig")

    required_columns = {"column_name", "used_in_ml"}
    missing_columns = required_columns - set(feature_table.columns)
    if missing_columns:
        raise ValueError(f"Feature catalog is missing columns: {sorted(missing_columns)}")

    # feature 조합은 CSV 작성자가 직접 통제하고, 학습 모듈은 used_in_ml만 반영
    mask = used_in_ml_mask(feature_table["used_in_ml"])
    selected_names = feature_table.loc[mask, "column_name"]

    missing_names = selected_names[selected_names.isna()]
    if not missing_names.empty:
        missing_rows = (missing_names.index + 2).tolist()
        raise ValueError(f"Selected feature rows contain missing column_name. csv_rows={missing_rows[:30]}")

    feature_columns = selected_names.astype(str).str.strip().tolist()
    blank_rows = [int(row_index) + 2 for row_index, column in zip(selected_names.index, feature_columns) if not column]
    if blank_rows:
        raise ValueError(f"Selected feature rows contain blank column_name. csv_rows={blank_rows[:30]}")
    
    # 같은 feature가 두 번 들어가면 모델 입력 순서와 중요도 해석이 꼬일 수 있으므로 차단
    duplicated = sorted({column for column in feature_columns if feature_columns.count(column) > 1})
    if duplicated:
        raise ValueError(f"Duplicated selected feature columns: {duplicated}")

    validate_no_forbidden_features(feature_columns, label_col=label_col)

    if not feature_columns:
        raise ValueError(f"No usable feature columns found. path={path}")

    return feature_columns


def validate_no_forbidden_features(feature_columns: list[str], label_col: str = "label") -> None:
    """
    feature column 목록에 정답 누수 위험이 있는 이름이 들어갔는지 검사

    차단 기준
    - 정확히 금지 이름과 일치: label, target, y, is_laundering 등
    - 금지 문자열 포함: laundering, pattern, typology, attempt 등
    """


    forbidden_exact = {name.lower() for name in FORBIDDEN_EXACT_NAMES}
    forbidden_exact.add(str(label_col).strip().lower())

    violations: list[str] = []
    for column in feature_columns:
        normalized = str(column).strip().lower()
        if normalized in forbidden_exact:
            violations.append(column)
            continue
        if any(pattern in normalized for pattern in FORBIDDEN_SUBSTRINGS):
            violations.append(column)

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

    payload = json.dumps(feature_columns, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """파일 내용의 SHA256 hash를 계산한다."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found for sha256: {path}")
    digest = hashlib.sha256()
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


    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Invalid feature_columns file: {path}")
    feature_columns = [str(col) for col in columns]

    saved_hash = payload.get("feature_columns_hash")
    if saved_hash is not None and saved_hash != feature_columns_hash(feature_columns):
        raise ValueError(
            f"feature_columns hash mismatch in saved file: {path}. "
            "The feature order may have been modified."
        )
    return feature_columns


# -----------------------------------------------------------------------------
# 5. parquet 읽기 함수
# -----------------------------------------------------------------------------
def _get_pyarrow_parquet_module():
    """ pyarrow.parquet 모듈 import """
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
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Only parquet input is supported. path={path}")
    pq = _get_pyarrow_parquet_module()
    return pq.ParquetFile(path).schema_arrow.names


def read_parquet_columns(
    path: str | Path,
    columns: list[str],
    sample_rows: int | None = None,
) -> pd.DataFrame:
    """
    parquet에서 지정한 컬럼만 read, 필요시 앞에서 N개 row만 read 하여 DataFrame으로 반환

    매개변수
    - path: parquet 파일 경로
    - columns: 읽을 컬럼명 목록
    - sample_rows: None이면 전체 row, 정수이면 앞에서 해당 개수만 읽음

    동작 의도
    - 전체 컬럼을 읽지 않고 feature + label + split 정도만 읽어 메모리 사용절감 
    - sample_rows는 빠른 테스트용, (데이터가 불균형하면 앞부분 샘플에 한 클래스만 들어갈 수 있음)

    주의
    - sample_rows=None일 때는 pandas.read_parquet을 직접 사용
    - sample_rows가 지정되면 pyarrow의 batch iterator로 필요한 row 수만큼 읽음 
    """

    path = Path(path)
    if sample_rows is None:
        return pd.read_parquet(path, columns=columns)

    if sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer.")

    pq = _get_pyarrow_parquet_module()
    parquet_file = pq.ParquetFile(path)
    remaining = int(sample_rows)
    frames: list[pd.DataFrame] = []
    batch_size = min(remaining, 65_536)

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if len(frame) > remaining:
            frame = frame.iloc[:remaining]
        frames.append(frame)
        remaining -= len(frame)
        if remaining <= 0:
            break

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)

# -----------------------------------------------------------------------------
# 6. X, y 검증 함수
# -----------------------------------------------------------------------------
def validate_features(
    x: pd.DataFrame,
    source_path: str | Path,
    allow_nan: bool = False,
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

    if x.empty:
        raise ValueError(f"Feature matrix is empty. source={source_path}")

    non_numeric = [column for column in x.columns if not pd.api.types.is_numeric_dtype(x[column])]
    if non_numeric:
        raise ValueError(f"All features must be numeric. source={source_path}, non_numeric={non_numeric}")

    if not allow_nan:
        missing_counts = x.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            raise ValueError(
                f"Feature matrix contains NaN values. source={source_path}, "
                f"missing_counts={missing_counts.head(30).to_dict()}"
            )

    infinite_counts = np.isinf(x.to_numpy(dtype="float64", copy=False)).sum()
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


    if y.empty:
        raise ValueError(f"Label vector is empty. source={source_path}")

    if y.isna().any():
        raise ValueError(f"Labels contain NaN. source={source_path}, label_col={label_col}")

    values = set(y.unique().tolist())
    if not values <= {0, 1}:
        raise ValueError(f"Labels must be binary 0/1. source={source_path}, label_col={label_col}, values={sorted(values)}")

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

    counts = y.value_counts().sort_index().to_dict()
    total = int(len(y))
    positive_count = int(counts.get(1, 0))
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

    if expected_split is None:
        return
    if "split" not in df.columns:
        raise ValueError(
            "Input parquet is missing required split column. "
            f"source={source_path}, expected_split={expected_split!r}"
        )

    raw_split = df["split"]
    if raw_split.isna().any():
        raise ValueError(
            "Split column contains missing values. "
            f"source={source_path}, expected_split={expected_split!r}, "
            f"missing_count={int(raw_split.isna().sum())}"
        )

    normalized = raw_split.astype("string").str.strip().str.lower()
    blank_mask = normalized == ""
    if blank_mask.any():
        raise ValueError(
            "Split column contains blank values. "
            f"source={source_path}, expected_split={expected_split!r}, blank_count={int(blank_mask.sum())}"
        )

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
    if label_col in feature_columns:
        raise ValueError(
            f"Data leakage risk: label_col={label_col!r} is included in feature_columns."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns contains duplicated names.")

    required_columns = list(feature_columns) + [label_col]
    available_columns = get_parquet_columns(path)
    missing = [column for column in required_columns if column not in available_columns]
    if missing:
        raise ValueError(
            f"Input parquet is missing required columns. path={path}, "
            f"missing={missing[:30]}, missing_count={len(missing)}"
        )

    read_columns = list(required_columns)
    if expected_split is not None:
        if "split" not in available_columns:
            raise ValueError(
                "Input parquet is missing required split column. "
                f"path={path}, expected_split={expected_split!r}"
            )
        if "split" not in read_columns:
            read_columns.append("split")

    df = read_parquet_columns(path, read_columns, sample_rows)
    validate_split_column(df, expected_split=expected_split, source_path=path)
    
    
    # int8 변환 전에 원본 label 값을 검증해 0.5 같은 값이 0으로 잘리는 문제를 차단한다.
    raw_y = pd.to_numeric(df[label_col], errors="raise")
    validate_labels(raw_y, path, label_col=label_col, sample_rows=sample_rows)
    y = raw_y.astype("int8")
    # feature_columns 순서를 그대로 유지하는 것이 중요, 모델 입력이 이 순서를 기준으로 해석되기 때문
    x = df[feature_columns].copy()

    validate_features(x, path, allow_nan=allow_nan)
    return x, y

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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """
    UTF-8 JSON 파일을 읽어 Python 객체로 반환합니다.
    주의
    - 타입 힌트는 dict[str, Any]이지만, JSON 파일 내용이 list이면 실제 반환값도 list가 됨
    - 반드시 dict만 허용해야 하는 상황이라면 isinstance(result, dict) 검사를 추가하는 것이 안전
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
