"""
Feature build 입출력 유틸리티 모듈

이 파일의 역할
----------------
1. feature build 기본 입력 경로를 제공한다.
2. 사용자가 입력한 절대경로/상대경로를 실제 사용할 절대경로로 변환한다.
3. parquet 파일에서 필요한 컬럼만 읽는다.
4. DataFrame/JSON 저장 함수를 한곳에 모아 저장 방식을 통일한다.

경로 처리 원칙
--------------
- 절대경로는 그대로 정규화해서 사용한다.
- 상대경로와 base_dir이 함께 들어오면 base_dir 기준으로 해석한다.
- 상대경로인데 base_dir이 없으면 `ml_02_fb_utils.BASE_DIR`, 즉 Git 프로젝트 루트 기준으로 해석한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import pandas as pd
import pyarrow.parquet as pq

from ml_02_fb_utils import BASE_DIR, PROCESSED_DIR


# -----------------------------------------------------------------------------
# 1. 기본 입력/출력 경로
# -----------------------------------------------------------------------------
# 기본 입력은 전처리 완료 산출물인 clean_base.parquet이다.
# 사용자가 FeatureBuildConfig(input_path=...)를 넘기면 이 기본값은 대체된다.
DEFAULT_INPUT_PATH = PROCESSED_DIR / "step01_clean_base" / "clean_base.parquet"

# -----------------------------------------------------------------------------
# 2. 경로 해석
# -----------------------------------------------------------------------------
def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    입력 경로를 실제 사용할 절대경로로 변환
    동작 의도
    - 노트북에서 절대경로를 넘기면 그대로 사용
    - 노트북에서 상대경로를 넘기면 base_dir 또는 Git 루트 기준으로 안전하게 해석
    - cwd에 의존하지 않아 실행 위치가 바뀌어도 같은 파일을 찾게 함
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    base = BASE_DIR if base_dir is None else Path(base_dir).expanduser().resolve()
    return (base / candidate).resolve()


# -----------------------------------------------------------------------------
# 3. parquet 로드
# -----------------------------------------------------------------------------
def parquet_columns(path: Union[str, Path]) -> list[str]:
    """parquet 파일을 전체 로드하지 않고 schema에서 컬럼명만 읽는다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    # pyarrow.parquet.ParquetFile 객체를 생성하여 파일의 메타데이터(Footer)를 로드한 뒤,
    # PyArrow 포맷의 스키마(schema_arrow)에서 전체 필드명(.names)을 추출하여 파이썬 리스트로 반환
    return list(pq.ParquetFile(path).schema_arrow.names)


def parquet_row_count(path: Union[str, Path]) -> int:
    """parquet 파일을 전체 로드하지 않고 row 수를 반환한다."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def parquet_schema_types(path: Union[str, Path]) -> dict[str, str]:
    """parquet 파일을 전체 로드하지 않고 schema의 컬럼별 타입을 반환한다."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}


def load_parquet_columns(
    path: Union[str, Path],
    columns: Iterable[str],
    sample_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    parquet 파일에서 필요한 컬럼만 읽는다.
    동작 방식
    - sample_rows=None이면 지정 컬럼 전체를 읽는다.
    - sample_rows가 정수이면 parquet 앞쪽 첫 batch를 최대 sample_rows행만 읽는다.
    - sample_rows는 랜덤 샘플이 아니라 파일 앞부분 sample이다.
    왜 필요한가
    - feature build는 선택된 FeatureSpec이 요구하는 컬럼만 필요하므로 불필요한 컬럼을 읽지 않아 메모리를 절약한다.
    - sample smoke build에서는 full load를 피하기 위해 첫 batch만 읽는 옵션을 제공한다.

    주의
    - sample_rows는 시간순/랜덤 split 샘플이 아니라 parquet 물리 순서 앞부분이다.
    - sample 결과는 smoke/debug 용도이며 feature 분포나 모델 성능 판단에 쓰면 안 된다.
    """

    path = Path(path)

    # 파일이 없으면 pandas/pyarrow 내부 에러까지 가지 않고 명시적으로 중단
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")

    # sample_rows는 전체 parquet를 다 읽지 않고 앞부분 일부 행만 읽어 빠르게 확인하는 옵션
    # 0 또는 음수는 의미가 없으므로 설정 오류로 간주
    if sample_rows is not None and sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer or None.")

    # columns에는 중복 컬럼이 들어올 수 있으므로 순서를 유지한 채 중복 제거
    selected_columns = list(dict.fromkeys(str(column) for column in columns))

    # 읽을 컬럼이 없다는 것은 FeatureSpec 또는 컬럼 resolve 단계의 오류, 명시적으로 중단
    if not selected_columns:
        raise ValueError("columns must not be empty.")

    # full build: 필요한 컬럼 전체를 read_parquet으로 읽는다.
    # 대용량 입력에서는 이 지점이 메모리/시간 병목이 될 수 있으므로 노트북에서 sample_rows smoke run을 먼저 권장한다.
    if sample_rows is None:
        return pd.read_parquet(path, columns=selected_columns)

    # sample build: 대용량 parquet 전체를 로드하지 않고 첫 batch만 read
    parquet_file = pq.ParquetFile(path)
    batches = parquet_file.iter_batches(batch_size=sample_rows, columns=selected_columns)
    try:
        first_batch = next(batches)
    except StopIteration as exc:
        # parquet 파일은 존재하지만 row가 없는 경우 명시적으로 중단한다.
        raise ValueError(f"parquet file has no rows: {path}") from exc
    # pyarrow RecordBatch를 pandas DataFrame으로 변환해 이후 pandas 기반 operation에서 사용한다.
    return first_batch.to_pandas()


# -----------------------------------------------------------------------------
# 4. 저장 함수
# -----------------------------------------------------------------------------
def save_json(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    """
    dict 형태 메타데이터를 UTF-8 JSON으로 저장한다.
    사용 위치
    ---------
    - build_summary.json 저장에 사용
    - build_summary에는 입력 경로, 출력 경로, feature 목록, row count 같은 재현성 메타데이터가 들어감
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dict(payload), file, ensure_ascii=False, indent=2)


def save_dataframe_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    DataFrame을 CSV로 저장
    사용 위치
    ---------
    - feature_contract.csv
    - feature_catalog.csv
    - split_summary.csv
    - feature_info.csv
    - category_mapping_train_only.csv
    - category_unknown_summary.csv
    CSV로 저장하는 이유
    ------------------
    사람이 Excel, pandas, 텍스트 에디터로 바로 확인하기 쉽기 때문이다.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_dataframe_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    DataFrame을 parquet로 저장
    사용 위치
    ---------
    - train feature parquet
    - validation feature parquet
    - test feature parquet
    parquet로 저장하는 이유
    ----------------------
    CSV보다 dtype 보존이 좋고, 대용량 테이블을 저장/로드하기에 적합
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def utc_now_iso() -> str:
    """
    build_summary.json에 기록할 UTC 생성 시각 문자열을 만든다.
    UTC를 쓰는 이유
    ---------------
    실행 환경마다 로컬 timezone이 다를 수 있으므로,
    재현성 메타데이터에는 timezone이 명확한 UTC 시간을 남긴다.
    """

    return datetime.now(timezone.utc).isoformat()
