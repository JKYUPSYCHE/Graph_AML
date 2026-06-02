"""ML-05 피처 생성 입력값 검증 헬퍼.

코드 맵:
- 입력      : split 정보가 포함된 DataFrame의 메타데이터 컬럼.
- 출력      : tx_id/timestamp/label/split 기준의 표준 메타데이터 frame.
- 공개 함수 : existing_split_metadata_frame, validate_time_split, validate_unique_tx_ids.
- 누수 방지 : train의 최대 timestamp는 val보다 이전이어야 하고, val은 test보다 이전이어야 한다.
- 참고      : ML-05는 기존 split을 사용하며 새 split을 만들지 않는다.
"""

from __future__ import annotations  
from pathlib import Path  
import pandas as pd  
from ml_05_fb_schema import parse_datetime_series_strict, parse_numeric_series_strict  

def validate_unique_tx_ids(df: pd.DataFrame) -> None:
    """Ensure tx_id is unique across the combined split frame."""

    duplicated = df["tx_id"].astype("string").duplicated(keep=False)          # 전체 split에서 중복된 tx_id 여부를 표시한다.
    if duplicated.any():                                                      # 중복 tx_id가 하나라도 있으면 실패시킨다.
        examples = df.loc[duplicated, "tx_id"].astype(str).head(10).tolist()  # 디버깅용 중복 tx_id 예시를 최대 10개 수집한다.
        raise ValueError(
            "Feature build failed: tx_id values are duplicated across split files. "
            f"duplicated_count={int(duplicated.sum())}, examples={examples}"
        )


def validate_time_split(df: pd.DataFrame) -> None:
    """Ensure train < val < test temporal split ordering."""

    counts = df["split"].value_counts().to_dict()     # split 값별 row 수를 집계한다.
    missing = {"train", "val", "test"} - set(counts)  # 필수 split 중 누락된 값을 찾는다.
    if missing:                                       # train/val/test 중 하나라도 없으면 기존 split을 사용할 수 없다.
        raise ValueError(f"Missing required split values in existing split column: {sorted(missing)}")

    train_max = df.loc[df["split"] == "train", "timestamp"].max()  # train 구간의 가장 늦은 timestamp.
    val_min = df.loc[df["split"] == "val", "timestamp"].min()      # val 구간의 가장 이른 timestamp.
    val_max = df.loc[df["split"] == "val", "timestamp"].max()      # val 구간의 가장 늦은 timestamp.
    test_min = df.loc[df["split"] == "test", "timestamp"].min()    # test 구간의 가장 이른 timestamp.
    if train_max >= val_min:  # train 시간이 val과 겹치거나 뒤에 있으면 시간 누수 위험이 있다.
        raise ValueError(f"Time split boundary violation: train_max={train_max}, val_min={val_min}")
    if val_max >= test_min:  # val 시간이 test와 겹치거나 뒤에 있으면 시간 누수 위험이 있다.
        raise ValueError(f"Time split boundary violation: val_max={val_max}, test_min={test_min}")



def normalize_existing_split_values(series: pd.Series, *, source_path: Path, split_col: str) -> pd.Series:
    """Normalize and validate existing split values."""

    if series.isna().any():  # split 값이 비어 있으면 해당 row의 학습/평가 구간을 알 수 없다.
        raise ValueError(
            "Existing split column has missing values. "
            f"path={source_path}, split_col={split_col!r}, missing_count={int(series.isna().sum())}"
        )
    normalized = series.astype("string").str.strip().str.lower()  # split 값을 문자열로 바꾸고 공백/대소문자를 정규화한다.
    blank_mask = normalized == ""                                 # 공백 제거 후 빈 문자열인 split 값을 찾는다.
    if blank_mask.any():                                          # 빈 split 값은 실패시킨다.
        raise ValueError(
            "Existing split column has blank values. "
            f"path={source_path}, split_col={split_col!r}, blank_count={int(blank_mask.sum())}"
        )
    allowed = {"train", "val", "test"}               # 허용되는 split 값 목록.
    invalid = normalized[~normalized.isin(allowed)]  # 허용 목록에 없는 split 값을 추출한다.
    if not invalid.empty:                            # train/val/test 외 값이 있으면 기존 split 계약을 위반한 것이다.
        raise ValueError(
            "Existing split column has unsupported values. "
            f"path={source_path}, split_col={split_col!r}, allowed={sorted(allowed)}, "
            f"observed_examples={sorted(invalid.unique().tolist())[:20]}"
        )
    return normalized  # 검증과 정규화가 끝난 split Series를 반환한다.


def existing_split_metadata_frame(
    df: pd.DataFrame,
    *,
    source_path: Path,
    tx_id_col: str,
    timestamp_col: str,
    label_col: str,
    split_col: str,
) -> pd.DataFrame:
    """Build and validate canonical split metadata."""

    required = {tx_id_col, timestamp_col, label_col, split_col}  # 기존 split 검증에 반드시 필요한 입력 컬럼 집합.
    missing = required - set(df.columns)                         # 입력 DataFrame에 없는 필수 컬럼을 찾는다.
    if missing:  # 필수 컬럼이 빠져 있으면 표준 메타데이터를 만들 수 없다.
        raise ValueError(
            "Single parquet input is missing columns required for existing split validation. "
            f"path={source_path}, missing={sorted(missing)}"
        )

    tx_id = df[tx_id_col]   # 설정된 tx_id 컬럼을 가져온다.
    if tx_id.isna().any():  # tx_id가 비어 있으면 거래 단위 식별이 불가능하다.
        raise ValueError(
            "tx_id column has missing values. "
            f"path={source_path}, tx_id_col={tx_id_col!r}, missing_count={int(tx_id.isna().sum())}"
        )

    raw_timestamp = df[timestamp_col]          # 설정된 timestamp 원본 컬럼을 가져온다.
    timestamp = parse_datetime_series_strict(  # timestamp를 datetime으로 엄격하게 변환한다.
        raw_timestamp,
        missing_message=lambda missing_count: (  # timestamp 결측치가 있을 때 사용할 에러 메시지.
            "timestamp column has missing values. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, missing_count={missing_count}"
        ),
        failed_message=lambda failed_count, examples: (  # timestamp 파싱 실패 값이 있을 때 사용할 에러 메시지.
            "timestamp parsing failed for existing split validation. "
            f"path={source_path}, timestamp_col={timestamp_col!r}, failed_count={failed_count}, "
            f"example_values={examples}"
        ),
    )

    raw_label = df[label_col]             # 설정된 label 원본 컬럼을 가져온다.
    label = parse_numeric_series_strict(  # label을 숫자로 엄격하게 변환한다.
        raw_label,
        missing_message=lambda missing_count: (  # label 결측치가 있을 때 사용할 에러 메시지.
            "label column has missing values. "
            f"path={source_path}, label_col={label_col!r}, missing_count={missing_count}"
        ),
        failed_message=lambda failed_count, examples: (  # label 숫자 변환 실패 값이 있을 때 사용할 에러 메시지.
            "label parsing failed for existing split validation. "
            f"path={source_path}, label_col={label_col!r}, failed_count={failed_count}, "
            f"example_values={examples}"
        ),
    )
    label_values = sorted(label.dropna().unique().tolist())  # label에 실제로 존재하는 고유 값을 정렬해 확인한다.
    if not set(label_values).issubset({0, 1}):               # AML 이진 분류 계약상 label은 0/1만 허용한다.
        raise ValueError(
            "label must be binary 0/1 for existing split validation. "
            f"path={source_path}, label_col={label_col!r}, observed_values={label_values[:20]}"
        )

    metadata = pd.DataFrame(  # 이후 build 로직이 공통으로 쓰는 표준 메타데이터 frame을 만든다.
        {
            "tx_id": tx_id.reset_index(drop=True),                 # 거래 ID를 표준 컬럼명 tx_id로 맞추고 index를 재정렬한다.
            "timestamp": timestamp.reset_index(drop=True),         # 파싱된 timestamp를 표준 컬럼명 timestamp로 저장한다.
            "label": label.astype("int8").reset_index(drop=True),  # label을 int8 타입의 표준 label 컬럼으로 저장한다.
            "split": normalize_existing_split_values(df[split_col], source_path=source_path, split_col=split_col).reset_index(drop=True),  # split 값을 train/val/test 기준으로 정규화한다.
        }
    )
    validate_unique_tx_ids(metadata)  # 전체 split에서 tx_id 중복이 없는지 검증한다.
    validate_time_split(metadata)     # train < val < test 시간 순서가 지켜졌는지 검증한다.
    return metadata                   # 검증이 끝난 표준 split 메타데이터를 반환한다.
