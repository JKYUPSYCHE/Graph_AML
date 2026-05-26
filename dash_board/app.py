"""
돈무브 프로젝트 대시보드

Secrets (.streamlit/secrets.toml):
    PROJECT_FOLDER_ID           Drive root folder ID
    [gcp_service_account]       서비스 계정 JSON 내용

Drive layout (PROJECT_FOLDER_ID 하위):
    ml/
        ml_leaderboard_representatives.json
        ml-01/  ml-02/ ...
            {run_id}/
                {exp}__{run}__{model}_metrics_val.json
                {exp}__{run}__{model}_train_summary.json
                {exp}__{run}__{model}_feature_importance.csv
                {exp}__{run}__{model}_confusion_matrix_val.csv
                {exp}__{run}__{model}_feature_columns.json
            {mlNN}_feature_catalog.csv

    data/ml/
        woe_iv/
            woe_iv_cache.json
            ml-01/
                iv_summary.json  bin_table.json  woe_meta.json
            ml-02/ ...
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from io import BytesIO, StringIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

st.set_page_config(page_title="돈무브 프로젝트 대시보드", layout="wide", page_icon="💱")

st.markdown("""
<style>
/* ── 전체 여백 ── */
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* ── 제목 ── */
h1 { font-size: 1.6rem !important; font-weight: 700; letter-spacing: -0.3px; }
h4, h5 { font-weight: 600; letter-spacing: -0.2px; }

/* ── 탭 ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #2c2f3e;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.45rem 1rem;
    border-radius: 6px 6px 0 0;
    color: #8b90a0;
    background: transparent;
    border: none;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #4f9cf9 !important;
    border-bottom: 2px solid #4f9cf9 !important;
    background: transparent !important;
}

/* ── st.metric 카드 ── */
[data-testid="stMetric"] {
    background: #1e2130;
    border: 1px solid #2c2f3e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; color: #8b90a0; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 600; color: #e2e5ec; }

/* ── expander ── */
[data-testid="stExpander"] {
    border: 1px solid #2c2f3e !important;
    border-radius: 8px !important;
    background: #1a1d27 !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.875rem;
    font-weight: 500;
    color: #b0b5c3;
}

/* ── 구분선 ── */
hr { border: none; border-top: 1px solid #2c2f3e; margin: 1rem 0; }

/* ── 버튼 ── */
.stButton button {
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid #2c2f3e;
    background: #1e2130;
    color: #b0b5c3;
    transition: background 0.15s, border-color 0.15s;
}
.stButton button:hover {
    background: #252838;
    border-color: #4f9cf9;
    color: #e2e5ec;
}

/* ── selectbox / text_input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    border-radius: 6px;
    border-color: #2c2f3e;
    font-size: 0.875rem;
}

/* ── caption ── */
[data-testid="stCaptionContainer"] { color: #6b7280; font-size: 0.8rem; }

/* ── progress bar ── */
[data-testid="stProgressBar"] > div { background: #4f9cf9; }
</style>
""", unsafe_allow_html=True)

API_KEY           = st.secrets.get("GOOGLE_API_KEY", "")
PROJECT_FOLDER_ID = st.secrets.get("PROJECT_FOLDER_ID", "")

IV_COLORS = {
    "suspicious": "#d62728",
    "strong":     "#ff7f0e",
    "medium":     "#2ca02c",
    "weak":       "#aec7e8",
    "useless":    "#d3d3d3",
    "na":         "#eeeeee",
}
IV_CUT = 1.5


# ── Drive helpers ──────────────────────────────────────────────────────────

def _drive_list(q: str) -> list[dict]:
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": q,
            "fields": "files(id,name)",
            "key": API_KEY,
            "orderBy": "name",
            "pageSize": 200,
        },
        timeout=15,
    )
    if not r.ok:
        st.error(f"Drive API 오류 {r.status_code}: {r.text[:300]}")
        st.stop()
    return r.json().get("files", [])


@st.cache_data(ttl=3600)
def _download_json(file_id: str) -> object:
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=3600)
def _download_csv(file_id: str) -> pd.DataFrame:
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=30,
    )
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content), encoding="utf-8-sig")


# ── Report: Drive write helpers (서비스 계정) ──────────────────────────────

def _get_sa_token() -> str:
    """서비스 계정으로 Drive 쓰기 토큰 발급. 만료 5분 전 자동 갱신."""
    import time
    now = time.time()
    if st.session_state.get("sa_token") and now < st.session_state.get("sa_token_exp", 0) - 300:
        return st.session_state["sa_token"]
    try:
        from google.oauth2 import service_account
        import google.auth.transport.requests as ga_requests
        sa_info = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/drive"]
        )
        creds.refresh(ga_requests.Request())
        st.session_state["sa_token"]     = creds.token
        st.session_state["sa_token_exp"] = creds.expiry.timestamp() if creds.expiry else now + 3600
        st.session_state.pop("_sa_last_error", None)
        return creds.token
    except Exception as e:
        st.session_state["_sa_last_error"] = f"토큰 발급 실패: {e}"
        return ""


def _sa_find_folder(name: str, parent_id: str, token: str) -> str:
    cache = st.session_state.setdefault("_sa_folder_cache", {})
    key = f"{parent_id}/{name}"
    if key in cache:
        return cache[key]
    q = (f"'{parent_id}' in parents and name='{name}'"
         " and mimeType='application/vnd.google-apps.folder' and trashed=false")
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": q, "fields": "files(id)", "pageSize": 1},
        timeout=15,
    )
    files = r.json().get("files", []) if r.ok else []
    result = files[0]["id"] if files else ""
    if result:
        cache[key] = result
    return result


def _sa_create_folder(name: str, parent_id: str, token: str) -> str:
    r = requests.post(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]},
        timeout=15,
    )
    folder_id = r.json().get("id", "") if r.ok else ""
    if folder_id:
        st.session_state.setdefault("_sa_folder_cache", {})[f"{parent_id}/{name}"] = folder_id
    return folder_id


def _sa_get_or_create_folder(name: str, parent_id: str, token: str) -> str:
    return _sa_find_folder(name, parent_id, token) or _sa_create_folder(name, parent_id, token)


def _sa_find_file(name: str, parent_id: str, token: str) -> str:
    cache = st.session_state.setdefault("_sa_file_cache", {})
    key = f"{parent_id}/{name}"
    if key in cache:
        return cache[key]
    q = f"'{parent_id}' in parents and name='{name}' and trashed=false"
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": q, "fields": "files(id)", "pageSize": 1},
        timeout=15,
    )
    files = r.json().get("files", []) if r.ok else []
    result = files[0]["id"] if files else ""
    if result:
        cache[key] = result
    return result


def _load_report(tab_name: str, exp_name: str) -> dict:
    """Drive dashboard/{tab_name}/{exp_name}/report.json 로드. 없으면 빈 dict."""
    token = _get_sa_token()
    if not token:
        return {}
    dash_id = _sa_find_folder("dashboard", PROJECT_FOLDER_ID, token)
    if not dash_id:
        return {}
    tab_id = _sa_find_folder(tab_name, dash_id, token)
    if not tab_id:
        return {}
    exp_id = _sa_find_folder(exp_name, tab_id, token)
    if not exp_id:
        return {}
    file_id = _sa_find_file("report.json", exp_id, token)
    if not file_id:
        return {}
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=15,
    )
    try:
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _save_report(tab_name: str, exp_name: str, content: str, author: str) -> bool:
    """Drive dashboard/{tab_name}/{exp_name}/report.json 저장/덮어쓰기."""
    token = _get_sa_token()
    if not token:
        return False
    dash_id = _sa_get_or_create_folder("dashboard", PROJECT_FOLDER_ID, token)
    tab_id  = _sa_get_or_create_folder(tab_name,    dash_id,            token)
    exp_id  = _sa_get_or_create_folder(exp_name,    tab_id,             token)
    if not exp_id:
        st.session_state["_sa_last_error"] = "Drive 폴더 생성 실패 (403 권한 오류일 가능성 높음) — 서비스 계정을 PROJECT_FOLDER_ID 폴더에 Editor로 공유했는지 확인하세요."
        return False

    payload = json.dumps(
        {"content": content, "author": author,
         "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M")},
        ensure_ascii=False, indent=2,
    ).encode("utf-8")

    file_id = _sa_find_file("report.json", exp_id, token)
    if not file_id:
        # 서비스 계정은 개인 Drive에 파일을 새로 생성할 수 없음 (스토리지 쿼터 없음).
        # report.json 플레이스홀더를 Drive에서 직접 만들어 두면 이후 PATCH로 저장 가능.
        st.session_state["_sa_last_error"] = (
            f"report.json 파일이 없습니다. "
            f"Drive에서 dashboard › {tab_name} › {exp_name} 폴더 안에 "
            f"report.json 파일을 직접 만들어 주세요 (내용: {{}})."
        )
        return False

    r = requests.patch(
        f"https://www.googleapis.com/upload/drive/v3/files/{file_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        params={"uploadType": "media"},
        data=payload,
        timeout=15,
    )
    if not r.ok:
        st.session_state["_sa_last_error"] = f"Drive API {r.status_code}: {r.text[:300]}"
    return r.ok


@st.fragment
def _render_report(tab_name: str, exp_name: str) -> None:
    """실험 선택창 아래 리포트 섹션 렌더링."""
    from streamlit_ace import st_ace

    VALID_AUTHORS: list[str] = list(st.secrets.get("REPORT_AUTHORS", []))
    sess_author = st.session_state.get("report_author", "")
    cache_key   = f"rpt_{tab_name}_{exp_name}"
    edit_key    = f"rpt_editing_{tab_name}_{exp_name}"

    # 리포트 캐시 최대 20개 유지 (오래된 것 제거)
    rpt_keys = [k for k in st.session_state if k.startswith("rpt_") and not k.startswith("rpt_editing_")]
    if len(rpt_keys) > 20:
        for old in rpt_keys[:-20]:
            st.session_state.pop(old, None)

    if cache_key not in st.session_state:
        with st.spinner("리포트 로드 중..."):
            st.session_state[cache_key] = _load_report(tab_name, exp_name)
    report     = st.session_state.get(cache_key, {})
    content    = report.get("content", "")
    written_by = report.get("author", "")
    updated_at = report.get("updated_at", "")
    is_editing = st.session_state.get(edit_key, False)

    # ── 제목 + 편집 버튼 ───────────────────────────────────────────────────
    col_title, col_btn = st.columns([8, 1])
    col_title.markdown("##### Report")
    if not is_editing:
        if col_btn.button("편집", key=f"rpt_edit_btn_{tab_name}_{exp_name}", use_container_width=True):
            st.session_state[edit_key] = True
            st.rerun()

    # ── 읽기 모드 ──────────────────────────────────────────────────────────
    if not is_editing:
        if content:
            st.markdown(content)
            st.caption(f"작성자: {written_by}  ({updated_at})")
        else:
            st.caption("작성된 리포트가 없습니다.")

    # ── 편집 모드 ──────────────────────────────────────────────────────────
    else:
        if not sess_author:
            # 비밀번호 입력
            st.caption("비밀번호를 입력하세요")
            name_in = st.text_input(
                "비밀번호", key=f"rpt_auth_{tab_name}_{exp_name}",
                placeholder="비밀번호 입력", label_visibility="collapsed",
            )
            col_ok, col_cancel = st.columns(2)
            if col_ok.button("확인", key=f"rpt_auth_btn_{tab_name}_{exp_name}", use_container_width=True):
                if name_in in VALID_AUTHORS:
                    st.session_state["report_author"] = name_in
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
            if col_cancel.button("취소", key=f"rpt_cancel_auth_{tab_name}_{exp_name}", use_container_width=True):
                st.session_state[edit_key] = False
                st.rerun()
        else:
            # ace 에디터 + 실시간 미리보기
            st.caption(f"편집 중: {sess_author}")
            col_edit, col_preview = st.columns(2)
            with col_edit:
                st.caption("편집 (Markdown)")
                new_content = st_ace(
                    value=content,
                    language="markdown",
                    theme="tomorrow_night",
                    font_size=14,
                    min_lines=15,
                    wrap=True,
                    auto_update=True,
                    key=f"rpt_ace_{tab_name}_{exp_name}",
                )
            with col_preview:
                st.caption("미리보기")
                st.markdown(new_content if new_content else "_내용 없음_")

            col_save, col_cancel = st.columns(2)
            if col_save.button("저장", key=f"rpt_save_{tab_name}_{exp_name}", use_container_width=True):
                with st.spinner("저장 중..."):
                    ok = _save_report(tab_name, exp_name, new_content, sess_author)
                if ok:
                    st.session_state.pop(cache_key, None)
                    st.session_state[edit_key] = False
                    st.rerun()
                else:
                    err = st.session_state.pop("_sa_last_error", "서비스 계정 설정을 확인하세요.")
                    st.error(f"저장 실패 — {err}")
            if col_cancel.button("취소", key=f"rpt_cancel_{tab_name}_{exp_name}", use_container_width=True):
                st.session_state[edit_key] = False
                st.session_state.pop("report_author", None)
                st.rerun()


@st.cache_data(ttl=3600)
def _get_folder_id(parent_id: str, name: str) -> str:
    folders = _drive_list(
        f"'{parent_id}' in parents"
        f" and name='{name}'"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    return folders[0]["id"] if folders else ""


@st.cache_data(ttl=3600)
def _list_files(folder_id: str) -> dict[str, str]:
    return {f["name"]: f["id"] for f in _drive_list(f"'{folder_id}' in parents and trashed=false")}


# ── UI helpers ────────────────────────────────────────────────────────────

def _metric(container, label: str, value: str, sub: str = "") -> None:
    sub_html = (f" <span style='font-size:0.78rem;font-weight:400;color:#888'>{sub}</span>"
                if sub else "")
    container.markdown(
        f"<div style='font-size:0.875rem;margin-bottom:2px'>{label}</div>"
        f"<div style='font-size:1.75rem;font-weight:600;line-height:1.2'>{value}{sub_html}</div>",
        unsafe_allow_html=True,
    )


# ── Experiment helpers ─────────────────────────────────────────────────────

def _artifact_prefix(rep: dict) -> str:
    return f"{rep['experiment_id']}__{rep['run_id']}__{rep['model_run_id']}"


def _woe_iv_folder_name(ml_folder: str) -> str:
    m = re.match(r"(ml-\d+)", ml_folder)
    return m.group(1) if m else ml_folder


def _catalog_filename(ml_folder: str) -> str:
    return _woe_iv_folder_name(ml_folder).replace("-", "") + "_feature_catalog.csv"

def _is_valid_rep(rep: dict) -> bool:
    woe_folder = _woe_iv_folder_name(rep.get("ml_folder", ""))
    return bool(re.match(r"ml-\d+", woe_folder)) and woe_folder != "ml-00"


def _exp_label(rep: dict) -> str:
    note = rep.get("note", "")
    return f"{_woe_iv_folder_name(rep['ml_folder'])}  {('— ' + note) if note else ''}".strip()


# ── Data loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_representatives(ml_folder_id: str) -> list[dict]:
    files = _drive_list(
        f"'{ml_folder_id}' in parents"
        " and name='ml_leaderboard_representatives.json'"
        " and trashed=false"
    )
    return _download_json(files[0]["id"]) if files else []


@st.cache_data(ttl=3600)
def _load_ml_results(folder_id: str, prefix: str) -> dict:
    fm  = _list_files(folder_id)
    out: dict = {}
    if f"{prefix}_metrics_val.json"                in fm: out["metrics"]           = _download_json(fm[f"{prefix}_metrics_val.json"])
    if f"{prefix}_train_summary.json"              in fm: out["train_summary"]     = _download_json(fm[f"{prefix}_train_summary.json"])
    if f"{prefix}_scores_train_summary.json"       in fm: out["scores_summary"]    = _download_json(fm[f"{prefix}_scores_train_summary.json"])
    if f"{prefix}_feature_importance.csv"          in fm: out["feature_importance"] = _download_csv(fm[f"{prefix}_feature_importance.csv"])
    if f"{prefix}_confusion_matrix_val.csv"        in fm: out["confusion_matrix"]   = _download_csv(fm[f"{prefix}_confusion_matrix_val.csv"])
    if f"{prefix}_feature_assoc_mixed_val.json"    in fm: out["feature_assoc"]     = _download_json(fm[f"{prefix}_feature_assoc_mixed_val.json"])
    elif f"{prefix}_feature_assoc_mixed_train.json" in fm: out["feature_assoc"]    = _download_json(fm[f"{prefix}_feature_assoc_mixed_train.json"])
    return out


@st.cache_data(ttl=3600)
def _get_woe_iv_root_id(project_folder_id: str) -> str:
    data_id = _get_folder_id(project_folder_id, "data")
    if not data_id: return ""
    ml_id = _get_folder_id(data_id, "ml")
    if not ml_id:   return ""
    return _get_folder_id(ml_id, "woe_iv")


@st.cache_data(ttl=3600)
def _load_woe_results(woe_iv_folder_id: str) -> dict:
    if not woe_iv_folder_id:
        return {}
    fm  = _list_files(woe_iv_folder_id)
    out: dict = {}
    if "iv_summary.json" in fm:
        out["iv_df"] = pd.DataFrame(_download_json(fm["iv_summary.json"]))
    if "bin_table.json" in fm:
        out["bin_df"] = pd.DataFrame(_download_json(fm["bin_table.json"]))
    if "woe_meta.json" in fm:
        out["meta"] = _download_json(fm["woe_meta.json"])
    return out


def _read_catalog_bytes(content: bytes) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "cp949", "utf-8", "euc-kr"):
        try:
            df = pd.read_csv(BytesIO(content), encoding=enc)
            df.columns = df.columns.str.strip()
            if "피처명" in df.columns:
                return df
            if "피쳐명" in df.columns:
                return df.rename(columns={"피쳐명": "피처명"})
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600)
def _load_catalog(folder_id: str, catalog_fn: str) -> pd.DataFrame | None:
    if not folder_id:
        return None
    fm = _list_files(folder_id)
    if catalog_fn not in fm:
        return None
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={fm[catalog_fn]}",
        timeout=30,
    )
    if not r.ok:
        return None
    return _read_catalog_bytes(r.content)


# ── GNN helpers ────────────────────────────────────────────────────────────

def _parse_gnn_log(text: str) -> dict:
    train_re = re.compile(
        r"Train F1:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|\s*Precision:\s*([\d.]+)\s*\|\s*AUPRC:\s*([\d.]+)"
        r"(?:\s*\|\s*LogLoss:\s*([\d.]+))?"
    )
    val_re = re.compile(
        r"Val\s+[—\-]\s*F1:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|\s*Precision:\s*([\d.]+)\s*\|\s*AUPRC:\s*([\d.]+)"
        r"(?:\s*\|\s*LogLoss:\s*([\d.]+))?"
    )
    test_re = re.compile(
        r"Test\s+[—\-]\s*F1:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*\|\s*Precision:\s*([\d.]+)\s*\|\s*AUPRC:\s*([\d.]+)"
        r"(?:\s*\|\s*LogLoss:\s*([\d.]+))?"
    )
    nodes_re        = re.search(r"Number of nodes = (\d+)", text)
    txns_re         = re.search(r"Number of transactions = (\d+)", text)
    ir_re           = re.search(r"Illicit ratio = \d+ / \d+ = ([\d.]+)%", text)
    edge_re         = re.search(r"Edge features being used: (\[.*?\])", text)
    params_re       = re.search(r"GraphModule\s+\|[^|]+\|[^|]+\|\s*([\d,]+)", text)
    time_re         = re.search(r"Total training time:\s*([\d.]+)s", text)
    best_epoch_re   = re.search(r"Best epoch:\s*(\d+)", text)
    early_stop_re   = re.search(r"Early stopping at epoch\s*(\d+)", text)
    xai_cm_re       = re.search(r"\[XAI\].*?TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+TN=(\d+)", text)

    epochs: list[dict] = []
    buf: dict = {}
    for line in text.splitlines():
        m = train_re.search(line)
        if m:
            buf = dict(train_f1=float(m[1]), train_recall=float(m[2]),
                       train_precision=float(m[3]), train_auprc=float(m[4]))
            if m[5] is not None:
                buf["train_logloss"] = float(m[5])
            continue
        m = val_re.search(line)
        if m and buf:
            buf.update(val_f1=float(m[1]), val_recall=float(m[2]),
                       val_precision=float(m[3]), val_auprc=float(m[4]))
            if m[5] is not None:
                buf["val_logloss"] = float(m[5])
            continue
        m = test_re.search(line)
        if m and buf:
            buf.update(test_f1=float(m[1]), test_recall=float(m[2]),
                       test_precision=float(m[3]), test_auprc=float(m[4]))
            if m[5] is not None:
                buf["test_logloss"] = float(m[5])
            epochs.append(buf)
            buf = {}

    edge_feats: list[str] = []
    if edge_re:
        try:
            edge_feats = json.loads(edge_re[1].replace("'", '"'))
        except Exception:
            pass

    return {
        "epochs":            epochs,
        "n_nodes":           int(nodes_re[1])                   if nodes_re      else None,
        "n_txns":            int(txns_re[1])                    if txns_re       else None,
        "illicit_ratio":     float(ir_re[1])                    if ir_re         else None,
        "edge_features":     edge_feats,
        "total_params":      int(params_re[1].replace(",", "")) if params_re     else None,
        "training_time_sec": float(time_re[1])                  if time_re       else None,
        "best_epoch":        int(best_epoch_re[1])              if best_epoch_re else None,
        "early_stop_epoch":  int(early_stop_re[1])              if early_stop_re else None,
        "xai_cm":            {"tp": int(xai_cm_re[1]), "fp": int(xai_cm_re[2]),
                              "fn": int(xai_cm_re[3]), "tn": int(xai_cm_re[4])}
                             if xai_cm_re else None,
    }


@st.cache_data(ttl=3600)
def _get_gnn_base_folders(project_folder_id: str) -> dict[str, str]:
    """gnn/ 하위 logs/, models/ 폴더 ID 반환"""
    gnn_id = _get_folder_id(project_folder_id, "gnn")
    if not gnn_id:
        return {}
    return {
        "logs":               _get_folder_id(gnn_id, "logs"),
        "models":             _get_folder_id(gnn_id, "models"),
        "feature_importance": _get_folder_id(gnn_id, "feature_importance"),
    }


@st.cache_data(ttl=3600)
def _list_gnn_experiments(project_folder_id: str) -> list[str]:
    """logs/ 폴더의 .log 파일 이름에서 실험 목록 추출"""
    base = _get_gnn_base_folders(project_folder_id)
    if not base.get("logs"):
        return []
    lf = _list_files(base["logs"])
    return sorted(name[:-4] for name in lf if name.endswith(".log"))


@st.cache_data(ttl=3600)
def _load_gnn_experiment(project_folder_id: str, exp_name: str) -> dict:
    base      = _get_gnn_base_folders(project_folder_id)
    logs_id   = base.get("logs", "")
    models_id = base.get("models", "")
    out: dict = {}

    if models_id:
        mf      = _list_files(models_id)
        args_fn = f"checkpoint_{exp_name}_args.json"
        if args_fn in mf:
            out["args"] = _download_json(mf[args_fn])

    if logs_id:
        lf     = _list_files(logs_id)
        log_fn = f"{exp_name}.log"
        if log_fn in lf:
            r = requests.get(
                f"https://drive.google.com/uc?export=download&id={lf[log_fn]}",
                timeout=60,
            )
            if r.ok:
                out["parsed"] = _parse_gnn_log(r.text)

    fi_id = base.get("feature_importance", "")
    if fi_id:
        fi_files = _list_files(fi_id)
        fi_fn    = f"{exp_name}_feature_importance.csv"
        if fi_fn in fi_files:
            out["feature_importance"] = _download_csv(fi_files[fi_fn])

    return out


# ── Figure cache helpers ───────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _make_gnn_lc_fig(epochs_json: str, mc: str, best_epoch: int, metric_sel: str) -> go.Figure:
    ep_df = pd.read_json(StringIO(epochs_json))
    fig = go.Figure()
    for split, color in [
        ("train", "#4f9cf9"),
        ("val",   "#ff7f0e"),
        ("test",  "#2ca02c"),
    ]:
        col_name = f"{split}_{mc}"
        if col_name in ep_df.columns:
            fig.add_trace(go.Scatter(
                x=ep_df["epoch"], y=ep_df[col_name],
                mode="lines", name=split,
                line=dict(color=color, width=2),
            ))
    fig.add_vline(
        x=best_epoch, line_dash="dot", line_color="#d62728",
        annotation_text=f"best epoch {best_epoch}",
        annotation_font_size=10,
    )
    fig.update_layout(
        height=380, margin=dict(t=20, b=20),
        yaxis_title=metric_sel, xaxis_title="Epoch",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


@st.cache_data(ttl=3600)
def _make_ml_lc_fig2(train_vals: tuple, val_vals: tuple, metric_label: str, best_iter: int) -> go.Figure:
    rows = (
        [{"Iteration": i + 1, "split": "train", metric_label: v} for i, v in enumerate(train_vals)]
        + [{"Iteration": i + 1, "split": "val",   metric_label: v} for i, v in enumerate(val_vals)]
    )
    fig = px.line(
        pd.DataFrame(rows), x="Iteration", y=metric_label, color="split",
        color_discrete_map={"train": "#4f9cf9", "val": "#ff7f0e"},
    )
    fig.add_vline(x=best_iter + 1, line_dash="dash", line_color="#d62728",
                  annotation_text="best", annotation_font_size=11)
    fig.update_layout(height=310, margin=dict(t=40, b=20), legend_title_text="")
    return fig


@st.cache_data(ttl=3600)
def _make_assoc_heatmap(
    features_json: str, matrix_json: str, top_features: tuple,
    metric_matrix_json: str = "[]", desc_map_json: str = "{}",
) -> go.Figure:
    features      = json.loads(features_json)
    matrix        = json.loads(matrix_json)
    metric_matrix = json.loads(metric_matrix_json)
    desc_map      = json.loads(desc_map_json)
    all_names     = [f["name"] for f in features]

    if top_features:
        top_set = set(top_features)
        idx           = [i for i, n in enumerate(all_names) if n in top_set]
        names         = [all_names[i] for i in idx]
        matrix        = [[matrix[r][c] for c in idx] for r in idx]
        metric_matrix = [[metric_matrix[r][c] for c in idx] for r in idx] if metric_matrix else []
    else:
        names = all_names

    n = len(names)
    custom = [
        [
            [desc_map.get(names[c], ""), desc_map.get(names[r], ""),
             metric_matrix[r][c] if metric_matrix else ""]
            for c in range(n)
        ]
        for r in range(n)
    ]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        customdata=custom,
        hovertemplate=(
            "<b>X: %{x}</b><br>%{customdata[0]}<br>"
            "<b>Y: %{y}</b><br>%{customdata[1]}<br>"
            "<b>%{customdata[2]}: %{z:.4f}</b><extra></extra>"
        ),
        showscale=False,
    ))
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=max(400, n * 28),
        margin=dict(t=40, b=20),
        yaxis=dict(scaleanchor="x", tickfont_size=9),
        xaxis=dict(showticklabels=False),
    )
    return fig


def _infer_model_name(train_summary: dict, metrics_raw: dict) -> str:
    run_id = (train_summary.get("run_id") or metrics_raw.get("run_id") or "").lower()
    if run_id.startswith("xgb"):
        return "XGBoost"
    return "XGBoost"


@st.cache_data(ttl=3600)
def _make_cm_fig(tp: int, fn: int, fp: int, tn: int) -> go.Figure:
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (fp + tn) if (fp + tn) > 0 else 0
    z_text = [
        [f"{tp:,}<br>TPR {tpr:.1%}", f"{fn:,}<br>FNR {fnr:.1%}"],
        [f"{fp:,}<br>FPR {fpr:.1%}", f"{tn:,}<br>TNR {tnr:.1%}"],
    ]
    fig = px.imshow(
        [[tp, fn], [fp, tn]],
        x=["Pred Fraud", "Pred Normal"],
        y=["Actual Fraud", "Actual Normal"],
        color_continuous_scale="Blues", text_auto=False, title="",
    )
    fig.update_traces(text=z_text, texttemplate="%{text}", textfont={"size": 11})
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=290, margin=dict(t=10, b=10))
    return fig


@st.cache_data(ttl=3600)
def _make_fi_bar_fig(fi_df_json: str, top_n_fi: int, fi_desc: bool) -> go.Figure:
    fi_df = pd.read_json(StringIO(fi_df_json))
    fig = px.bar(
        fi_df, x="importance_gain", y="feature", orientation="h",
        color="importance_gain", color_continuous_scale="Blues",
        labels={"importance_gain": "Gain", "feature": "Feature"},
        custom_data=["importance_weight", "importance_cover", "rank_by_gain", "_desc"],
    )
    fig.update_traces(hovertemplate=(
        "<b>%{y}</b><br>Gain: %{x:,.1f}<br>"
        "Weight: %{customdata[0]:,.0f}<br>Cover: %{customdata[1]:,.1f}<br>"
        "Rank: %{customdata[2]}<br>%{customdata[3]}<extra></extra>"
    ))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(
        height=max(400, top_n_fi * 28),
        yaxis={"categoryorder": "total ascending" if fi_desc else "total descending"},
        margin=dict(t=40, b=20),
    )
    return fig


@st.cache_data(ttl=3600)
def _make_iv_bar_fig(
    top_df_json: str, woe_desc: bool, iv_cut: float,
    unregistered_tuple: tuple, has_overflow: bool,
) -> go.Figure:
    top_df = pd.read_json(StringIO(top_df_json))
    unregistered = set(unregistered_tuple)
    fig = px.bar(
        top_df, x="_iv_bar", y="feature_name", orientation="h",
        color="iv_strength", color_discrete_map=IV_COLORS,
        custom_data=["iv", "_desc"],
        labels={"_iv_bar": "IV", "feature_name": "Feature", "iv_strength": "강도"},
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>IV: %{customdata[0]:.4f}<br>%{customdata[1]}<extra></extra>"
    )
    fig.update_layout(
        height=max(420, len(top_df) * 40),
        yaxis={"categoryorder": "total ascending" if woe_desc else "total descending"},
        xaxis={
            "range":    [0, iv_cut + (0.35 if has_overflow else 0.05)],
            "tickvals": [0, 0.5, 1.0, 1.5],
            "ticktext": ["0", "0.5", "1.0", "1.5"],
            "title":    "IV",
        },
        legend_title_text="예측력",
    )
    for i, row in top_df[top_df["iv"] > iv_cut].iterrows():
        fig.add_annotation(x=iv_cut + 0.04, y=i, text=f"{row['iv']:.4f}",
                           showarrow=False, xanchor="left", font=dict(size=10))
    unreg_rows = top_df[top_df["feature_name"].isin(unregistered)]
    if not unreg_rows.empty:
        fig.add_trace(go.Scatter(
            x=unreg_rows["_iv_bar"] + 0.02, y=unreg_rows["feature_name"],
            mode="markers",
            marker=dict(symbol="triangle-right", size=10, color="#ff7f0e"),
            name="카탈로그 미등록", hoverinfo="skip",
        ))
    for val, label, color in [
        (0.02, "weak",       "#aaaaaa"),
        (0.10, "medium",     "#888888"),
        (0.30, "strong",     "#555555"),
        (0.50, "suspicious", "#222222"),
    ]:
        fig.add_vline(x=val, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_font_size=10)
    return fig


@st.cache_data(ttl=3600)
def _make_woe_bin_fig(feat_bins_json: str, sel_feature: str) -> go.Figure:
    feat_bins    = pd.read_json(StringIO(feat_bins_json))
    main_bins    = feat_bins[~feat_bins["missing_flag"]].sort_values("bin_id")
    missing_bins = feat_bins[feat_bins["missing_flag"]]
    feat_sorted  = pd.concat([main_bins, missing_bins], ignore_index=True)
    feat_sorted["_color"] = feat_sorted["woe"].apply(lambda w: "fraud↑" if w >= 0 else "fraud↓")
    fig_woe = make_subplots(specs=[[{"secondary_y": True}]])
    for _lbl, _hex in [("fraud↑", "#d62728"), ("fraud↓", "#2ca02c")]:
        _sub = feat_sorted[feat_sorted["_color"] == _lbl]
        if _sub.empty:
            continue
        fig_woe.add_trace(
            go.Bar(
                x=_sub["bin_label"], y=_sub["woe"],
                name=_lbl, marker_color=_hex,
                customdata=_sub[["count", "positive_count", "positive_rate", "iv_bin"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>WOE: %{y:.4f}<br>"
                    "Count: %{customdata[0]:,}<br>"
                    "Positive: %{customdata[1]:,}<br>"
                    "Positive Rate: %{customdata[2]:.5f}<br>"
                    "IV Bin: %{customdata[3]:.4f}<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
    fig_woe.add_trace(
        go.Scatter(
            x=feat_sorted["bin_label"], y=feat_sorted["count"],
            mode="lines+markers", name="Count",
            line=dict(color="#aaaaaa", width=1.5, dash="dot"),
            marker=dict(size=4, color="#888888"),
            hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_woe.add_hline(y=0, line_dash="dash", line_color="#333333", line_width=1)
    fig_woe.update_layout(
        height=400, xaxis_tickangle=-40,
        legend=dict(title=""),
        margin=dict(t=20),
        xaxis=dict(
            categoryorder="array",
            categoryarray=feat_sorted["bin_label"].tolist(),
        ),
    )
    fig_woe.update_yaxes(title_text="WOE", secondary_y=False)
    fig_woe.update_yaxes(title_text="Count", secondary_y=True, showgrid=False)
    return fig_woe


# ── Fragment sections ──────────────────────────────────────────────────────

@st.fragment
def _gnn_lc_section(ep_df: pd.DataFrame, best_ep: pd.Series) -> None:
    _mc, _ = st.columns([3, 5])
    metric_sel = _mc.radio(
        "지표", ["F1", "AUPRC", "Recall", "Precision", "LogLoss"],
        horizontal=True, key="gnn_metric", label_visibility="collapsed",
    )
    mc = "logloss" if metric_sel == "LogLoss" else metric_sel.lower()
    fig_lc = _make_gnn_lc_fig(ep_df.to_json(orient="records"), mc, int(best_ep["epoch"]), metric_sel)
    st.plotly_chart(fig_lc, use_container_width=True)


@st.fragment
def _woe_iv_bin_section(
    top_df: pd.DataFrame,
    bin_df: pd.DataFrame | None,
    unregistered: set,
    woe_desc: bool,
) -> None:
    has_overflow = (top_df["iv"] > IV_CUT).any()
    fig_iv = _make_iv_bar_fig(
        top_df.to_json(orient="records"), woe_desc, IV_CUT,
        tuple(sorted(unregistered)), bool(has_overflow),
    )
    iv_event = st.plotly_chart(fig_iv, use_container_width=True, on_select="rerun", key="iv_chart")

    sel_feature: str | None = None
    pts = (iv_event.selection or {}).get("points", []) if iv_event else []
    if pts:
        sel_feature = pts[0].get("label") or pts[0].get("y")

    if sel_feature:
        st.markdown(f"#### WOE — `{sel_feature}`")
        if bin_df is None:
            st.info("bin_table.json이 없습니다. compute_woe_iv.ipynb를 다시 실행하세요.")
        else:
            feat_bins = bin_df[bin_df["feature_name"] == sel_feature].copy()
            if feat_bins.empty:
                st.info(f"'{sel_feature}'의 WOE 구간 데이터가 없습니다.")
            else:
                fig_woe = _make_woe_bin_fig(feat_bins.to_json(orient="records"), sel_feature)
                st.plotly_chart(fig_woe, use_container_width=True)


# ── Page header ────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([8, 1])
col_title.title("돈무브 프로젝트 대시보드")
if col_btn.button("🔄 새로고침", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if not API_KEY or not PROJECT_FOLDER_ID:
    st.error(
        "**Streamlit secrets 설정 필요** — `.streamlit/secrets.toml` 또는 "
        "`~/.streamlit/secrets.toml`에 아래 항목을 추가하세요.\n\n"
        "```toml\n"
        'GOOGLE_API_KEY    = "AIzaSy..."\n'
        'PROJECT_FOLDER_ID = "1abc..."\n'
        "```"
    )
    st.stop()

# ── 데이터 로드 ────────────────────────────────────────────────────────────

with st.spinner("Drive 연결 중..."):
    ml_folder_id   = _get_folder_id(PROJECT_FOLDER_ID, "ml")
    woe_iv_root_id = _get_woe_iv_root_id(PROJECT_FOLDER_ID)

if not ml_folder_id:
    st.error("Drive에서 'ml' 폴더를 찾을 수 없습니다. PROJECT_FOLDER_ID를 확인하세요.")
    st.stop()

if not woe_iv_root_id:
    st.warning("Drive에서 'data/ml/woe_iv' 폴더를 찾을 수 없습니다. WOE/IV 탭은 비어 있을 수 있습니다.")

reps       = _load_representatives(ml_folder_id)
valid_reps = [r for r in reps if _is_valid_rep(r)]

if not valid_reps:
    st.warning("ml_leaderboard_representatives.json에 유효한 실험이 없습니다.")
    st.stop()

# 실험별 데이터 로드 + stale 상태 계산
exp_data: dict[str, dict] = {}
bar = st.progress(0, text="실험 데이터 로드 중...")
for i, rep in enumerate(valid_reps):
    ml_exp_folder_id = _get_folder_id(ml_folder_id, rep["ml_folder"])
    run_folder_id    = _get_folder_id(ml_exp_folder_id, rep["run_id"]) if ml_exp_folder_id else ""
    woe_iv_name      = _woe_iv_folder_name(rep["ml_folder"])
    woe_iv_exp_id    = _get_folder_id(woe_iv_root_id, woe_iv_name) if woe_iv_root_id else ""
    prefix           = _artifact_prefix(rep)
    cat_fn           = _catalog_filename(rep["ml_folder"])
    label            = _exp_label(rep)

    woe     = _load_woe_results(woe_iv_exp_id)
    catalog = _load_catalog(ml_exp_folder_id, cat_fn)
    cached_prefix = woe.get("meta", {}).get("prefix") if woe else None

    if not woe or "iv_df" not in woe:
        stale_status = "no_woe"
    elif cached_prefix != prefix:
        stale_status = "stale"
    else:
        stale_status = "fresh"

    exp_data[label] = {
        "rep":          rep,
        "ml":           _load_ml_results(run_folder_id, prefix) if run_folder_id else {},
        "woe":          woe,
        "catalog":      catalog,
        "stale_status": stale_status,
        "prefix":       prefix,
    }
    bar.progress((i + 1) / len(valid_reps), text=f"로드: {rep['ml_folder']}")
bar.empty()

if not exp_data:
    st.warning("로드 가능한 실험이 없습니다.")
    st.stop()

exp_labels   = list(exp_data.keys())
_default_idx = 0

# GNN 실험 목록 로드
gnn_exp_names = _list_gnn_experiments(PROJECT_FOLDER_ID)


# ══════════════════════════════════════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_gnn, tab_ml, tab_woe = st.tabs(["Overview", "GNN Result", "ML Result", "Univariate Analysis"])


# ──────────────────────────────────────────────────────────────────────────────
# 탭 0: Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown(
        "<div style='text-align:center; padding: 60px 0 20px'>"
        "<div style='font-size: 3rem'>🚧</div>"
        "<div style='font-size: 1.4rem; font-weight: 600; margin-top: 12px'>공사중</div>"
        "<div style='color: #888; margin-top: 8px'>이 탭은 현재 준비 중입니다.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 탭 1: GNN 결과
# ──────────────────────────────────────────────────────────────────────────────
with tab_gnn:
    if not gnn_exp_names:
        st.info("Drive의 gnn/logs 폴더에서 실험을 찾을 수 없습니다.")
    else:
        sel_gnn = st.selectbox("실험 선택", gnn_exp_names, key="gnn_sel",
                               label_visibility="collapsed")

        _render_report("GNN Result", sel_gnn)
        st.divider()

        with st.spinner("GNN 실험 로드 중..."):
            gnn_d  = _load_gnn_experiment(PROJECT_FOLDER_ID, sel_gnn)

        args   = gnn_d.get("args", {})
        parsed = gnn_d.get("parsed", {})
        epochs = parsed.get("epochs", [])

        if not epochs:
            st.info("학습 로그를 파싱할 수 없습니다.")
        else:
            ep_df = pd.DataFrame(epochs)
            ep_df.index = ep_df.index + 1
            ep_df.index.name = "epoch"
            ep_df = ep_df.reset_index()

            best_idx = ep_df["val_auprc"].idxmax()
            best_ep  = ep_df.loc[best_idx]

            # ── 핵심 지표 ──────────────────────────────────────────────────
            _model_name = args.get("model", "—") if args else "—"
            st.markdown(f"#### Metrics (model: {_model_name})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("F1",            f"{best_ep['test_f1']:.4f}")
            c2.metric("AUPRC",         f"{best_ep['test_auprc']:.4f}")
            c3.metric("Precision",     f"{best_ep['test_precision']:.4f}")
            c4.metric("Recall",        f"{best_ep['test_recall']:.4f}")
            c5.metric("Best Val AUPRC", f"{best_ep['val_auprc']:.4f}")

            c6, c7 = st.columns(2)
            _n_txns  = parsed.get("n_txns")
            _ir      = parsed.get("illicit_ratio")
            _metric(c6, "거래 수", f"{_n_txns:,}" if _n_txns else "—",
                    f"pos {_ir:.2f}%" if _ir else "")
            _train_time = parsed.get("training_time_sec")
            _metric(c7, "Best epoch / 학습시간",
                    f"{int(best_ep['epoch'])}  /  {f'{_train_time:.0f}' if _train_time else '—'}",
                    "sec")

            st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

            # ── Hyper Parameters ───────────────────────────────────────────
            if args:
                with st.expander("Hyper Parameters"):
                    _skip = {"model", "unique_name", "save_model", "inference", "tqdm", "finetune", "data"}
                    _params = {k: v for k, v in args.items() if k not in _skip and v is not None}
                    _n_cols = 4
                    _rows = [list(_params.items())[i:i+_n_cols] for i in range(0, len(_params), _n_cols)]
                    for _row in _rows:
                        for _col, (_k, _v) in zip(st.columns(_n_cols), _row):
                            _col.metric(_k, str(_v) if not isinstance(_v, (int, float)) else _v)

            st.divider()

            # ── Learning Curve + Confusion Matrix ─────────────────────────
            col_curve, col_cm = st.columns([3, 2])

            with col_curve:
                st.markdown("##### Learning Curve")
                _gnn_lc_section(ep_df, best_ep)

            with col_cm:
                st.markdown("##### Confusion Matrix")
                xai_cm = parsed.get("xai_cm")
                if xai_cm:
                    tp, fp, fn, tn = xai_cm["tp"], xai_cm["fp"], xai_cm["fn"], xai_cm["tn"]
                    fig_cm = _make_cm_fig(tp, fn, fp, tn)
                    st.plotly_chart(fig_cm, use_container_width=True)
                    st.markdown(
                        f"<div style='text-align:center;font-size:0.8rem;color:#888'>"
                        f"TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Confusion matrix 정보 없음 (XAI 미실행)")

            st.divider()

            # ── Feature Importance (XAI) ───────────────────────────────────
            fi_df_gnn = gnn_d.get("feature_importance")
            st.markdown("##### Feature Importance (XAI — Gradient Saliency)")
            if fi_df_gnn is not None and not fi_df_gnn.empty:
                mean_cols = [c for c in fi_df_gnn.columns if c.endswith("__mean")]
                feat_names = [c.replace("__mean", "") for c in mean_cols]
                fi_long_rows = []
                for _, row in fi_df_gnn.iterrows():
                    n_samples = int(row.get("n_samples", 0))
                    for mc2, fname in zip(mean_cols, feat_names):
                        std_col = fname + "__std"
                        fi_long_rows.append({
                            "group":    row["group"],
                            "feature":  fname,
                            "saliency": row[mc2],
                            "std":      row.get(std_col, 0),
                            "n":        n_samples,
                        })
                fi_long = pd.DataFrame(fi_long_rows)

                groups_order = [g for g in ["TP", "FP", "FN", "TN"]
                                if g in fi_long["group"].values]
                subplot_titles = []
                for grp in groups_order:
                    n_val = int(fi_long.loc[fi_long["group"] == grp, "n"].iloc[0])
                    subplot_titles.append(f"{grp}  (n={n_val:,})")

                fig_gnn_fi = make_subplots(
                    rows=2, cols=2,
                    shared_yaxes="all",
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.32,
                    horizontal_spacing=0.06,
                )
                for idx, grp in enumerate(groups_order):
                    r, c = idx // 2 + 1, idx % 2 + 1
                    fi_grp = fi_long[fi_long["group"] == grp].sort_values("saliency", ascending=False)
                    fig_gnn_fi.add_trace(go.Bar(
                        x=fi_grp["feature"],
                        y=fi_grp["saliency"],
                        error_y=dict(type="data", array=fi_grp["std"].tolist(),
                                     visible=True, color="#888888", thickness=1.2, width=4),
                        marker_color="#4f9cf9",
                        showlegend=False,
                        hovertemplate="<b>%{x}</b><br>Mean: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>",
                    ), row=r, col=c)

                y_max = (fi_long["saliency"] + fi_long["std"]).max() * 1.15
                y_min = max(0, (fi_long["saliency"] - fi_long["std"]).min() * 0.9)
                fig_gnn_fi.update_layout(
                    height=660, margin=dict(t=60, b=60),
                )
                fig_gnn_fi.update_xaxes(tickangle=-40)
                fig_gnn_fi.update_yaxes(range=[y_min, y_max])
                fig_gnn_fi.update_yaxes(title_text="Mean Saliency", col=1)
                st.plotly_chart(fig_gnn_fi, use_container_width=True)
            else:
                st.info("Feature importance 파일 없음 (feature_importance/ 폴더 확인)")

            st.divider()

            # ── 데이터 통계 ────────────────────────────────────────────────
            st.markdown("##### 데이터 통계")
            _dc1, _dc2 = st.columns([1, 2])
            _params = parsed.get("total_params")
            _dc1.markdown(f"""
| 항목 | 값 |
|------|-----|
| 노드 수 | {f'{parsed.get("n_nodes"):,}' if parsed.get("n_nodes") else '—'} |
| 총 파라미터 | {f'{_params:,}' if _params else '—'} |
""")
            _ef = parsed.get("edge_features", [])
            if _ef:
                with _dc2.expander(f"Edge Features ({len(_ef)}개)"):
                    st.dataframe(
                        pd.DataFrame({"feature": _ef}),
                        use_container_width=True, hide_index=True,
                    )


# ──────────────────────────────────────────────────────────────────────────────
# 탭 2: ML 결과
# ──────────────────────────────────────────────────────────────────────────────
with tab_ml:
    sel = st.selectbox("실험 선택", exp_labels, key="ml_sel",
                       index=_default_idx, label_visibility="collapsed")

    # 실험 변경 시 이전 fi_* 키 정리
    if st.session_state.get("_ml_prev_sel") != sel:
        prev = st.session_state.get("_ml_prev_sel")
        if prev:
            for k in [f"fi_sel_{prev}", f"fi_bar_ver_{prev}", f"fi_scat_ver_{prev}"]:
                st.session_state.pop(k, None)
        st.session_state["_ml_prev_sel"] = sel

    d   = exp_data[sel]
    rep = d["rep"]
    ml  = d["ml"]

    note = rep.get("note", "")
    st.caption(f"**Status**: {rep.get('status', '—')}")

    _render_report("ML Result", sel)
    st.divider()

    if not ml:
        st.info("학습된 모델이 없습니다.")
    else:
        metrics_raw    = ml.get("metrics", {})
        train_summary  = ml.get("train_summary", {})
        scores_summary = ml.get("scores_summary", {})
        feat_imp       = ml.get("feature_importance")
        conf_mat       = ml.get("confusion_matrix")
        feature_assoc  = ml.get("feature_assoc")
        m              = metrics_raw.get("metrics", metrics_raw)

        # ── 성능 지표 카드 ────────────────────────────────────────────────────
        _model_name = _infer_model_name(train_summary, metrics_raw)
        st.markdown(f"#### Metrics (model: {_model_name})")
        c1, c2, c3, c4, c5 = st.columns(5)
        f1    = m.get("f1", 0)
        aucpr = m.get("average_precision") or train_summary.get("best_score")
        c1.metric("F1",        f"{f1:.4f}")
        c2.metric("AUPRC",     f"{aucpr:.4f}" if aucpr is not None else "—")
        c3.metric("Precision", f"{m.get('precision', 0):.4f}")
        c4.metric("Recall",    f"{m.get('recall', 0):.4f}")
        c5.metric("Threshold", f"{m.get('threshold', 0):.4f}",
                  help=f"전략: {train_summary.get('xgboost_params', {}).get('eval_metric', '—')}")

        c6, c7, c8 = st.columns(3)
        train_rows = train_summary.get("train_rows", 0)
        train_pos  = train_summary.get("train_positive_ratio", 0)
        val_rows   = train_summary.get("val_rows", 0)
        val_pos    = (train_summary.get("val_label_summary") or {}).get("positive_ratio", 0)
        best_iter  = train_summary.get("best_iteration", 0)
        train_time = train_summary.get("training_time_sec", 0)
        _metric(c6, "Train",               f"{train_rows:,}",              f"pos {train_pos*100:.3f}%")
        _metric(c7, "Val",                 f"{val_rows:,}",                f"pos {val_pos*100:.3f}%")
        _metric(c8, "Best iter / 학습시간", f"{best_iter + 1}  /  {train_time:.0f}", "sec")

        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        xgb_params = train_summary.get("xgboost_params", {})
        if xgb_params:
            with st.expander("Hyper Parameters"):
                _p = {k: v for k, v in xgb_params.items() if v is not None}
                n_cols = 4
                rows = [list(_p.items())[i:i+n_cols] for i in range(0, len(_p), n_cols)]
                for row in rows:
                    for col, (k, v) in zip(st.columns(n_cols), row):
                        col.metric(k, v if isinstance(v, (int, float, str)) else str(v))

        st.divider()

        # ── 학습 곡선 + Confusion Matrix ──────────────────────────────────────
        col_curve, col_cm = st.columns([3, 2])

        with col_curve:
            st.markdown("##### Learning Curve")
            # scores_train_summary.learning_curve 우선, 없으면 evals_result fallback
            lc       = (scores_summary.get("learning_curve") or {})
            lc_curves = lc.get("curves", {})
            aliases  = lc.get("eval_set_aliases", {})
            diag     = train_summary.get("xgboost_diagnostics", {})
            evals    = diag.get("evals_result", {})

            # metric → (train_vals, val_vals)
            available: dict[str, tuple] = {}
            for metric, vals in lc_curves.get("train", {}).items():
                val_vals = lc_curves.get("val", {}).get(metric)
                if val_vals:
                    available[metric] = (tuple(vals), tuple(val_vals))
            # evals_result에서 추가 지표 수집
            train_key = next((k for k, v in aliases.items() if v == "train"), None)
            val_key   = next((k for k, v in aliases.items() if v == "val"),   None)
            if not train_key and evals:
                keys = list(evals.keys())
                train_key, val_key = (keys[0], keys[1]) if len(keys) >= 2 else (keys[0], None) if keys else (None, None)
            if train_key and val_key and evals.get(train_key) and evals.get(val_key):
                for metric, t_vals in evals[train_key].items():
                    if metric not in available and metric in evals[val_key]:
                        available[metric] = (tuple(t_vals), tuple(evals[val_key][metric]))

            if available:
                metric_options = list(available.keys())
                lc_sel = st.radio("지표", metric_options, horizontal=True,
                                  key=f"lc_metric_{sel}", label_visibility="collapsed")
                t_v, v_v = available[lc_sel]
                fig_curve = _make_ml_lc_fig2(t_v, v_v, lc_sel, int(best_iter))
                st.plotly_chart(fig_curve, use_container_width=True)
            else:
                st.info("학습 곡선 데이터 없음")

        with col_cm:
            st.markdown("##### Confusion Matrix")
            if conf_mat is not None and not conf_mat.empty:
                row = conf_mat.iloc[0]
                tn, fp, fn, tp = int(row.get("tn",0)), int(row.get("fp",0)), int(row.get("fn",0)), int(row.get("tp",0))
                fig_cm = _make_cm_fig(tp, fn, fp, tn)
                st.plotly_chart(fig_cm, use_container_width=True)
                st.markdown(
                    f"<div style='text-align:center;font-size:0.8rem;color:#888'>"
                    f"TP={tp:,} | FP={fp:,} | FN={fn:,} | TN={tn:,}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Confusion matrix 파일 없음")

        st.divider()

        # ── Feature Importance ────────────────────────────────────────────────
        if feat_imp is not None and not feat_imp.empty:
            st.markdown("##### Feature Importance")
            n_total  = len(feat_imp)
            _col_n, _col_s = st.columns([4, 1])
            top_n_fi = _col_n.slider("N Features", 10, n_total, min(20, n_total), key="fi_slider", label_visibility="collapsed")
            fi_desc  = _col_s.radio("정렬", ["높은 순", "낮은 순"], horizontal=True, key="fi_sort") == "높은 순"
            fi_df = (
                feat_imp
                .sort_values("importance_gain", ascending=not fi_desc)
                .head(top_n_fi)
                .sort_values("importance_gain")
                .reset_index(drop=True)
            )
            _cat_fi   = d.get("catalog")
            _desc_map: dict = {}
            if _cat_fi is not None and not _cat_fi.empty and "피처명" in _cat_fi.columns:
                _desc_map = _cat_fi.set_index("피처명")["설명"].to_dict()
            fi_df["_desc"] = fi_df["feature"].map(lambda f: _desc_map.get(f) or "")
            fig_fi = _make_fi_bar_fig(fi_df.to_json(orient="records"), top_n_fi, fi_desc)
            # 바/버블 선택이 서로를 덮어쓰지 않도록 versioned key 사용
            _bar_ver  = st.session_state.get(f"fi_bar_ver_{sel}", 0)
            _scat_ver = st.session_state.get(f"fi_scat_ver_{sel}", 0)
            _sel_fi_pre = st.session_state.get(f"fi_sel_{sel}")
            if _sel_fi_pre and _sel_fi_pre in fi_df["feature"].values:
                fig_fi.data[0].marker.opacity = [
                    1.0 if f == _sel_fi_pre else 0.15 for f in fi_df["feature"]
                ]

            fi_event = st.plotly_chart(fig_fi, use_container_width=True, on_select="rerun",
                                       key=f"fi_chart_{sel}_v{_bar_ver}")

            _fi_pts  = (fi_event.selection or {}).get("points", []) if fi_event else []
            _from_bar = None
            if _fi_pts:
                _c = _fi_pts[0].get("label") or _fi_pts[0].get("y")
                if _c:
                    _from_bar = _c

            _sel_fi = _sel_fi_pre

            with st.expander("Feature Association Heatmap"):
                if feature_assoc:
                    assoc     = feature_assoc.get("association", {})
                    feat_list = feature_assoc.get("features", [])  # top-level: [{name, feature_type}, ...]
                    matrix    = assoc.get("matrix", [])
                    if feat_list and matrix:
                        top_fi_names = tuple(fi_df["feature"].tolist())
                        split_label  = feature_assoc.get("split", "?")
                        methods      = feature_assoc.get("association_methods", {})
                        st.caption(f"Split: {split_label}  |  numeric↔numeric: {methods.get('numeric_numeric','?')}  |  cat↔cat: {methods.get('categorical_categorical','?')}  |  mixed: {methods.get('numeric_categorical','?')}")
                        fig_heatmap = _make_assoc_heatmap(
                            json.dumps(feat_list), json.dumps(matrix), top_fi_names,
                            json.dumps(assoc.get("metric_matrix", [])),
                            json.dumps(_desc_map),
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("Association 데이터 없음")
                else:
                    st.info("feature_assoc_mixed 파일 없음")

            st.divider()
            st.markdown("##### Weight vs Gain")

            fig_scat = px.scatter(
                fi_df, x="importance_gain", y="importance_weight",
                size="importance_cover", hover_name="feature",
                color="importance_gain", color_continuous_scale="Blues",
                labels={"importance_gain": "Gain", "importance_weight": "Weight", "importance_cover": "Cover"},
                custom_data=["importance_cover", "_desc", "feature"],
            )
            fig_scat.update_traces(hovertemplate=(
                "<b>%{hovertext}</b><br>Gain: %{x:,.1f}<br>Weight: %{y:,.0f}<br>"
                "Cover: %{customdata[0]:,.1f}<br>%{customdata[1]}<extra></extra>"
            ))
            fig_scat.update_coloraxes(showscale=False)
            if _sel_fi and _sel_fi in fi_df["feature"].values:
                _opacities = [1.0 if f == _sel_fi else 0.12 for f in fi_df["feature"]]
                fig_scat.data[0].marker.opacity = _opacities
                _row = fi_df[fi_df["feature"] == _sel_fi].iloc[0]
                _desc_txt = _row["_desc"]
                _ann_text = (
                    f"<b>{_sel_fi}</b><br>"
                    f"Gain: {_row['importance_gain']:,.1f}<br>"
                    f"Weight: {_row['importance_weight']:,.0f}<br>"
                    f"Cover: {_row['importance_cover']:,.1f}"
                    + (f"<br>{_desc_txt}" if _desc_txt else "")
                )
                _sizeref  = fig_scat.data[0].marker.sizeref or 1
                _bubble_px = 2 * (_row["importance_cover"] / _sizeref) ** 0.5
                _standoff  = max(6, _bubble_px / 2 + 3)
                fig_scat.add_trace(go.Scatter(
                    x=[_row["importance_gain"]], y=[_row["importance_weight"]],
                    mode="markers",
                    marker=dict(
                        size=_bubble_px, symbol="circle-open",
                        color="rgba(0,0,0,0)",
                        line=dict(color="#d62728", width=2.5),
                    ),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_scat.add_annotation(
                    x=_row["importance_gain"], y=_row["importance_weight"],
                    text=_ann_text,
                    showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#d62728",
                    ax=55, ay=-65, standoff=_standoff,
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#d62728", borderwidth=1,
                    font=dict(size=10, color="#333333"), align="left",
                )
            fig_scat.update_layout(height=420, margin=dict(t=20, b=20))
            st.caption("버블 크기 = Cover")
            scat_event = st.plotly_chart(fig_scat, use_container_width=True, on_select="rerun",
                                         key=f"fi_scat_{sel}_v{_scat_ver}")

            _scat_pts = (scat_event.selection or {}).get("points", []) if scat_event else []
            _from_scat = None
            if _scat_pts:
                _cd = _scat_pts[0].get("customdata") or []
                if len(_cd) > 2:
                    _from_scat = _cd[2]

            # 산점도 클릭 우선 처리 — 변경된 경우에만 반응
            _cur = st.session_state.get(f"fi_sel_{sel}")
            if _from_scat and _from_scat != _cur:
                st.session_state[f"fi_sel_{sel}"]       = _from_scat
                st.session_state[f"fi_bar_ver_{sel}"]   = _bar_ver + 1   # 바 선택 초기화
                st.rerun()
            elif _from_bar and _from_bar != _cur:
                st.session_state[f"fi_sel_{sel}"]        = _from_bar
                st.session_state[f"fi_scat_ver_{sel}"]   = _scat_ver + 1  # 버블 선택 초기화
                st.rerun()
        else:
            st.info("Feature importance 파일 없음")


# ──────────────────────────────────────────────────────────────────────────────
# 탭 2: WOE / IV
# ──────────────────────────────────────────────────────────────────────────────
with tab_woe:
    sel_woe  = st.selectbox("실험 선택", exp_labels, key="woe_sel",
                            index=_default_idx, label_visibility="collapsed")

    # 실험 변경 시 이전 catalog 편집 상태 정리
    if st.session_state.get("_woe_prev_sel") != sel_woe:
        prev_woe = st.session_state.get("_woe_prev_sel")
        if prev_woe:
            st.session_state.pop(f"catalog_{prev_woe}", None)
        st.session_state["_woe_prev_sel"] = sel_woe

    _render_report("Univariate Analysis", sel_woe)
    st.divider()

    d_woe    = exp_data[sel_woe]
    woe      = d_woe["woe"]
    rep_woe  = d_woe["rep"]
    stale_st = d_woe["stale_status"]
    cur_pfx  = d_woe["prefix"]

    # ── stale / no_woe 아이콘 ─────────────────────────────────────────────────
    if stale_st == "stale":
        meta_woe    = woe.get("meta", {})
        cached_pfx  = meta_woe.get("prefix", "—")
        computed_at = str(meta_woe.get("computed_at", "—"))[:19]
        tip = (
            "WOE/IV 결과가 최신이 아닙니다. compute_woe_iv.ipynb를 실행하세요.&#10;"
            f"현재 prefix: {cur_pfx}&#10;"
            f"캐시 prefix: {cached_pfx}&#10;"
            f"계산일: {computed_at}"
        )
        st.markdown(
            f'<span title="{tip}" style="cursor:help;font-size:1.1rem">⚠️</span>',
            unsafe_allow_html=True,
        )

    elif stale_st == "no_woe":
        tip = (
            "WOE/IV 결과가 없습니다. compute_woe_iv.ipynb를 실행하세요.&#10;"
            f"prefix: {cur_pfx}"
        )
        st.markdown(
            f'<span title="{tip}" style="cursor:help;font-size:1.1rem">⚠️</span>',
            unsafe_allow_html=True,
        )

    # ── 정상 표시 ─────────────────────────────────────────────────────────────
    if "iv_df" in woe:
        iv_df      = woe["iv_df"].copy()
        bin_df     = woe.get("bin_df")
        meta       = woe.get("meta", {})
        catalog_df = d_woe.get("catalog")

        # 카탈로그 미등록 피처 탐지
        unregistered: set[str] = set()
        if catalog_df is not None and not catalog_df.empty:
            reg_set      = set(catalog_df["피처명"].tolist())
            unregistered = set(iv_df["feature_name"].tolist()) - reg_set

        # session state 기반 catalog used_in_ml 편집
        ss_key = f"catalog_{sel_woe}"
        if catalog_df is not None and not catalog_df.empty:
            if ss_key not in st.session_state:
                keep_cols = [c for c in ["피처명", "설명", "used_in_ml", "데이터 타입", "비고"]
                             if c in catalog_df.columns]
                _init = catalog_df[keep_cols].copy()
                _init["used_in_ml"] = _init["used_in_ml"].map(
                    lambda x: x if isinstance(x, bool) else str(x).upper() == "TRUE"
                ).astype(bool)
                st.session_state[ss_key] = _init
            active_catalog = st.session_state[ss_key]
            excluded = set(active_catalog.loc[~active_catalog["used_in_ml"], "피처명"])
            iv_df = iv_df[~iv_df["feature_name"].isin(excluded)]
        else:
            active_catalog = None
            excluded = set()

        # 메타 + 미등록 경고
        col_meta, col_warn = st.columns([1, 3])
        with col_meta:
            n_rows = meta.get("n_rows") or 0
            st.markdown(f"""
| 항목 | 값 |
|------|-----|
| 계산일 | {meta.get('computed_at', '—')[:19]} |
| feature 수 | {len(iv_df):,} |
| 데이터 | {'전체' if meta.get('full_run') else '샘플'} |
| 행 수 | {n_rows:,} |
| positive rate | {meta.get('positive_rate', 0):.5f} |
""")
        with col_warn:
            if unregistered:
                st.warning(
                    f"**{len(unregistered)}개 피처가 feature_columns에는 있으나 카탈로그 미등록**\n\n"
                    + "  ".join(f"`{f}`" for f in sorted(unregistered))
                )

        # Feature catalog 편집기
        if active_catalog is not None:
            with st.expander("Feature Catalog"):
                col_cfg = {
                    "used_in_ml": st.column_config.CheckboxColumn("used_in_ml",
                                      help="체크 해제 시 차트에서 제외 (앱 내에서만 적용)"),
                    "피처명":    st.column_config.TextColumn("피처명", disabled=True),
                    "설명":      st.column_config.TextColumn("설명",   disabled=True),
                }
                for _col in ["데이터 타입", "비고"]:
                    if _col in active_catalog.columns:
                        col_cfg[_col] = st.column_config.TextColumn(_col, disabled=True)
                edited = st.data_editor(
                    active_catalog, use_container_width=True, hide_index=True,
                    column_config=col_cfg, key=f"editor_{sel_woe}",
                )
                if not edited.equals(active_catalog):
                    st.session_state[ss_key] = edited
                    st.rerun()
        st.divider()

        st.markdown("##### Information Value")
        st.caption(
            "**WOE(Weight of Evidence)**: 각 구간에서 fraud 비율과 정상 비율의 로그 비.  \n"
            "**IV(Information Value)**: WOE를 전체 구간에 걸쳐 집계한 변수 단위 예측력 요약."
        )
        _col_n, _col_s = st.columns([4, 1])
        top_n    = _col_n.slider("N Features", 10, max(10, len(iv_df)), min(20, len(iv_df)), key="woe_top_n", label_visibility="collapsed")
        woe_desc = _col_s.radio("정렬", ["높은 순", "낮은 순"], horizontal=True, key="woe_sort") == "높은 순"

        top_df = (
            iv_df.copy()
            .assign(iv=lambda d: d["iv"].fillna(0))
            .pipe(lambda d: d.head(top_n) if woe_desc else d.tail(top_n))
            .sort_values("iv")
            .reset_index(drop=True)
        )

        if catalog_df is not None and not catalog_df.empty:
            desc_map    = catalog_df.set_index("피처명")["설명"].to_dict()
            top_df["_desc"] = top_df["feature_name"].apply(
                lambda f: "⚠ 카탈로그 미등록" if f in unregistered else (desc_map.get(f) or "")
            )
        else:
            top_df["_desc"] = top_df["feature_name"].apply(
                lambda f: "⚠ 카탈로그 미등록" if f in unregistered else ""
            )

        top_df["_iv_bar"] = top_df["iv"].clip(lower=0.003, upper=IV_CUT)

        _woe_iv_bin_section(top_df, bin_df, unregistered, woe_desc)
