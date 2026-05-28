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
        # 파일이 없으면 multipart POST로 직접 생성
        boundary = "mpart_boundary_report"
        meta     = json.dumps({"name": "report.json", "parents": [exp_id]})
        body     = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{meta}\r\n"
            f"--{boundary}\r\nContent-Type: application/json\r\n\r\n"
            f"{payload.decode('utf-8')}\r\n"
            f"--{boundary}--"
        ).encode("utf-8")
        r = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  f"multipart/related; boundary={boundary}",
            },
            data=body,
            timeout=15,
        )
        if not r.ok:
            st.session_state["_sa_last_error"] = f"Drive API {r.status_code}: {r.text[:300]}"
        return r.ok

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
            st.rerun(scope="fragment")

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
                    st.rerun(scope="fragment")
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
            if col_cancel.button("취소", key=f"rpt_cancel_auth_{tab_name}_{exp_name}", use_container_width=True):
                st.session_state[edit_key] = False
                st.rerun(scope="fragment")
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
                    st.rerun(scope="fragment")
                else:
                    err = st.session_state.pop("_sa_last_error", "서비스 계정 설정을 확인하세요.")
                    st.error(f"저장 실패 — {err}")
            if col_cancel.button("취소", key=f"rpt_cancel_{tab_name}_{exp_name}", use_container_width=True):
                st.session_state[edit_key] = False
                st.session_state.pop("report_author", None)
                st.rerun(scope="fragment")


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


@st.cache_data(ttl=3600)
def _list_subfolders(folder_id: str) -> dict[str, str]:
    """Returns {name: id} for direct subfolders only."""
    return {f["name"]: f["id"] for f in _drive_list(
        f"'{folder_id}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )}


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

def _ml_folder_num(ml_folder: str) -> int:
    m = re.match(r"ml-(\d+)", ml_folder)
    return int(m.group(1)) if m else -1

def _gnn_folder_num(folder: str) -> int:
    m = re.search(r"(\d+)", folder)
    return int(m.group(1)) if m else -1


def _catalog_filename(ml_folder: str) -> str:
    return _woe_iv_folder_name(ml_folder).replace("-", "") + "_feature_catalog.csv"

def _is_valid_rep(rep: dict) -> bool:
    woe_folder = _woe_iv_folder_name(rep.get("ml_folder", ""))
    return bool(re.match(r"ml-\d+", woe_folder)) and woe_folder != "ml-00"


def _exp_label(rep: dict, is_rep: bool = False) -> str:
    folder = _woe_iv_folder_name(rep["ml_folder"]).upper()
    if is_rep:
        desc = rep.get("description", "")
        return f"* {folder}{(' — ' + desc) if desc else ''}"
    run_id = rep.get("run_id", "")
    model_run_id = rep.get("model_run_id", "")
    return f"   {folder} — {run_id}{model_run_id}"


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


# ── Experiment discovery ───────────────────────────────────────────────────

def _discover_all_ml_runs(ml_folder_id: str) -> list[dict]:
    """ml/ 하위의 모든 실험 run을 탐색. 폴더 목록만 조회하고 파일 다운로드는 없음."""
    from concurrent.futures import ThreadPoolExecutor

    ml_exp_folders = _list_subfolders(ml_folder_id)
    valid_ml = sorted(
        [(name, fid) for name, fid in ml_exp_folders.items()
         if re.match(r"ml-\d+", name) and name != "ml-00"]
    )
    if not valid_ml:
        return []

    def _runs_for_folder(name_fid: tuple) -> list[dict]:
        folder_name, folder_id = name_fid
        run_folders = _list_subfolders(folder_id)
        runs: list[dict] = []
        for run_name, run_fid in sorted(run_folders.items()):
            files = _list_files(run_fid)
            prefixes = sorted(
                re.match(r"(.+)_metrics_val\.json$", fn).group(1)
                for fn in files if re.match(r"(.+)_metrics_val\.json$", fn)
            )
            for prefix in prefixes:
                runs.append({"ml_folder": folder_name, "run_id": run_name, "prefix": prefix})
        return runs

    all_runs: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for runs in ex.map(_runs_for_folder, valid_ml):
            all_runs.extend(runs)
    return sorted(all_runs, key=lambda x: (x["ml_folder"], x["run_id"], x["prefix"]))


def _discover_all_gnn_runs(project_folder_id: str) -> list[dict]:
    """gnn/ 하위의 모든 실험 run을 탐색. 폴더 목록만 조회하고 파일 다운로드는 없음."""
    from concurrent.futures import ThreadPoolExecutor

    gnn_id = _get_folder_id(project_folder_id, "gnn") if project_folder_id else ""
    if not gnn_id:
        return []
    exp_folders = _list_subfolders(gnn_id)
    valid_exps  = sorted(
        [(name, fid) for name, fid in exp_folders.items() if not name.endswith(".json")]
    )
    if not valid_exps:
        return []

    def _runs_for_exp(name_fid: tuple) -> list[dict]:
        folder_name, folder_id = name_fid
        run_folders = _list_subfolders(folder_id)
        return [
            {"folder": folder_name, "run_id": run_name}
            for run_name in sorted(run_folders.keys())
        ]

    all_runs: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for runs in ex.map(_runs_for_exp, valid_exps):
            all_runs.extend(runs)
    return sorted(all_runs, key=lambda x: (x["folder"], x["run_id"]))


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


@st.cache_data(ttl=3600)
def _load_gnn_representatives(project_folder_id: str) -> list[dict]:
    gnn_id = _get_folder_id(project_folder_id, "gnn")
    if not gnn_id:
        return []
    files = _drive_list(
        f"'{gnn_id}' in parents"
        " and name='gnn_leaderboard_representatives.json'"
        " and trashed=false"
    )
    return _download_json(files[0]["id"]) if files else []


def _gnn_exp_label(rep: dict, is_rep: bool = False) -> str:
    folder = rep.get("folder", "")
    if is_rep:
        desc = rep.get("description", "")
        return f"* {folder}{(' — ' + desc) if desc else ''}"
    return f"   {folder} — {rep['run_id']}"



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
        height=max(400, n * 28),
        margin=dict(t=40, b=20),
        yaxis=dict(scaleanchor="x", tickfont_size=9),
        xaxis=dict(showticklabels=False, ticks=""),
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


# ── Mann-Whitney U + FDR ──────────────────────────────────────────────────

def _compute_mann_whitney(indiv_df: pd.DataFrame, feat_names: list[str]) -> pd.DataFrame:
    """TP/FP/FN 쌍별 Mann-Whitney U 검정 + BH FDR 보정, rank-biserial r 반환."""
    from scipy import stats
    try:
        from scipy.stats import false_discovery_control as _fdr
    except ImportError:
        _fdr = None  # scipy < 1.11 fallback: Bonferroni

    comparisons = [("TP", "TN"), ("TP", "FP"), ("TP", "FN"),
                   ("FP", "TN"), ("FN", "TN"), ("FP", "FN")]
    rows = []
    for g1, g2 in comparisons:
        d1 = indiv_df[indiv_df["group"] == g1][feat_names].values
        d2 = indiv_df[indiv_df["group"] == g2][feat_names].values
        if len(d1) < 5 or len(d2) < 5:
            continue
        p_vals, u_vals = [], []
        for i in range(len(feat_names)):
            res = stats.mannwhitneyu(d1[:, i], d2[:, i], alternative="two-sided")
            p_vals.append(res.pvalue)
            u_vals.append(res.statistic)
        import numpy as _np
        p_arr = _np.array(p_vals)
        p_adj = _fdr(p_arr, method="bh") if _fdr is not None else _np.minimum(p_arr * len(p_arr), 1.0)
        for f, u, pa in zip(feat_names, u_vals, p_adj):
            n1, n2 = len(d1), len(d2)
            r = (2 * u) / (n1 * n2) - 1   # rank-biserial = (U1-U2)/(n1*n2): 양수 = g1이 더 높음
            rows.append({"comparison": f"{g1} vs {g2}", "feature": f,
                         "r": round(r, 3), "p_adj": float(pa),
                         "sig": "*" if pa < 0.05 else ""})
    return pd.DataFrame(rows)


_MW_INTERP: dict[tuple, tuple[str, str]] = {
    # (comparison, r>0) → (_, interpretation)  |  r>0 = 앞 그룹 saliency 높음
    ("TP vs TN", True):  ("", "TP > TN: 이상 탐지 시 이 피처가 정상 판별보다 더 강하게 활성화"),
    ("TP vs TN", False): ("", "TN > TP: 정상 판별 시 이 피처가 더 강하게 활성화"),
    ("TP vs FP", True):  ("", "TP > FP: 정탐이 오탐보다 이 피처 saliency 높음"),
    ("TP vs FP", False): ("", "FP > TP: 오탐(FP)에서 이 피처가 더 강하게 활성화 → 오탐 원인 후보"),
    ("TP vs FN", True):  ("", "TP > FN: 탐지 성공한 이상거래에서 이 피처 신호가 더 강함"),
    ("TP vs FN", False): ("", "FN > TP: 미탐 케이스에서 이 피처 saliency가 더 높음"),
    ("FP vs TN", True):  ("", "FP > TN: 오탐(FP)이 정상보다 이 피처 강하게 활성화 → 이 피처들이 오탐 유발"),
    ("FP vs TN", False): ("", "TN > FP: 정상(TN)이 오탐보다 이 피처 더 활성화"),
    ("FN vs TN", True):  ("", "FN > TN: 미탐(FN)이 정상보다 이 피처 더 활성화 → 미탐 거래도 이 피처에서 이상 신호 존재"),
    ("FN vs TN", False): ("", "TN > FN: 정상(TN)이 미탐보다 이 피처 더 활성화"),
    ("FP vs FN", True):  ("", "FP > FN: 오탐이 미탐보다 이 피처 더 강하게 활성화"),
    ("FP vs FN", False): ("", "FN > FP: 미탐이 오탐보다 이 피처 더 활성화"),
}

def _mw_effect_label(r: float) -> str:
    a = abs(r)
    if a >= 0.7: return "매우 강한 효과"
    if a >= 0.5: return "강한 효과"
    if a >= 0.3: return "중간 효과"
    return "약한 효과"


@st.cache_data(ttl=3600)
def _make_mw_heatmap(indiv_json: str, feat_names: tuple) -> go.Figure:
    """Mann-Whitney effect size heatmap (rank-biserial r)."""
    import pandas as _pd
    indiv_df = _pd.read_json(StringIO(indiv_json))
    mw_df    = _compute_mann_whitney(indiv_df, list(feat_names))
    if mw_df.empty:
        return go.Figure()

    _comp_labels = {
        "TP vs TN": "TP vs TN<br><sub>탐지 기준</sub>",
        "TP vs FP": "TP vs FP<br><sub>오탐 원인</sub>",
        "TP vs FN": "TP vs FN<br><sub>미탐 원인</sub>",
        "FP vs TN": "FP vs TN<br><sub>오탐 특성</sub>",
        "FN vs TN": "FN vs TN<br><sub>미탐 특성</sub>",
        "FP vs FN": "FP vs FN<br><sub>두 오류 비교</sub>",
    }
    comparisons     = mw_df["comparison"].unique().tolist()
    comparisons_lbl = [_comp_labels.get(c, c) for c in comparisons]
    r_mat, t_mat, cd_mat = [], [], []
    for f in feat_names:
        r_row, t_row, cd_row = [], [], []
        for c in comparisons:
            sub = mw_df[(mw_df["feature"] == f) & (mw_df["comparison"] == c)]
            if sub.empty:
                r_row.append(None); t_row.append(""); cd_row.append(["", "", ""])
            else:
                rv   = sub.iloc[0]["r"]
                p    = sub.iloc[0]["p_adj"]
                _, interp = _MW_INTERP.get((c, rv > 0), ("", ""))
                r_row.append(rv)
                t_row.append(f"{rv:+.2f}{sub.iloc[0]['sig']}")
                cd_row.append([f"{p:.4f}", interp, _mw_effect_label(rv)])
        r_mat.append(r_row)
        t_mat.append(t_row)
        cd_mat.append(cd_row)

    fig = go.Figure(go.Heatmap(
        z=r_mat, x=comparisons_lbl, y=list(feat_names),
        text=t_mat, texttemplate="%{text}",
        customdata=cd_mat,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        showscale=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{x}<br>"
            "<b>r = %{z:+.3f}</b>  <i>(%{customdata[2]})</i><br>"
            "p (FDR) = %{customdata[0]}<br>"
            "<br>%{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=max(260, 42 * len(feat_names) + 110),
        margin=dict(t=20, b=60, l=10, r=10),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


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


# ── On-demand experiment loaders ──────────────────────────────────────────

def _compute_exp_data(rep: dict, ml_fid: str, woe_root_id: str) -> dict:
    """Load one ML experiment's files in parallel. Pure — no Streamlit state access."""
    from concurrent.futures import ThreadPoolExecutor

    ml_exp_fid  = _get_folder_id(ml_fid, rep["ml_folder"]) if ml_fid else ""
    run_fid     = _get_folder_id(ml_exp_fid, rep["run_id"]) if ml_exp_fid else ""
    woe_iv_name = _woe_iv_folder_name(rep["ml_folder"])
    woe_iv_fid  = _get_folder_id(woe_root_id, woe_iv_name) if woe_root_id else ""
    prefix      = _artifact_prefix(rep)
    cat_fn      = _catalog_filename(rep["ml_folder"])

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_woe = ex.submit(_load_woe_results, woe_iv_fid)
        f_cat = ex.submit(_load_catalog, ml_exp_fid, cat_fn)
        f_ml  = ex.submit(_load_ml_results, run_fid, prefix) if run_fid else None
        woe     = f_woe.result()
        catalog = f_cat.result()
        ml      = f_ml.result() if f_ml else {}

    cached_pfx   = (woe.get("meta", {}) or {}).get("prefix") if woe else None
    stale_status = ("no_woe" if not woe or "iv_df" not in woe
                    else "stale" if cached_pfx != prefix
                    else "fresh")
    return {
        "rep": rep, "ml": ml, "woe": woe, "catalog": catalog,
        "stale_status": stale_status, "prefix": prefix, "_loaded": True,
    }


def _compute_gnn_exp_data(rep: dict, project_folder_id: str) -> dict:
    """Load one GNN experiment's files in parallel. Pure — no Streamlit state access."""
    from concurrent.futures import ThreadPoolExecutor

    gnn_id = _get_folder_id(project_folder_id, "gnn") if project_folder_id else ""
    if not gnn_id:
        return {"rep": rep, "d": {}, "_loaded": True}
    exp_id = _get_folder_id(gnn_id, rep["folder"])
    if not exp_id:
        return {"rep": rep, "d": {}, "_loaded": True}
    run_fid = _get_folder_id(exp_id, rep["run_id"])
    if not run_fid:
        return {"rep": rep, "d": {}, "_loaded": True}

    # 하위 폴더 ID 병렬 조회
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_logs   = ex.submit(_get_folder_id, run_fid, "logs")
        f_models = ex.submit(_get_folder_id, run_fid, "models")
        f_fi_dir = ex.submit(_get_folder_id, run_fid, "feature_importance")
        logs_id   = f_logs.result()
        models_id = f_models.result()
        fi_dir_id = f_fi_dir.result()

    def _fetch_log():
        if not logs_id:
            return None
        lf      = _list_files(logs_id)
        log_fns = [n for n in lf if n.endswith(".log")]
        if not log_fns:
            return None
        r = requests.get(
            f"https://drive.google.com/uc?export=download&id={lf[log_fns[0]]}",
            timeout=60,
        )
        return _parse_gnn_log(r.text) if r.ok else None

    def _fetch_args():
        if not models_id:
            return None
        mf       = _list_files(models_id)
        args_fns = [n for n in mf if n.endswith("_args.json")]
        return _download_json(mf[args_fns[0]]) if args_fns else None

    def _fetch_fi():
        if not fi_dir_id:
            return None, None
        fi_files     = _list_files(fi_dir_id)
        fi_fns       = [n for n in fi_files
                        if n.endswith("_feature_importance.csv") and "individual" not in n]
        fi_indiv_fns = [n for n in fi_files
                        if n.endswith("_feature_importance_individual.csv")]
        fi       = _download_csv(fi_files[fi_fns[0]])       if fi_fns       else None
        fi_indiv = _download_csv(fi_files[fi_indiv_fns[0]]) if fi_indiv_fns else None
        return fi, fi_indiv

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_parsed = ex.submit(_fetch_log)
        f_args   = ex.submit(_fetch_args)
        f_fi_res = ex.submit(_fetch_fi)
        parsed       = f_parsed.result()
        args_data    = f_args.result()
        fi, fi_indiv = f_fi_res.result()

    out: dict = {}
    if parsed   is not None: out["parsed"]                        = parsed
    if args_data is not None: out["args"]                         = args_data
    if fi        is not None: out["feature_importance"]           = fi
    if fi_indiv  is not None: out["feature_importance_individual"] = fi_indiv

    return {"rep": rep, "d": out, "_loaded": True}


def _load_exp_data(label: str) -> None:
    """Load one ML experiment and store in session_state (main thread only)."""
    entry      = st.session_state["exp_data"][label]
    is_rep     = entry.get("is_rep", False)
    is_ongoing = entry.get("is_ongoing", False)
    result     = _compute_exp_data(
        entry["rep"],
        st.session_state.get("_ml_folder_id", ""),
        st.session_state.get("_woe_iv_root_id", ""),
    )
    result["is_rep"]     = is_rep
    result["is_ongoing"] = is_ongoing
    st.session_state["exp_data"][label] = result


def _load_gnn_exp_data(label: str) -> None:
    """Load one GNN experiment and store in session_state (main thread only)."""
    entry      = st.session_state["gnn_exp_data"][label]
    is_rep     = entry.get("is_rep", False)
    is_ongoing = entry.get("is_ongoing", False)
    result     = _compute_gnn_exp_data(entry["rep"], st.session_state.get("_project_folder_id", ""))
    result["is_rep"]     = is_rep
    result["is_ongoing"] = is_ongoing
    st.session_state["gnn_exp_data"][label] = result


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
gnn_reps   = _load_gnn_representatives(PROJECT_FOLDER_ID)

# ── 폴더 ID 세션 저장 (loader 함수에서 참조) ──────────────────────────────
st.session_state["_ml_folder_id"]      = ml_folder_id
st.session_state["_woe_iv_root_id"]    = woe_iv_root_id
st.session_state["_project_folder_id"] = PROJECT_FOLDER_ID

# ── 전체 실험 탐색 (폴더 목록만, 다운로드 없음) ────────────────────────────
with st.spinner("실험 목록 탐색 중..."):
    all_ml_runs  = _discover_all_ml_runs(ml_folder_id)
    all_gnn_runs = _discover_all_gnn_runs(PROJECT_FOLDER_ID)

if not all_ml_runs:
    st.warning("탐색된 ML 실험이 없습니다. PROJECT_FOLDER_ID와 Drive 구조를 확인하세요.")
    st.stop()

# ── 대표 실험 맵 ───────────────────────────────────────────────────────────
_rep_map     = {(_woe_iv_folder_name(r["ml_folder"]), r["run_id"], r["model_run_id"]): r for r in valid_reps}
_gnn_rep_map = {(r["folder"], r["run_id"]): r                          for r in gnn_reps}

# ── ML 실험 메타데이터 초기화 (파일 다운로드 없음) ─────────────────────────
_ml_fp       = tuple((r["ml_folder"], r["run_id"], r["prefix"])    for r in all_ml_runs)
_rp_fp       = tuple((r["ml_folder"], _artifact_prefix(r))         for r in valid_reps)
_combined_fp = (_ml_fp, _rp_fp)
if st.session_state.get("_exp_data_fp") != _combined_fp:
    _exp_init: dict[str, dict] = {}
    for _run in all_ml_runs:
        _parts  = _run["prefix"].split("__", 2)
        _model_run_id = _parts[2] if len(_parts) >= 3 else _run["prefix"]
        _rep_e  = _rep_map.get((_woe_iv_folder_name(_run["ml_folder"]), _run["run_id"], _model_run_id))
        _is_rep = _rep_e is not None
        _srep   = _rep_e or {
            "ml_folder":     _run["ml_folder"],
            "run_id":        _run["run_id"],
            "experiment_id": _parts[0] if len(_parts) >= 1 else "",
            "model_run_id":  _model_run_id,
            "description":   "",
            "note":          "",
        }
        _lbl = _exp_label(_srep, _is_rep)
        _exp_init[_lbl] = {
            "rep": _srep, "is_rep": _is_rep,
            "ml": {}, "woe": {}, "catalog": None,
            "stale_status": "unknown", "prefix": _run["prefix"], "_loaded": False,
        }
    st.session_state["exp_data"]    = _exp_init
    st.session_state["_exp_data_fp"] = _combined_fp

exp_data: dict[str, dict] = st.session_state["exp_data"]

if not exp_data:
    st.warning("로드 가능한 실험이 없습니다.")
    st.stop()

exp_labels   = list(exp_data.keys())
_default_idx = 0
st.session_state["_exp_labels"] = exp_labels

# ── GNN 실험 메타데이터 초기화 (파일 다운로드 없음) ───────────────────────
st.session_state["_gnn_reps"] = gnn_reps

_gnn_fp       = tuple((r["folder"], r["run_id"]) for r in all_gnn_runs)
_grp_fp       = tuple((r["folder"], r["run_id"]) for r in gnn_reps)
_combined_gfp = (_gnn_fp, _grp_fp)
if st.session_state.get("_gnn_exp_data_fp") != _combined_gfp:
    _gnn_init: dict[str, dict] = {}
    for _run in all_gnn_runs:
        _rep_e  = _gnn_rep_map.get((_run["folder"], _run["run_id"]))
        _is_rep = _rep_e is not None
        _srep   = _rep_e or {"folder": _run["folder"], "run_id": _run["run_id"], "description": "", "note": ""}
        _lbl    = _gnn_exp_label(_srep, _is_rep)
        _gnn_init[_lbl] = {"rep": _srep, "is_rep": _is_rep, "d": {}, "_loaded": False}
    st.session_state["gnn_exp_data"]    = _gnn_init
    st.session_state["_gnn_exp_data_fp"] = _combined_gfp

gnn_exp_data: dict[str, dict] = st.session_state.get("gnn_exp_data", {})

# ── Ongoing 실험 마킹 (대표보다 높은 넘버링의 최신 비대표 실험) ──────────────
_max_rep_ml_n  = max((_ml_folder_num(d["rep"]["ml_folder"])  for d in exp_data.values()     if d.get("is_rep")), default=-1)
_max_all_ml_n  = max((_ml_folder_num(d["rep"]["ml_folder"])  for d in exp_data.values()),     default=-1)
_max_rep_gnn_n = max((_gnn_folder_num(d["rep"]["folder"])    for d in gnn_exp_data.values() if d.get("is_rep")), default=-1)
_max_all_gnn_n = max((_gnn_folder_num(d["rep"]["folder"])    for d in gnn_exp_data.values()), default=-1)

for _d in exp_data.values():
    _d["is_ongoing"] = (
        not _d.get("is_rep") and
        _max_all_ml_n > _max_rep_ml_n and
        _ml_folder_num(_d["rep"]["ml_folder"]) == _max_all_ml_n
    )
for _d in gnn_exp_data.values():
    _d["is_ongoing"] = (
        not _d.get("is_rep") and
        _max_all_gnn_n > _max_rep_gnn_n and
        _gnn_folder_num(_d["rep"]["folder"]) == _max_all_gnn_n
    )


# ══════════════════════════════════════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_gnn, tab_ml, tab_woe = st.tabs(["Overview", "GNN Result", "ML Result", "Univariate Analysis"])


# ──────────────────────────────────────────────────────────────────────────────
# 탭 0: Overview
# ──────────────────────────────────────────────────────────────────────────────
with tab_overview:
    # ── 미로드 실험 병렬 로드 ─────────────────────────────────────────────
    _unloaded_ml  = [lbl for lbl in exp_data     if not exp_data[lbl].get("_loaded")     and (exp_data[lbl].get("is_rep")     or exp_data[lbl].get("is_ongoing"))]
    _unloaded_gnn = [lbl for lbl in gnn_exp_data if not gnn_exp_data[lbl].get("_loaded") and (gnn_exp_data[lbl].get("is_rep") or gnn_exp_data[lbl].get("is_ongoing"))]
    if _unloaded_ml or _unloaded_gnn:
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        _ml_fid      = st.session_state.get("_ml_folder_id", "")
        _woe_root_id = st.session_state.get("_woe_iv_root_id", "")
        _pfid        = st.session_state.get("_project_folder_id", "")
        _ml_args  = [(lbl, exp_data[lbl]["rep"])     for lbl in _unloaded_ml]
        _gnn_args = [(lbl, gnn_exp_data[lbl]["rep"]) for lbl in _unloaded_gnn]
        _total = len(_ml_args) + len(_gnn_args)
        _bar   = st.progress(0, text=f"데이터 로드 중 (0/{_total})...")
        with ThreadPoolExecutor(max_workers=6) as _ex:
            _ml_futs  = {_ex.submit(_compute_exp_data,     rep, _ml_fid, _woe_root_id): lbl
                         for lbl, rep in _ml_args}
            _gnn_futs = {_ex.submit(_compute_gnn_exp_data, rep, _pfid):                 lbl
                         for lbl, rep in _gnn_args}
            _all_futs = {**_ml_futs, **_gnn_futs}
            _unloaded_ml_set = set(_unloaded_ml)
            for _n, _fut in enumerate(_as_completed(_all_futs), 1):
                _lbl = _all_futs[_fut]
                try:
                    _result = _fut.result()
                    if _lbl in _unloaded_ml_set:
                        _result["is_rep"]     = exp_data.get(_lbl, {}).get("is_rep", False)
                        _result["is_ongoing"] = exp_data.get(_lbl, {}).get("is_ongoing", False)
                        st.session_state["exp_data"][_lbl] = _result
                    else:
                        _result["is_rep"]     = gnn_exp_data.get(_lbl, {}).get("is_rep", False)
                        _result["is_ongoing"] = gnn_exp_data.get(_lbl, {}).get("is_ongoing", False)
                        st.session_state["gnn_exp_data"][_lbl] = _result
                except Exception as _e:
                    st.warning(f"'{_lbl}' 로드 실패: {_e}")
                _bar.progress(_n / _total, text=f"데이터 로드 중 ({_n}/{_total}): {_lbl}")
        _bar.empty()
        exp_data     = st.session_state["exp_data"]
        gnn_exp_data = st.session_state.get("gnn_exp_data", {})

    # Overview: 대표 실험 + ongoing 실험 표시
    _ov_exp = {lbl: d for lbl, d in exp_data.items()     if d.get("is_rep") or d.get("is_ongoing")}
    _ov_gnn = {lbl: d for lbl, d in gnn_exp_data.items() if d.get("is_rep") or d.get("is_ongoing")}

    st.markdown("#### Project Summary")

    # ── 최신 실험 F1 불릿 차트 ───────────────────────────────────────────────
    _bullet_rows: list[dict] = []

    _ml_sorted = sorted([(lbl, d) for lbl, d in _ov_exp.items() if d.get("is_rep")],
                        key=lambda kv: _woe_iv_folder_name(kv[1]["rep"]["ml_folder"]))
    _ml_all_f1s = []
    for _, _d in _ml_sorted:
        _m = _d["ml"].get("metrics", {}); _m = _m.get("metrics", _m)
        _v = _m.get("f1")
        if _v is not None: _ml_all_f1s.append(_v)
    if _ml_sorted:
        _, _ld = _ml_sorted[-1]
        _lm = _ld["ml"].get("metrics", {})
        _lm = _lm.get("metrics", _lm)
        _lf1 = _lm.get("f1")
        if _lf1 is not None:
            _bullet_rows.append({
                "label": f"ML  ({_woe_iv_folder_name(_ld['rep']['ml_folder']).upper()})",
                "f1": _lf1, "max_f1": max(_ml_all_f1s) if _ml_all_f1s else _lf1,
                "color": "#fbbf24",
            })

    _gnn_sorted = sorted([(lbl, d) for lbl, d in _ov_gnn.items() if d.get("is_rep")],
                         key=lambda kv: kv[1]["rep"]["folder"])
    _gnn_all_f1s = []
    for _, _d in _gnn_sorted:
        _gp2 = _d["d"].get("parsed", {}); _ge2 = _gp2.get("epochs", [])
        if not _ge2: continue
        _gdf2 = pd.DataFrame(_ge2); _gdf2.index = _gdf2.index + 1; _gdf2.index.name = "epoch"; _gdf2 = _gdf2.reset_index()
        _glb2 = _gp2.get("best_epoch")
        _gm2  = _gdf2[_gdf2["epoch"] == _glb2].index if _glb2 is not None else pd.Index([])
        _gi2  = _gm2[0] if len(_gm2) else _gdf2["val_auprc"].idxmax()
        _gv2  = _gdf2.loc[_gi2].get("test_f1")
        if _gv2 is not None: _gnn_all_f1s.append(_gv2)
    if _gnn_sorted:
        _, _gd = _gnn_sorted[-1]
        _gp = _gd["d"].get("parsed", {})
        _ge = _gp.get("epochs", [])
        if _ge:
            _gdf = pd.DataFrame(_ge)
            _gdf.index = _gdf.index + 1
            _gdf.index.name = "epoch"
            _gdf = _gdf.reset_index()
            _glb = _gp.get("best_epoch")
            if _glb is not None:
                _gm = _gdf[_gdf["epoch"] == _glb].index
                _gi = _gm[0] if len(_gm) else _gdf["val_auprc"].idxmax()
            else:
                _gi = _gdf["val_auprc"].idxmax()
            _gf1 = _gdf.loc[_gi].get("test_f1")
            if _gf1 is not None:
                _bullet_rows.append({
                    "label": f"GNN  ({_gd['rep']['folder']})",
                    "f1": _gf1, "max_f1": max(_gnn_all_f1s) if _gnn_all_f1s else _gf1,
                    "color": "#f43f5e",
                })

    if _bullet_rows:
        _labels   = [r["label"]   for r in _bullet_rows]
        _f1s      = [r["f1"]      for r in _bullet_rows]
        _max_f1s  = [r["max_f1"]  for r in _bullet_rows]
        _colors   = [r["color"]   for r in _bullet_rows]
        _fig_b    = go.Figure()
        # 배경 바 (전체 범위)
        _fig_b.add_trace(go.Bar(
            y=_labels, x=[1.0] * len(_bullet_rows), orientation="h",
            marker_color="rgba(255,255,255,0.06)", marker_line_width=0,
            showlegend=False, hoverinfo="skip",
        ))
        # 최대값 회색 바
        _fig_b.add_trace(go.Bar(
            y=_labels, x=_max_f1s, orientation="h",
            marker_color="rgba(180,180,180,0.30)", marker_line_width=0,
            showlegend=False,
            hovertemplate="<b>%{y}</b><br>Max F1: %{x:.4f}<extra></extra>",
        ))
        # 현재(최신 대표) 값 컬러 바
        _fig_b.add_trace(go.Bar(
            y=_labels, x=_f1s, orientation="h",
            marker_color=_colors, marker_line_width=0,
            showlegend=False,
            text=[f"F1  {v:.4f}" for v in _f1s],
            textposition="outside",
            textfont=dict(size=13, color="#e2e5ec"),
            hovertemplate="<b>%{y}</b><br>F1: %{x:.4f}<extra></extra>",
        ))
        _fig_b.update_layout(
            barmode="overlay",
            height=110 + 50 * len(_bullet_rows),
            margin=dict(t=10, b=10, l=10, r=130),
            xaxis=dict(
                range=[0, 1.18],
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                gridcolor="rgba(255,255,255,0.08)",
            ),
            yaxis=dict(showgrid=False, autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig_b, use_container_width=True)

    _METRIC_COLORS = {"F1": "#4f9cf9", "AUPRC": "#a78bfa", "Recall": "#34d399"}

    # ── ML ────────────────────────────────────────────────────────────────────
    ml_rows: list[dict] = []
    for label, d in sorted(_ov_exp.items(),
                            key=lambda kv: _woe_iv_folder_name(kv[1]["rep"]["ml_folder"])):
        rep  = d["rep"]
        m    = d["ml"].get("metrics", {})
        m    = m.get("metrics", m)
        desc = rep.get("description", "")
        exp  = _woe_iv_folder_name(rep["ml_folder"]).upper()
        if d.get("is_ongoing"):
            exp = f"{exp} (ongoing)"
        f1      = m.get("f1")
        auprc   = m.get("average_precision") or d["ml"].get("train_summary", {}).get("best_score")
        recall  = m.get("recall")
        for metric, val in [("F1", f1), ("AUPRC", auprc), ("Recall", recall)]:
            if val is not None:
                ml_rows.append({"exp": exp, "metric": metric, "value": val, "description": desc})

    # ── GNN ───────────────────────────────────────────────────────────────────
    gnn_rows:      list[dict] = []
    gnn_time_rows: list[dict] = []
    for label, d in sorted(_ov_gnn.items(),
                            key=lambda kv: kv[1]["rep"]["folder"]):
        rep    = d["rep"]
        parsed = d["d"].get("parsed", {})
        epochs = parsed.get("epochs", [])
        desc   = rep.get("description", "")
        exp    = rep["folder"]
        if d.get("is_ongoing"):
            exp = f"{exp} (ongoing)"
        t_sec  = parsed.get("training_time_sec")
        if t_sec is not None:
            gnn_time_rows.append({"exp": exp, "time_sec": t_sec, "description": desc})
        if epochs:
            _ep_df = pd.DataFrame(epochs)
            _ep_df.index = _ep_df.index + 1
            _ep_df.index.name = "epoch"
            _ep_df = _ep_df.reset_index()
            _log_best = parsed.get("best_epoch")
            if _log_best is not None:
                _match = _ep_df[_ep_df["epoch"] == _log_best].index
                _bi = _match[0] if len(_match) else _ep_df["val_auprc"].idxmax()
            else:
                _bi = _ep_df["val_auprc"].idxmax()
            _bp = _ep_df.loc[_bi]
            for metric, val in [
                ("F1",     _bp.get("test_f1")),
                ("AUPRC",  _bp.get("test_auprc")),
                ("Recall", _bp.get("test_recall")),
            ]:
                if val is not None:
                    gnn_rows.append({"exp": exp, "metric": metric, "value": val, "description": desc})

    def _overview_line(rows: list[dict], title: str) -> None:
        if not rows:
            st.caption(f"{title} — 데이터 없음")
            return
        df_ov = pd.DataFrame(rows)
        exps  = df_ov["exp"].unique().tolist()
        fig   = go.Figure()
        for metric, color in _METRIC_COLORS.items():
            sub  = df_ov[df_ov["metric"] == metric]
            dash = "solid" if metric == "F1" else "dash"
            fig.add_trace(go.Scatter(
                x=sub["exp"],
                y=sub["value"],
                mode="lines+markers",
                name=metric,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=8, color=color),
                customdata=sub["description"],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{metric}: " + "%{y:.4f}<br>"
                    "<i>%{customdata}</i><extra></extra>"
                ),
            ))
        # x label 호버용 투명 마커 (description 표시)
        desc_map = df_ov.drop_duplicates("exp").set_index("exp")["description"]
        fig.add_trace(go.Scatter(
            x=exps,
            y=[0] * len(exps),
            mode="markers",
            marker=dict(opacity=0, size=16),
            customdata=[desc_map.get(e, "") for e in exps],
            hovertemplate="<b>%{x}</b><br><i>%{customdata}</i><extra></extra>",
            showlegend=False,
            name="",
        ))
        fig.update_layout(
            title=title,
            height=300,
            margin=dict(t=40, b=20),
            xaxis=dict(categoryorder="array", categoryarray=exps),
            yaxis=dict(range=[0, 1]),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                itemclick="toggleothers", itemdoubleclick="toggle",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    def _overview_line_gnn(rows: list[dict], time_rows: list[dict], title: str) -> None:
        if not rows:
            st.caption(f"{title} — 데이터 없음")
            return
        from plotly.subplots import make_subplots as _make_subplots
        df_ov   = pd.DataFrame(rows)
        df_time = pd.DataFrame(time_rows) if time_rows else pd.DataFrame(columns=["exp", "time_sec", "description"])
        exps    = df_ov["exp"].unique().tolist()
        fig = _make_subplots(specs=[[{"secondary_y": True}]])
        for metric, color in _METRIC_COLORS.items():
            sub  = df_ov[df_ov["metric"] == metric]
            dash = "solid" if metric == "F1" else "dash"
            fig.add_trace(go.Scatter(
                x=sub["exp"],
                y=sub["value"],
                mode="lines+markers",
                name=metric,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=8, color=color),
                customdata=sub["description"],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{metric}: " + "%{y:.4f}<br>"
                    "<i>%{customdata}</i><extra></extra>"
                ),
            ), secondary_y=False)
        if not df_time.empty:
            fig.add_trace(go.Bar(
                x=df_time["exp"],
                y=df_time["time_sec"],
                name="학습시간 (s)",
                width=0.3,
                marker_color="rgba(255,200,80,0.25)",
                marker_line=dict(color="rgba(255,200,80,0.6)", width=1),
                customdata=df_time["description"],
                hovertemplate=(
                    "<b>%{x}</b><br>학습시간: %{y:.0f}s<br>"
                    "<i>%{customdata}</i><extra></extra>"
                ),
            ), secondary_y=True)
        # x label 호버용 투명 마커
        desc_map = df_ov.drop_duplicates("exp").set_index("exp")["description"]
        fig.add_trace(go.Scatter(
            x=exps, y=[0] * len(exps), mode="markers",
            marker=dict(opacity=0, size=16),
            customdata=[desc_map.get(e, "") for e in exps],
            hovertemplate="<b>%{x}</b><br><i>%{customdata}</i><extra></extra>",
            showlegend=False, name="",
        ), secondary_y=False)
        fig.update_layout(
            title=title, height=320, margin=dict(t=40, b=20),
            xaxis=dict(categoryorder="array", categoryarray=exps),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                itemclick="toggleothers", itemdoubleclick="toggle",
            ),
            barmode="overlay",
        )
        fig.update_yaxes(range=[0, 1], title_text="Metric", secondary_y=False)
        fig.update_yaxes(title_text="학습시간 (s)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    _overview_line(ml_rows, "ML")
    _overview_line_gnn(gnn_rows, gnn_time_rows, "GNN")


# ──────────────────────────────────────────────────────────────────────────────
# 탭 1: GNN 결과
# ──────────────────────────────────────────────────────────────────────────────
@st.fragment
def _tab_gnn_render():
    gnn_exp_data = st.session_state.get("gnn_exp_data", {})

    if not gnn_exp_data:
        st.info("탐색된 GNN 실험이 없습니다.")
    else:
        _gnn_labels   = list(gnn_exp_data.keys())
        sel_gnn_label = st.selectbox("실험 선택", _gnn_labels, key="gnn_sel",
                                     label_visibility="collapsed")
        sel_gnn_rep   = gnn_exp_data[sel_gnn_label]["rep"]
        if not gnn_exp_data.get(sel_gnn_label, {}).get("_loaded"):
            with st.spinner("실험 데이터 로드 중..."):
                _load_gnn_exp_data(sel_gnn_label)
            gnn_exp_data = st.session_state.get("gnn_exp_data", {})
        _gnn_note = sel_gnn_rep.get("note", "")
        if _gnn_note:
            st.caption(f"**Note**: {_gnn_note}")
        _render_report("GNN Result", sel_gnn_rep["folder"])
        st.divider()
        gnn_d  = gnn_exp_data.get(sel_gnn_label, {}).get("d", {})

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

            _log_best = parsed.get("best_epoch")
            if _log_best is not None:
                _match = ep_df[ep_df["epoch"] == _log_best].index
                best_idx = _match[0] if len(_match) else ep_df["val_auprc"].idxmax()
            else:
                best_idx = ep_df["val_auprc"].idxmax()
            best_ep  = ep_df.loc[best_idx]

            # ── 핵심 지표 ──────────────────────────────────────────────────
            _model_name = args.get("model", "—") if args else "—"
            st.markdown(f"#### Metrics (model: {_model_name})")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("F1",             f"{best_ep['test_f1']:.4f}")
            c2.metric("AUPRC",          f"{best_ep['test_auprc']:.4f}")
            c3.metric("Recall",         f"{best_ep['test_recall']:.4f}")
            c4.metric("Precision",      f"{best_ep['test_precision']:.4f}")
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

                # ── Group Comparison (Mann-Whitney U) ─────────────────────
                fi_indiv = gnn_d.get("feature_importance_individual")
                with st.expander("Group Comparison (Mann-Whitney U, FDR corrected)"):
                    if fi_indiv is not None and not fi_indiv.empty:
                        fig_mw = _make_mw_heatmap(
                            fi_indiv.to_json(orient="records"),
                            tuple(feat_names),
                        )
                        st.plotly_chart(fig_mw, use_container_width=True)
                        st.caption("\\* FDR < 0.05 (Benjamini-Hochberg)  |  r: rank-biserial correlation  |  양수 = 왼쪽 그룹이 saliency 더 높음")
                    else:
                        st.info("개별 샘플 파일 없음 (feature_importance_individual.csv 확인)")
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



with tab_gnn:
    _tab_gnn_render()


# ──────────────────────────────────────────────────────────────────────────────
# 탭 2: ML 결과
# ──────────────────────────────────────────────────────────────────────────────
@st.fragment
def _tab_ml_render():
    exp_data   = st.session_state.get("exp_data", {})
    exp_labels = list(exp_data.keys())
    _default_idx = 0

    sel = st.selectbox("실험 선택", exp_labels, key="ml_sel",
                       index=_default_idx, label_visibility="collapsed")

    if not sel or sel not in exp_data:
        sel = next((lbl for lbl in exp_labels if lbl in exp_data), None)
        if not sel:
            st.info("로드 가능한 실험이 없습니다.")
            return

    # 실험 변경 시 이전 fi_* 키 정리
    if st.session_state.get("_ml_prev_sel") != sel:
        prev = st.session_state.get("_ml_prev_sel")
        if prev:
            for k in [f"fi_sel_{prev}", f"fi_bar_ver_{prev}", f"fi_scat_ver_{prev}"]:
                st.session_state.pop(k, None)
        st.session_state["_ml_prev_sel"] = sel

    if not exp_data.get(sel, {}).get("_loaded"):
        with st.spinner("실험 데이터 로드 중..."):
            _load_exp_data(sel)
        exp_data = st.session_state.get("exp_data", {})

    d   = exp_data[sel]
    rep = d["rep"]
    ml  = d["ml"]

    _note = rep.get("note", "")
    if _note:
        st.caption(f"**Note**: {_note}")

    _render_report("ML Result", _woe_iv_folder_name(rep["ml_folder"]))
    st.divider()

    if not ml:
        st.info("학습된 모델이 없습니다.")
    else:
        metrics_raw   = ml.get("metrics", {})
        train_summary = ml.get("train_summary", {})
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
        c3.metric("Recall",    f"{m.get('recall', 0):.4f}")
        c4.metric("Precision", f"{m.get('precision', 0):.4f}")
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
            diag  = train_summary.get("xgboost_diagnostics", {})
            evals = diag.get("evals_result", {})

            # metric → (train_vals, val_vals)
            available: dict[str, tuple] = {}
            if evals:
                keys = list(evals.keys())
                train_key, val_key = (keys[0], keys[1]) if len(keys) >= 2 else (keys[0], None) if keys else (None, None)
                if train_key and val_key and evals.get(train_key) and evals.get(val_key):
                    for metric, t_vals in evals[train_key].items():
                        if metric in evals[val_key]:
                            available[metric] = (tuple(t_vals), tuple(evals[val_key][metric]))

            if available:
                _lc_display = {"aucpr": "AUPRC", "logloss": "LogLoss"}
                metric_options = [_lc_display.get(m, m) for m in available]
                _lc_raw = {_lc_display.get(m, m): m for m in available}
                lc_sel = st.radio("지표", metric_options, horizontal=True,
                                  key=f"lc_metric_{sel}", label_visibility="collapsed")
                t_v, v_v = available[_lc_raw[lc_sel]]
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

            with st.expander("Feature Correlation Heatmap"):
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
                st.rerun(scope="fragment")
            elif _from_bar and _from_bar != _cur:
                st.session_state[f"fi_sel_{sel}"]        = _from_bar
                st.session_state[f"fi_scat_ver_{sel}"]   = _scat_ver + 1  # 버블 선택 초기화
                st.rerun(scope="fragment")
        else:
            st.info("Feature importance 파일 없음")



with tab_ml:
    _tab_ml_render()


# ──────────────────────────────────────────────────────────────────────────────
# 탭 2: WOE / IV
# ──────────────────────────────────────────────────────────────────────────────
@st.fragment
def _tab_woe_render():
    exp_data   = st.session_state.get("exp_data", {})
    exp_labels = list(exp_data.keys())
    _default_idx = 0

    sel_woe = st.selectbox("실험 선택", exp_labels, key="woe_sel",
                           index=_default_idx, label_visibility="collapsed")

    if not sel_woe or sel_woe not in exp_data:
        sel_woe = next((lbl for lbl in exp_labels if lbl in exp_data), None)
        if not sel_woe:
            st.info("로드 가능한 실험이 없습니다.")
            return

    # 실험 변경 시 이전 catalog 편집 상태 정리
    if st.session_state.get("_woe_prev_sel") != sel_woe:
        prev_woe = st.session_state.get("_woe_prev_sel")
        if prev_woe:
            st.session_state.pop(f"catalog_{prev_woe}", None)
        st.session_state["_woe_prev_sel"] = sel_woe

    if not exp_data.get(sel_woe, {}).get("_loaded"):
        with st.spinner("실험 데이터 로드 중..."):
            _load_exp_data(sel_woe)
        exp_data = st.session_state.get("exp_data", {})

    _render_report("Univariate Analysis", _woe_iv_folder_name(exp_data[sel_woe]["rep"]["ml_folder"]))
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
                    st.rerun(scope="fragment")
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


with tab_woe:
    _tab_woe_render()
