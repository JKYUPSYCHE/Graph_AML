"""
돈무브 프로젝트 대시보드

Secrets (.streamlit/secrets.toml):
    PROJECT_FOLDER_ID           Drive root folder ID
    [gcp_service_account]       서비스 계정 JSON 내용

Drive layout (PROJECT_FOLDER_ID 하위):
    ml/
        ml_dashboard_representatives.json
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
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="돈무브 프로젝트 대시보드", layout="wide", page_icon="📊")

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
            "fields": "files(id,name,modifiedTime)",
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


def _download_json(file_id: str) -> object:
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _download_csv(file_id: str) -> pd.DataFrame:
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=30,
    )
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content), encoding="utf-8-sig")


@st.cache_data(ttl=300)
def _get_folder_id(parent_id: str, name: str) -> str:
    folders = _drive_list(
        f"'{parent_id}' in parents"
        f" and name='{name}'"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )
    return folders[0]["id"] if folders else ""


@st.cache_data(ttl=300)
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

@st.cache_data(ttl=300)
def _load_representatives(ml_folder_id: str) -> list[dict]:
    files = _drive_list(
        f"'{ml_folder_id}' in parents"
        " and name='ml_dashboard_representatives.json'"
        " and trashed=false"
    )
    return _download_json(files[0]["id"]) if files else []


@st.cache_data(ttl=300)
def _load_ml_results(folder_id: str, prefix: str) -> dict:
    fm  = _list_files(folder_id)
    out: dict = {}
    if f"{prefix}_metrics_val.json"         in fm: out["metrics"]           = _download_json(fm[f"{prefix}_metrics_val.json"])
    if f"{prefix}_train_summary.json"        in fm: out["train_summary"]     = _download_json(fm[f"{prefix}_train_summary.json"])
    if f"{prefix}_feature_importance.csv"    in fm: out["feature_importance"] = _download_csv(fm[f"{prefix}_feature_importance.csv"])
    if f"{prefix}_confusion_matrix_val.csv"  in fm: out["confusion_matrix"]   = _download_csv(fm[f"{prefix}_confusion_matrix_val.csv"])
    return out


@st.cache_data(ttl=300)
def _get_woe_iv_root_id(project_folder_id: str) -> str:
    data_id = _get_folder_id(project_folder_id, "data")
    if not data_id: return ""
    ml_id = _get_folder_id(data_id, "ml")
    if not ml_id:   return ""
    return _get_folder_id(ml_id, "woe_iv")


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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
    st.warning("ml_dashboard_representatives.json에 유효한 실험이 없습니다.")
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


# ══════════════════════════════════════════════════════════════════════════════
# 탭
# ══════════════════════════════════════════════════════════════════════════════

tab_ml, tab_woe = st.tabs(["📊 ML 결과", "🔍 WOE / IV"])


# ──────────────────────────────────────────────────────────────────────────────
# 탭 1: ML 결과
# ──────────────────────────────────────────────────────────────────────────────
with tab_ml:
    sel = st.selectbox("실험 선택", exp_labels, key="ml_sel",
                       index=_default_idx, label_visibility="collapsed")
    d   = exp_data[sel]
    rep = d["rep"]
    ml  = d["ml"]

    note = rep.get("note", "")
    st.caption(f"**Status**: {rep.get('status', '—')}")

    if not ml:
        st.info("학습된 모델이 없습니다.")
    else:
        metrics_raw   = ml.get("metrics", {})
        train_summary = ml.get("train_summary", {})
        feat_imp      = ml.get("feature_importance")
        conf_mat      = ml.get("confusion_matrix")
        m             = metrics_raw.get("metrics", metrics_raw)

        # ── 성능 지표 카드 ────────────────────────────────────────────────────
        st.markdown("#### Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        f1    = m.get("f1", 0)
        aucpr = m.get("average_precision") or train_summary.get("best_score")
        c1.metric("F1",        f"{f1:.4f}")
        c2.metric("AUCPR",     f"{aucpr:.4f}" if aucpr is not None else "—")
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
        _metric(c6, "Train",               f"{train_rows:,}",              f"pos {train_pos:.5f}")
        _metric(c7, "Val",                 f"{val_rows:,}",                f"pos {val_pos:.5f}")
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
                        col.metric(k, v)

        st.divider()

        # ── 학습 곡선 + Confusion Matrix ──────────────────────────────────────
        col_curve, col_cm = st.columns([3, 2])

        with col_curve:
            st.markdown("##### Learning Curve")
            diag     = train_summary.get("xgboost_diagnostics", {})
            eval_res = (diag.get("evals_result") or {}).get("validation_0", {})
            if eval_res.get("f1"):
                curve_key, curve_label, curve_best = "f1", "F1", f1
            else:
                curve_key, curve_label, curve_best = "aucpr", "AUCPR", aucpr or 0
            curve_vals = eval_res.get(curve_key, [])
            if curve_vals:
                fig_curve = px.line(
                    pd.DataFrame({"Iteration": range(1, len(curve_vals) + 1), curve_label: curve_vals}),
                    x="Iteration", y=curve_label,
                    title=f"Validation {curve_label}  (best={curve_best:.4f} @ iter {best_iter + 1})",
                )
                fig_curve.add_vline(x=best_iter + 1, line_dash="dash", line_color="#d62728",
                                    annotation_text="best", annotation_font_size=11)
                fig_curve.update_layout(height=310, margin=dict(t=40, b=20))
                st.plotly_chart(fig_curve, use_container_width=True)
            else:
                st.info("학습 곡선 데이터 없음")

        with col_cm:
            st.markdown("##### Confusion Matrix")
            if conf_mat is not None and not conf_mat.empty:
                row = conf_mat.iloc[0]
                tn, fp, fn, tp = int(row.get("tn",0)), int(row.get("fp",0)), int(row.get("fn",0)), int(row.get("tp",0))
                fig_cm = px.imshow(
                    [[tp, fn], [fp, tn]],
                    x=["Pred Fraud", "Pred Normal"],
                    y=["Actual Fraud", "Actual Normal"],
                    color_continuous_scale="Blues", text_auto=True, title="",
                )
                fig_cm.update_coloraxes(showscale=False)
                fig_cm.update_layout(height=270, margin=dict(t=10, b=10))
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
            _cat_fi = d.get("catalog")
            if _cat_fi is not None and not _cat_fi.empty and "피처명" in _cat_fi.columns:
                _desc_map = _cat_fi.set_index("피처명")["설명"].to_dict()
                fi_df["_desc"] = fi_df["feature"].map(lambda f: _desc_map.get(f) or "")
            else:
                fi_df["_desc"] = ""
            fig_fi = px.bar(
                fi_df, x="importance_gain", y="feature", orientation="h",
                color="importance_gain", color_continuous_scale="Blues",
                labels={"importance_gain": "Gain", "feature": "Feature"},
                custom_data=["importance_weight", "importance_cover", "rank_by_gain", "_desc"],
            )
            fig_fi.update_traces(hovertemplate=(
                "<b>%{y}</b><br>Gain: %{x:,.1f}<br>"
                "Weight: %{customdata[0]:,.0f}<br>Cover: %{customdata[1]:,.1f}<br>"
                "Rank: %{customdata[2]}<br>%{customdata[3]}<extra></extra>"
            ))
            fig_fi.update_coloraxes(showscale=False)
            fig_fi.update_layout(
                height=max(400, top_n_fi * 28),
                yaxis={"categoryorder": "total ascending" if fi_desc else "total descending"},
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance 파일 없음")


# ──────────────────────────────────────────────────────────────────────────────
# 탭 2: WOE / IV
# ──────────────────────────────────────────────────────────────────────────────
with tab_woe:
    st.caption(
        "**WOE(Weight of Evidence)**: 각 구간에서 fraud 비율과 정상 비율의 로그 비.  \n"
        "**IV(Information Value)**: WOE를 전체 구간에 걸쳐 집계한 변수 단위 예측력 요약."
    )

    sel_woe  = st.selectbox("실험 선택", exp_labels, key="woe_sel",
                            index=_default_idx, label_visibility="collapsed")
    d_woe    = exp_data[sel_woe]
    woe      = d_woe["woe"]
    rep_woe  = d_woe["rep"]
    stale_st = d_woe["stale_status"]
    cur_pfx  = d_woe["prefix"]

    _woe_note = rep_woe.get("note", "")
    if _woe_note:
        st.caption(_woe_note)

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
        has_overflow      = (top_df["iv"] > IV_CUT).any()

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
            height=max(420, top_n * 40),
            yaxis={"categoryorder": "total ascending" if woe_desc else "total descending"},
            xaxis={
                "range":    [0, IV_CUT + (0.35 if has_overflow else 0.05)],
                "tickvals": [0, 0.5, 1.0, 1.5],
                "ticktext": ["0", "0.5", "1.0", "1.5"],
                "title":    "IV",
            },
            legend_title_text="IV 강도",
        )

        for i, row in top_df[top_df["iv"] > IV_CUT].iterrows():
            fig.add_annotation(x=IV_CUT + 0.04, y=i, text=f"{row['iv']:.4f}",
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

        iv_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="iv_chart")

        # ── WOE 구간 차트 ──────────────────────────────────────────────────────
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
                    main_bins    = feat_bins[~feat_bins["missing_flag"]].sort_values("bin_id")
                    missing_bins = feat_bins[feat_bins["missing_flag"]]
                    feat_sorted  = pd.concat([main_bins, missing_bins], ignore_index=True)
                    feat_sorted["_color"] = feat_sorted["woe"].apply(lambda w: "fraud↑" if w >= 0 else "fraud↓")
                    fig_woe = px.bar(
                        feat_sorted, x="bin_label", y="woe",
                        color="_color",
                        color_discrete_map={"fraud↑": "#d62728", "fraud↓": "#2ca02c"},
                        custom_data=["count", "positive_count", "positive_rate", "iv_bin"],
                        labels={"bin_label": "구간", "woe": "WOE", "_color": ""},
                        title=f"{sel_feature} — WOE by Bin",
                    )
                    fig_woe.update_traces(hovertemplate=(
                        "<b>%{x}</b><br>WOE: %{y:.4f}<br>Count: %{customdata[0]:,}<br>"
                        "Positive: %{customdata[1]:,}<br>Positive Rate: %{customdata[2]:.5f}<br>"
                        "IV Bin: %{customdata[3]:.4f}<extra></extra>"
                    ))
                    fig_woe.add_hline(y=0, line_dash="dash", line_color="#333333", line_width=1)
                    fig_woe.update_layout(height=380, xaxis_tickangle=-40, legend_title_text="WOE 방향")
                    st.plotly_chart(fig_woe, use_container_width=True)
