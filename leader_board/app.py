"""
WOE/IV Leaderboard
Streamlit Community Cloud에 배포해서 팀원 누구나 링크로 접근 가능합니다.

필요한 Streamlit secrets:
    GOOGLE_API_KEY      Google Drive API key (공개 Drive 접근용)
    RESULTS_FOLDER_ID   실험 결과 폴더의 Drive ID

결과 폴더 구조:
    woe_iv_results/          ← RESULTS_FOLDER_ID 가 가리키는 곳
        ml_exp00/
            iv_summary.json
            meta.json
        gnn_exp01/
            ...
"""
from __future__ import annotations

import numpy as np
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WOE/IV Leaderboard", layout="wide", page_icon="📊")

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY           = st.secrets.get("GOOGLE_API_KEY", "")
RESULTS_FOLDER_ID = st.secrets.get("RESULTS_FOLDER_ID", "")

IV_COLORS = {
    "suspicious": "#d62728",
    "strong":     "#ff7f0e",
    "medium":     "#2ca02c",
    "weak":       "#aec7e8",
    "useless":    "#d3d3d3",
    "na":         "#eeeeee",
}


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
        st.error(f"Drive API 오류 {r.status_code}: {r.text}")
        st.stop()
    return r.json().get("files", [])


def _download_json(file_id: str) -> object:
    r = requests.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def list_experiments() -> list[dict]:
    return _drive_list(
        f"'{RESULTS_FOLDER_ID}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )


@st.cache_data(ttl=300)
def load_experiment(folder_id: str) -> tuple[dict | None, pd.DataFrame | None]:
    files = _drive_list(
        f"'{folder_id}' in parents"
        " and (name = 'iv_summary.json' or name = 'meta.json')"
        " and trashed=false"
    )
    file_map = {f["name"]: f["id"] for f in files}
    if "iv_summary.json" not in file_map or "meta.json" not in file_map:
        return None, None
    meta  = _download_json(file_map["meta.json"])
    iv_df = pd.DataFrame(_download_json(file_map["iv_summary.json"]))
    return meta, iv_df


# ── UI ─────────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([8, 1])
col_title.title("돈무브 프로젝트 리더보드")
if col_btn.button("🔄 새로고침", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if not API_KEY or not RESULTS_FOLDER_ID:
    st.error(
        "**Streamlit secrets 설정 필요** — `.streamlit/secrets.toml` 또는 "
        "Streamlit Cloud 대시보드에 아래 항목을 추가하세요.\n\n"
        "```toml\n"
        'GOOGLE_API_KEY    = "AIzaSy..."\n'
        'RESULTS_FOLDER_ID = "1abc..."\n'
        "```\n\n"
        "Google API 키 발급: console.cloud.google.com → Drive API 활성화 → 사용자 인증 정보 → API 키"
    )
    st.stop()

# ── 실험 목록 로드 ──────────────────────────────────────────────────────────
with st.spinner("실험 목록 로드 중..."):
    experiments = list_experiments()

if not experiments:
    st.info("결과 폴더가 비어 있습니다. `compute_woe_iv.ipynb`를 실행해 결과를 저장하세요.")
    st.stop()

# ── 모든 실험 데이터 로드 ───────────────────────────────────────────────────
all_meta: dict[str, dict]         = {}
all_iv:   dict[str, pd.DataFrame] = {}

bar = st.progress(0, text="실험 데이터 로드 중...")
for i, exp in enumerate(experiments):
    meta, iv_df = load_experiment(exp["id"])
    if meta is not None and iv_df is not None:
        all_meta[exp["name"]] = meta
        all_iv[exp["name"]]   = iv_df
    bar.progress((i + 1) / len(experiments), text=f"로드: {exp['name']}")
bar.empty()

if not all_iv:
    st.warning("유효한 실험 결과가 없습니다.")
    st.stop()

exp_names = list(all_iv.keys())

st.divider()
st.markdown("#### WOE/IV")

# ── 실험별 IV 바 차트 ───────────────────────────────────────────────────────
left, right = st.columns([1, 3])

with left:
    sel_exp     = st.selectbox("실험 선택", exp_names, label_visibility="collapsed")
    meta        = all_meta[sel_exp]
    iv_df       = all_iv[sel_exp]
    top_n       = st.slider("Top N", 10, min(50, len(iv_df)), 20)

    n_rows = meta.get("n_rows") or (meta.get("run_shape") or [0])[0]
    st.markdown(f"""
| 항목 | 값 |
|------|-----|
| 계산일 | {meta.get('computed_at','')[:19]} |
| 데이터 | {'전체' if meta.get('full_run') else '샘플'} |
| 행 수 | {n_rows:,} |
| positive | {meta.get('positive_rate', 0):.5f} |
| 소요 | {meta.get('elapsed_seconds','?')}초 |
""")

# ── Broken-axis chart constants ────────────────────────────────────────────
LOW_MAX    = 1.5   # left region shows 0 … LOW_MAX
GAP        = 0.2   # visual gap width in display coords
HIGH_START = LOW_MAX + GAP          # 1.7 — right region starts here
WAVE_AMP   = 0.015
N_WAVES    = 8


def _wave_path(x_center: float, y_bot: float, y_top: float) -> str:
    ys = np.linspace(y_bot, y_top, N_WAVES * 2 + 1)
    pts = [f"M {x_center - WAVE_AMP} {ys[0]}"]
    for k, y in enumerate(ys[1:]):
        x = x_center + WAVE_AMP if k % 2 == 0 else x_center - WAVE_AMP
        pts.append(f"L {x} {y}")
    return " ".join(pts)


with right:
    top_df = (
        iv_df.dropna(subset=["iv"])
        .head(top_n)
        .sort_values("iv")
        .reset_index(drop=True)
    )
    n_feats   = len(top_df)
    has_break = (top_df["iv"] > LOW_MAX).any()

    # overflow amount in display coords (1 real-IV unit = 1 display unit after HIGH_START)
    overflow_max = (top_df["iv"] - LOW_MAX).clip(lower=0).max() if has_break else 0
    x_max = HIGH_START + overflow_max + 0.1 if has_break else LOW_MAX + 0.1

    # ── colour list preserving category order ──────────────────────────────
    strength_order = ["suspicious", "strong", "medium", "weak", "useless", "na"]
    legend_seen: set[str] = set()
    traces = []

    for strength in strength_order:
        sub = top_df[top_df["iv_strength"] == strength]
        if sub.empty:
            continue
        color = IV_COLORS.get(strength, "#cccccc")

        # left bar: clipped at LOW_MAX
        traces.append(go.Bar(
            x=sub["iv"].clip(upper=LOW_MAX),
            y=sub["feature_name"],
            orientation="h",
            marker_color=color,
            name=strength,
            legendgroup=strength,
            showlegend=True,
            customdata=sub[["iv"]].values,
            hovertemplate="<b>%{y}</b><br>IV: %{customdata[0]:.4f}<extra></extra>",
        ))

        # right bar: overflow portion starting at HIGH_START
        overflow_sub = sub[sub["iv"] > LOW_MAX]
        if not overflow_sub.empty:
            traces.append(go.Bar(
                x=overflow_sub["iv"] - LOW_MAX,          # width = actual overflow
                y=overflow_sub["feature_name"],
                base=[HIGH_START] * len(overflow_sub),   # start at HIGH_START
                orientation="h",
                marker_color=color,
                name=strength,
                legendgroup=strength,
                showlegend=False,
                customdata=overflow_sub[["iv"]].values,
                hovertemplate="<b>%{y}</b><br>IV: %{customdata[0]:.4f}<extra></extra>",
            ))

    fig = go.Figure(data=traces)

    # ── x-axis ticks ───────────────────────────────────────────────────────
    left_ticks  = [v for v in [0.0, 0.5, 1.0, 1.5] if v <= LOW_MAX]
    right_step  = 0.5
    right_reals = np.arange(LOW_MAX + right_step, top_df["iv"].max() + right_step, right_step) if has_break else []
    tickvals = left_ticks + [HIGH_START + (r - LOW_MAX) for r in right_reals]
    ticktext = [str(v) for v in left_ticks] + [f"{r:.1f}" for r in right_reals]

    # threshold vlines (left region only)
    iv_thresholds = [
        (0.02, "weak",       "#aaaaaa"),
        (0.10, "medium",     "#888888"),
        (0.30, "strong",     "#555555"),
        (0.50, "suspicious", "#222222"),
    ]

    fig.update_layout(
        title=f"{sel_exp} — Top {top_n} Features by IV",
        barmode="overlay",
        height=max(420, n_feats * 24),
        yaxis={"categoryorder": "total ascending"},
        xaxis={"range": [0, x_max], "tickvals": tickvals, "ticktext": ticktext, "title": "IV"},
        legend_title_text="IV 강도",
        shapes=[],
    )

    for val, _, color in iv_thresholds:
        if val <= LOW_MAX:
            fig.add_vline(x=val, line_dash="dot", line_color=color)

    if has_break:
        # gap background rectangle
        fig.add_shape(
            type="rect",
            x0=LOW_MAX, x1=HIGH_START,
            y0=-0.5, y1=n_feats - 0.5,
            fillcolor="white", line_width=0, layer="above",
        )
        # two wave lines spanning full chart height
        for x_c in [LOW_MAX + GAP * 0.3, LOW_MAX + GAP * 0.7]:
            fig.add_shape(
                type="path",
                path=_wave_path(x_c, -0.5, n_feats - 0.5),
                line=dict(color="#555555", width=1.5),
                layer="above",
            )
        # actual IV value labels for clipped bars
        for _, row in top_df[top_df["iv"] > LOW_MAX].iterrows():
            fig.add_annotation(
                x=HIGH_START + (row["iv"] - LOW_MAX) + 0.02,
                y=row["feature_name"],
                text=f"{row['iv']:.4f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10),
            )

    st.plotly_chart(fig, use_container_width=True)
