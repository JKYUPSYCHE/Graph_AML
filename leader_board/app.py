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

# ── IV bar chart constants ─────────────────────────────────────────────────
IV_CUT      = 1.5
_WAVE1_X    = 1.57
_WAVE2_X    = 1.63
_SHORT_ST   = 1.68
_SHORT_W    = 0.20
_W_AMP      = 0.012
_N_WAVES    = 7
_BAR_HH     = 0.35


def _wave(x_c: float, y_mid: float) -> str:
    ys = np.linspace(y_mid - _BAR_HH, y_mid + _BAR_HH, _N_WAVES * 2 + 1)
    pts = [f"M {x_c - _W_AMP} {ys[0]}"]
    for k, y in enumerate(ys[1:]):
        x = x_c + _W_AMP if k % 2 == 0 else x_c - _W_AMP
        pts.append(f"L {x} {y}")
    return " ".join(pts)


with right:
    top_df = (
        iv_df.dropna(subset=["iv"])
        .head(top_n)
        .sort_values("iv")
        .reset_index(drop=True)
    )
    has_overflow = (top_df["iv"] > IV_CUT).any()
    x_max = _SHORT_ST + _SHORT_W + 0.35 if has_overflow else IV_CUT + 0.1

    strength_order = ["suspicious", "strong", "medium", "weak", "useless", "na"]
    traces = []

    for strength in strength_order:
        sub = top_df[top_df["iv_strength"] == strength]
        if sub.empty:
            continue
        color = IV_COLORS.get(strength, "#cccccc")

        # 메인 bar (0 → IV_CUT 클립)
        traces.append(go.Bar(
            x=sub["iv"].clip(upper=IV_CUT),
            y=sub["feature_name"],
            orientation="h",
            marker_color=color,
            name=strength,
            legendgroup=strength,
            showlegend=True,
            customdata=sub[["iv"]].values,
            hovertemplate="<b>%{y}</b><br>IV: %{customdata[0]:.4f}<extra></extra>",
        ))

        # 짧은 bar (물결 이후 고정 너비)
        overflow_sub = sub[sub["iv"] > IV_CUT]
        if not overflow_sub.empty:
            traces.append(go.Bar(
                x=[_SHORT_W] * len(overflow_sub),
                y=overflow_sub["feature_name"],
                base=[_SHORT_ST] * len(overflow_sub),
                orientation="h",
                marker_color=color,
                name=strength,
                legendgroup=strength,
                showlegend=False,
                customdata=overflow_sub[["iv"]].values,
                hovertemplate="<b>%{y}</b><br>IV: %{customdata[0]:.4f}<extra></extra>",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{sel_exp} — Top {top_n} Features by IV",
        barmode="overlay",
        height=max(420, len(top_df) * 24),
        yaxis={"categoryorder": "total ascending"},
        xaxis={
            "range": [0, x_max],
            "tickvals": [0, 0.5, 1.0, 1.5],
            "ticktext": ["0", "0.5", "1.0", "1.5"],
            "title": "IV",
        },
        legend_title_text="IV 강도",
    )

    for val, label, color in [
        (0.02, "weak",       "#aaaaaa"),
        (0.10, "medium",     "#888888"),
        (0.30, "strong",     "#555555"),
        (0.50, "suspicious", "#222222"),
    ]:
        fig.add_vline(x=val, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_font_size=10)

    for i, row in top_df[top_df["iv"] > IV_CUT].iterrows():
        # 흰 직사각형으로 갭 가리기
        fig.add_shape(
            type="rect",
            x0=IV_CUT, x1=_SHORT_ST,
            y0=i - _BAR_HH, y1=i + _BAR_HH,
            fillcolor="white", line_width=0, layer="above",
        )
        # 두 물결선
        for x_c in [_WAVE1_X, _WAVE2_X]:
            fig.add_shape(
                type="path",
                path=_wave(x_c, i),
                line=dict(color="#555555", width=1.5),
                layer="above",
            )
        # 실제값 텍스트
        fig.add_annotation(
            x=_SHORT_ST + _SHORT_W + 0.02, y=i,
            text=f"{row['iv']:.4f}",
            showarrow=False, xanchor="left",
            font=dict(size=10),
        )

    st.plotly_chart(fig, use_container_width=True)
