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

import requests
import pandas as pd
import plotly.express as px
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
st.caption(
    "**WOE(Weight of Evidence)**: 각 구간에서 fraud 비율과 정상 비율의 로그 비 — 변수가 타겟을 어느 방향으로 얼마나 분리하는지 나타냄. &nbsp;|&nbsp; "
    "**IV(Information Value)**: WOE를 전체 구간에 걸쳐 집계한 변수 단위 예측력 요약 지표 — 값이 클수록 타겟 분류에 유용한 변수."
)

# ── 실험별 IV 바 차트 ───────────────────────────────────────────────────────
ctrl_left, ctrl_right = st.columns([1, 3])

with ctrl_left:
    sel_exp = st.selectbox("실험 선택", exp_names, label_visibility="collapsed")

with ctrl_right:
    meta   = all_meta[sel_exp]
    iv_df  = all_iv[sel_exp]
    top_n  = st.slider("Top N", 10, min(50, len(iv_df)), 20)

n_rows     = meta.get("n_rows") or (meta.get("run_shape") or [0])[0]
n_features = len(iv_df)
st.markdown(f"""
| 항목 | 값 |
|------|-----|
| 계산일 | {meta.get('computed_at','')[:19]} |
| feature 수 | {n_features:,} |
| 데이터 | {'전체' if meta.get('full_run') else '샘플'} |
| 행 수 | {n_rows:,} |
| positive | {meta.get('positive_rate', 0):.5f} |
""")

IV_CUT = 1.5

top_df = (
    iv_df.dropna(subset=["iv"])
    .head(top_n)
    .sort_values("iv")
    .reset_index(drop=True)
)
top_df["_iv_bar"] = top_df["iv"].clip(upper=IV_CUT)
has_overflow = (top_df["iv"] > IV_CUT).any()

fig = px.bar(
    top_df,
    x="_iv_bar", y="feature_name",
    orientation="h",
    color="iv_strength",
    color_discrete_map=IV_COLORS,
    custom_data=["iv"],
    labels={"_iv_bar": "IV", "feature_name": "Feature", "iv_strength": "강도"},
    title=f"{sel_exp} — Top {top_n} Features by IV",
)
fig.update_traces(
    hovertemplate="<b>%{y}</b><br>IV: %{customdata[0]:.4f}<extra></extra>"
)
fig.update_layout(
    height=max(420, top_n * 40),
    yaxis={"categoryorder": "total ascending"},
    xaxis={
        "range": [0, IV_CUT + (0.35 if has_overflow else 0.05)],
        "tickvals": [0, 0.5, 1.0, 1.5],
        "ticktext": ["0", "0.5", "1.0", "1.5"],
        "title": "IV",
    },
    legend_title_text="IV 강도",
)

for i, row in top_df[top_df["iv"] > IV_CUT].iterrows():
    fig.add_annotation(
        x=IV_CUT + 0.04, y=i,
        text=f"{row['iv']:.4f}",
        showarrow=False,
        xanchor="left",
        font=dict(size=10),
    )

for val, label, color in [
    (0.02, "weak", "#aaaaaa"), (0.10, "medium", "#888888"),
    (0.30, "strong", "#555555"), (0.50, "suspicious", "#222222"),
]:
    fig.add_vline(x=val, line_dash="dot", line_color=color,
                  annotation_text=label, annotation_font_size=10)
st.plotly_chart(fig, use_container_width=True)
