"""
WOE/IV Leaderboard

필요한 Streamlit secrets:
    GOOGLE_API_KEY       Google Drive API key (공개 Drive 접근용)
    PROJECT_FOLDER_ID    프로젝트 루트 폴더의 Drive ID

결과 폴더 구조 (루트 하위 어느 깊이든 자동 탐색):
    프로젝트 루트/          ← PROJECT_FOLDER_ID 가 가리키는 곳
        leader_board/
            ml_exp00/
                iv_summary.json
                meta.json
        gnn/
            gnn_exp01/
                ...
"""
from __future__ import annotations

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="돈무브 프로젝트 리더보드", layout="wide", page_icon="📊")

# ── Config ─────────────────────────────────────────────────────────────────
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


# ── Drive helpers ──────────────────────────────────────────────────────────

def _drive_list(q: str) -> list[dict]:
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": q,
            "fields": "files(id,name,modifiedTime,parents)",
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


def _subfolders(parent_id: str) -> list[dict]:
    return _drive_list(
        f"'{parent_id}' in parents"
        " and mimeType='application/vnd.google-apps.folder'"
        " and trashed=false"
    )


def _has_iv_summary(folder_id: str) -> bool:
    return bool(_drive_list(
        f"'{folder_id}' in parents"
        " and name = 'iv_summary.json'"
        " and trashed=false"
    ))


@st.cache_data(ttl=300)
def list_experiments() -> list[dict]:
    # 루트 → depth1 → depth2 순으로 iv_summary.json 보유 폴더 수집
    result: list[dict] = []
    seen: set[str] = set()

    for d1 in _subfolders(PROJECT_FOLDER_ID):
        if _has_iv_summary(d1["id"]) and d1["id"] not in seen:
            seen.add(d1["id"])
            result.append({"id": d1["id"], "name": d1["name"]})
        for d2 in _subfolders(d1["id"]):
            if _has_iv_summary(d2["id"]) and d2["id"] not in seen:
                seen.add(d2["id"])
                result.append({"id": d2["id"], "name": d2["name"]})

    return sorted(result, key=lambda x: x["name"])


@st.cache_data(ttl=300)
def load_experiment(folder_id: str) -> tuple[dict | None, pd.DataFrame | None, pd.DataFrame | None]:
    files = _drive_list(
        f"'{folder_id}' in parents"
        " and (name = 'iv_summary.json' or name = 'meta.json' or name = 'bin_table.json')"
        " and trashed=false"
    )
    file_map = {f["name"]: f["id"] for f in files}
    if "iv_summary.json" not in file_map or "meta.json" not in file_map:
        return None, None, None
    meta  = _download_json(file_map["meta.json"])
    iv_df = pd.DataFrame(_download_json(file_map["iv_summary.json"]))
    bin_df = (
        pd.DataFrame(_download_json(file_map["bin_table.json"]))
        if "bin_table.json" in file_map else None
    )
    return meta, iv_df, bin_df


# ── UI ─────────────────────────────────────────────────────────────────────

col_title, col_btn = st.columns([8, 1])
col_title.title("돈무브 프로젝트 리더보드")
if col_btn.button("🔄 새로고침", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if not API_KEY or not PROJECT_FOLDER_ID:
    st.error(
        "**Streamlit secrets 설정 필요** — `.streamlit/secrets.toml` 또는 "
        "Streamlit Cloud 대시보드에 아래 항목을 추가하세요.\n\n"
        "```toml\n"
        'GOOGLE_API_KEY    = "AIzaSy..."\n'
        'PROJECT_FOLDER_ID = "1abc..."\n'
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
all_bin:  dict[str, pd.DataFrame] = {}

bar = st.progress(0, text="실험 데이터 로드 중...")
for i, exp in enumerate(experiments):
    meta, iv_df, bin_df = load_experiment(exp["id"])
    if meta is not None and iv_df is not None:
        all_meta[exp["name"]] = meta
        all_iv[exp["name"]]   = iv_df
        if bin_df is not None:
            all_bin[exp["name"]] = bin_df
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
ctrl_left, _ = st.columns([1, 3])

with ctrl_left:
    sel_exp = st.selectbox("실험 선택", exp_names, label_visibility="collapsed")

meta       = all_meta[sel_exp]
iv_df      = all_iv[sel_exp]
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

top_n = st.slider("Top N", 10, min(50, len(iv_df)), 20)

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
iv_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="iv_bar_chart")

# ── 클릭된 피처의 WOE 차트 ─────────────────────────────────────────────────
sel_feature: str | None = None
pts = (iv_event.selection or {}).get("points", []) if iv_event else []
if pts:
    sel_feature = pts[0].get("label") or pts[0].get("y")

bin_df_exp = all_bin.get(sel_exp)

if sel_feature:
    st.markdown(f"#### WOE — `{sel_feature}`")
    if bin_df_exp is None:
        st.info("bin_table.json이 없습니다. compute_woe_iv.ipynb를 다시 실행해 저장하세요.")
    else:
        feat_bins = bin_df_exp[bin_df_exp["feature_name"] == sel_feature].copy()
        if feat_bins.empty:
            st.info(f"'{sel_feature}'의 WOE 구간 데이터가 없습니다.")
        else:
            main_bins    = feat_bins[~feat_bins["missing_flag"]].sort_values("bin_id")
            missing_bins = feat_bins[feat_bins["missing_flag"]]
            feat_sorted  = pd.concat([main_bins, missing_bins], ignore_index=True)

            feat_sorted["_color"] = feat_sorted["woe"].apply(
                lambda w: "fraud↑" if w >= 0 else "fraud↓"
            )

            fig_woe = px.bar(
                feat_sorted,
                x="bin_label", y="woe",
                color="_color",
                color_discrete_map={"fraud↑": "#d62728", "fraud↓": "#2ca02c"},
                custom_data=["count", "positive_count", "positive_rate", "iv_bin"],
                labels={"bin_label": "구간", "woe": "WOE", "_color": ""},
                title=f"{sel_feature} — WOE by Bin",
            )
            fig_woe.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "WOE: %{y:.4f}<br>"
                    "Count: %{customdata[0]:,}<br>"
                    "Positive: %{customdata[1]:,}<br>"
                    "Positive Rate: %{customdata[2]:.5f}<br>"
                    "IV Bin: %{customdata[3]:.4f}<extra></extra>"
                )
            )
            fig_woe.add_hline(y=0, line_dash="dash", line_color="#333333", line_width=1)
            fig_woe.update_layout(
                height=380,
                xaxis_tickangle=-40,
                legend_title_text="WOE 방향",
            )
            st.plotly_chart(fig_woe, use_container_width=True)
