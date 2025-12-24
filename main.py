# main.py
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =========================
# Page / Global Style
# =========================
st.set_page_config(page_title="🌱 극지식물 최적 EC/온도 연구", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


def apply_plotly_korean_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"),
        legend=dict(title=None),
        margin=dict(l=30, r=30, t=70, b=40),
    )
    return fig


# =========================
# Experiment Constants (display & fallback)
# =========================
SCHOOL_ORDER = ["송도고", "하늘고", "아라고", "동산고"]

# 보고서(텍스트)에서 제시된 대표값(학교별 평균 추정치)
# - 온도 평균: 송도 23.54, 동산 22.37, 하늘 18.18, 아라고 19.26
# - EC 평균: 송도 0.72, 동산 1.11, 하늘 4.00, 아라고 7.82
# - 생중량 평균: 하늘 3.94, 송도 3.73, 동산 3.53, 아라고 1.89
REPORT_FALLBACK = pd.DataFrame(
    {
        "학교": ["송도고", "동산고", "하늘고", "아라고"],
        "평균온도": [23.54, 22.37, 18.18, 19.26],
        "평균EC": [0.72, 1.11, 4.00, 7.82],
        "평균생중량": [3.73, 3.53, 3.94, 1.89],
    }
)

# 사이드바에서 "학교 선택" 용(전체 포함)
SCHOOL_SELECT = ["전체"] + SCHOOL_ORDER


# =========================
# File / Unicode Robust
# =========================
def get_data_dir() -> Path:
    """
    Streamlit Cloud/로컬에서 data 폴더를 확실히 찾는다.
    - main.py 위치 기준 ./data 우선
    - 그 다음 현재 작업폴더 ./data
    """
    here = Path(__file__).resolve().parent
    cand1 = here / "data"
    if cand1.exists():
        return cand1

    cand2 = Path.cwd() / "data"
    if cand2.exists():
        return cand2

    return cand1


def _norm_all(s: str) -> set[str]:
    return {
        unicodedata.normalize("NFC", s),
        unicodedata.normalize("NFD", s),
    }


def canonical_filename(name: str) -> str:
    n = unicodedata.normalize("NFC", str(name)).strip()
    low = n.lower()
    if low.endswith(".csv.csv"):
        n = n[:-4]
    if low.endswith(".xlsx.xlsx"):
        n = n[:-5]
    return n


def filename_match(candidate: str, desired: str) -> bool:
    c_nfc = canonical_filename(candidate)
    d_nfc = canonical_filename(desired)
    c_nfd = unicodedata.normalize("NFD", c_nfc)
    d_nfd = unicodedata.normalize("NFD", d_nfc)

    if c_nfc == d_nfc or c_nfd == d_nfd:
        return True
    if c_nfc.endswith(d_nfc) or c_nfd.endswith(d_nfd):
        return True
    return False


def find_file_by_name(directory: Path, desired_name: str) -> Path | None:
    if not directory.exists():
        return None

    desired_norms = _norm_all(canonical_filename(desired_name))
    for p in directory.iterdir():
        if not p.is_file():
            continue

        cand_name = canonical_filename(p.name)
        cand_norms = _norm_all(cand_name)

        if desired_norms.intersection(cand_norms):
            return p
        if filename_match(p.name, desired_name):
            return p
    return None


# =========================
# CSV Safety & Column Standardization
# =========================
def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def normalize_colname(c: str) -> str:
    c = unicodedata.normalize("NFC", str(c)).strip().lower()
    c = c.replace("\ufeff", "")
    c = re.sub(r"\s+", "", c)
    c = c.replace("-", "").replace(".", "")
    return c


def standardize_env_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    env: time, temperature, humidity, ph, ec 로 표준화
    """
    df2 = df.copy()
    colmap = {c: normalize_colname(c) for c in df2.columns}
    inv = {}
    for orig, n in colmap.items():
        inv.setdefault(n, []).append(orig)

    target = {
        "time": {"time", "datetime", "timestamp", "측정시간", "시간", "date", "날짜"},
        "temperature": {"temperature", "temp", "t", "온도"},
        "humidity": {"humidity", "hum", "h", "습도"},
        "ph": {"ph", "산도"},
        "ec": {"ec", "전기전도도"},
    }

    rename = {}
    for std, cands in target.items():
        found = None
        for cand in cands:
            if cand in inv:
                found = inv[cand][0]
                break
        if found is not None:
            rename[found] = std

    return df2.rename(columns=rename)


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    return out.sort_values(time_col)


# =========================
# Growth Helpers
# =========================
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    norm_map = {unicodedata.normalize("NFC", str(c)): c for c in cols}
    for cand in candidates:
        cand_nfc = unicodedata.normalize("NFC", cand)
        if cand_nfc in norm_map:
            return norm_map[cand_nfc]
    return None


# =========================
# Data Loading (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_environment_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    env = {}
    desired_files = {
        "송도고": "송도고_환경데이터.csv",
        "하늘고": "하늘고_환경데이터.csv",
        "아라고": "아라고_환경데이터.csv",
        "동산고": "동산고_환경데이터.csv",
    }

    for school, desired_name in desired_files.items():
        p = find_file_by_name(data_dir, desired_name)
        if p is None:
            continue

        df = read_csv_safely(p)
        df = standardize_env_columns(df)
        df["학교"] = school
        env[school] = df

    return env


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir: Path) -> tuple[pd.DataFrame, list[str], Path | None]:
    desired_name = "4개교_생육결과데이터.xlsx"
    xlsx_path = find_file_by_name(data_dir, desired_name)
    if xlsx_path is None:
        xlsx_path = find_file_by_name(data_dir, "4개교_생육결과데이터.xlsx.xlsx")

    if xlsx_path is None:
        return pd.DataFrame(), [], None

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = list(xls.sheet_names)  # 시트명 하드코딩 금지

    frames = []
    for sh in sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
        df["학교"] = sh
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, sheet_names, xlsx_path


# =========================
# Metrics Builder (real data -> fallback)
# =========================
def build_school_metrics(
    env_by_school: dict[str, pd.DataFrame],
    growth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    학교별 평균온도/평균EC/평균생중량을 만든다.
    - 가능하면 실제 CSV/XLSX에서 계산
    - 부족하면 보고서 대표값으로 채움
    """
    # 1) 환경 평균 (실데이터)
    env_rows = []
    for s in SCHOOL_ORDER:
        df = env_by_school.get(s)
        if df is None or df.empty:
            continue
        if "temperature" not in df.columns or "ec" not in df.columns:
            continue
        t_mean = pd.to_numeric(df["temperature"], errors="coerce").mean()
        ec_mean = pd.to_numeric(df["ec"], errors="coerce").mean()
        env_rows.append({"학교": s, "평균온도": t_mean, "평균EC": ec_mean})
    env_mean = pd.DataFrame(env_rows)

    # 2) 생중량 평균 (실데이터)
    g_mean = pd.DataFrame()
    if growth_df is not None and not growth_df.empty and "학교" in growth_df.columns:
        col_weight = pick_col(growth_df, ["생중량(g)", "생중량"])
        if col_weight is not None:
            tmp = growth_df.copy()
            tmp[col_weight] = pd.to_numeric(tmp[col_weight], errors="coerce")
            g_mean = (
                tmp.groupby("학교", dropna=False)[col_weight]
                .mean()
                .reset_index()
                .rename(columns={col_weight: "평균생중량"})
            )

    # 3) 병합 후 누락은 fallback으로 채우기
    m = pd.DataFrame({"학교": SCHOOL_ORDER})
    if not env_mean.empty:
        m = m.merge(env_mean, on="학교", how="left")
    else:
        m["평균온도"] = pd.NA
        m["평균EC"] = pd.NA

    if not g_mean.empty:
        m = m.merge(g_mean, on="학교", how="left")
    else:
        m["평균생중량"] = pd.NA

    # fallback join
    fb = REPORT_FALLBACK.copy()
    m = m.merge(fb, on="학교", how="left", suffixes=("", "_fb"))

    for col in ["평균온도", "평균EC", "평균생중량"]:
        m[col] = m[col].astype("float64")
        fb_col = f"{col}_fb"
        m[col] = m[col].fillna(m[fb_col])

    m = m[["학교", "평균온도", "평균EC", "평균생중량"]]
    m["학교"] = pd.Categorical(m["학교"], categories=SCHOOL_ORDER, ordered=True)
    m = m.sort_values("학교").reset_index(drop=True)
    return m


# =========================
# Sidebar
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

data_dir = get_data_dir()

with st.sidebar:
    st.header("⚙️ 설정")
    school_option = st.selectbox("학교 선택", SCHOOL_SELECT, index=0)

    with st.expander("🧪 디버그: data 폴더/파일 확인"):
        st.write("data_dir =", str(data_dir))
        if data_dir.exists():
            st.write([p.name for p in data_dir.iterdir() if p.is_file()])
        else:
            st.error("data 폴더를 찾지 못했습니다.")


# =========================
# Load Data
# =========================
with st.spinner("데이터를 불러오는 중..."):
    env_by_school = load_environment_data(data_dir)
    growth_df, sheet_names, growth_path = load_growth_data(data_dir)

if growth_df is None or growth_df.empty:
    st.error("생육 결과 XLSX를 불러오지 못했습니다. data/에 '4개교_생육결과데이터.xlsx'가 있는지 확인하세요.")
    st.stop()

missing_env = [s for s in SCHOOL_ORDER if s not in env_by_school]
if missing_env:
    st.warning(f"환경 데이터가 없는 학교: {', '.join(missing_env)} (data/ 파일명 또는 인코딩/컬럼 확인)")

metrics = build_school_metrics(env_by_school, growth_df)

# school filter
metrics_show = metrics.copy()
if school_option != "전체":
    metrics_show = metrics_show[metrics_show["학교"] == school_option].copy()

# =========================
# Tabs (요구사항 3개)
# =========================
tab1, tab2, tab3 = st.tabs(
    ["📖 실험 개요", "📊 학교별 온도·EC 막대그래프", "🔍 생중량·EC·온도 상관관계(융합)"]
)

# -------------------------
# Tab 1: 실험개요 (보고서 기반)
# -------------------------
with tab1:
    st.subheader("1) 실험 개요(보고서 요약 기반)")
    st.write(
        """
**대상 식물:** 극지 모델식물 ‘나도수영’  
**목표:** 4개 고등학교에서 수집한 환경(온도·EC 등)과 생육(생중량 등) 데이터를 비교하여,
극지식물이 가장 잘 자라는 **최적 환경 범위**를 규명한다.

**핵심 결론(보고서):**
- 극지식물은 대체로 **18~22℃** 범위에서 무난하게 자랐으며,
- 특히 **EC 3~4 mS/cm** 구간에서 생중량이 최대이고,
- 이번 데이터에서는 **온도보다 EC가 생육에 더 큰 영향**을 보였다.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("참여 학교", f"{len(SCHOOL_ORDER)}개")
    c2.metric("최적 온도(보고서)", "18~22℃")
    c3.metric("최적 EC(보고서)", "3~4 mS/cm")
    # 데이터 기반으로도 최댓값(평균생중량 최대) 확인
    best_row = metrics.sort_values("평균생중량", ascending=False).head(1)
    if not best_row.empty:
        best_school = str(best_row.iloc[0]["학교"])
        best_w = float(best_row.iloc[0]["평균생중량"])
        c4.metric("데이터 기준 생중량 1위", f"{best_w:.2f} g", best_school)

    st.divider()
    st.subheader("학교별 대표값(대시보드 계산용)")
    st.dataframe(metrics, use_container_width=True)

# -------------------------
# Tab 2: 학교별 온도 & EC 막대그래프 (요구사항 2)
# -------------------------
with tab2:
    st.subheader("2) 학교별 평균 온도와 평균 EC(막대그래프)")
    if metrics_show.empty:
        st.error("표시할 데이터가 없습니다.")
    else:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("평균 온도(℃)", "평균 EC(mS/cm)"),
        )
        fig.add_trace(go.Bar(x=metrics_show["학교"], y=metrics_show["평균온도"], name="평균 온도"), row=1, col=1)
        fig.add_trace(go.Bar(x=metrics_show["학교"], y=metrics_show["평균EC"], name="평균 EC"), row=1, col=2)

        fig.update_layout(height=520, title="학교별 환경 조건 비교")
        fig = apply_plotly_korean_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 3: 생중량·EC·온도 상관관계(산점도 + 꺾은선 융합) (요구사항 3)
# -------------------------
with tab3:
    st.subheader("3) 생중량·EC·온도 상관관계(산점도 + 꺾은선 융합 표현)")

    if metrics_show.empty:
        st.error("표시할 데이터가 없습니다.")
        st.stop()

    # EC 기준 정렬(꺾은선 연결을 위해)
    mline = metrics_show.copy()
    mline = mline.sort_values("평균EC").reset_index(drop=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### ✅ 산점도(EC ↔ 생중량) + 온도 반영(마커 크기)")
        fig_sc = px.scatter(
            mline,
            x="평균EC",
            y="평균생중량",
            color="학교",
            size="평균온도",  # 온도까지 반영
            hover_data={"평균온도": ":.2f", "평균EC": ":.2f", "평균생중량": ":.2f"},
            labels={"평균EC": "평균 EC(mS/cm)", "평균생중량": "평균 생중량(g)", "평균온도": "평균 온도(℃)"},
            title="EC-생중량 관계(온도까지 동시에 반영)",
        )
        # EC가 3~4 근처를 “권장 구간”으로 시각적 가이드(보고서 결론 반영)
        fig_sc.add_vrect(x0=3, x1=4, opacity=0.15, annotation_text="권장 EC(3~4)", annotation_position="top left")
        fig_sc = apply_plotly_korean_font(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    with right:
        st.markdown("### ✅ 융합 꺾은선(이중축): x=EC / y1=생중량 / y2=온도")
        fig_mix = make_subplots(specs=[[{"secondary_y": True}]])

        # 생중량(좌축)
        fig_mix.add_trace(
            go.Scatter(
                x=mline["평균EC"],
                y=mline["평균생중량"],
                mode="lines+markers",
                name="평균 생중량(g)",
            ),
            secondary_y=False,
        )

        # 온도(우축)
        fig_mix.add_trace(
            go.Scatter(
                x=mline["평균EC"],
                y=mline["평균온도"],
                mode="lines+markers",
                name="평균 온도(℃)",
            ),
            secondary_y=True,
        )

        # 권장 EC 구간 강조(3~4)
        fig_mix.add_vrect(x0=3, x1=4, opacity=0.15, annotation_text="권장 EC(3~4)", annotation_position="top left")

        fig_mix.update_xaxes(title_text="평균 EC(mS/cm)")
        fig_mix.update_yaxes(title_text="평균 생중량(g)", secondary_y=False)
        fig_mix.update_yaxes(title_text="평균 온도(℃)", secondary_y=True)
        fig_mix.update_layout(height=520, title="EC를 기준으로 생중량·온도를 동시에 해석(융합)")
        fig_mix = apply_plotly_korean_font(fig_mix)

        st.plotly_chart(fig_mix, use_container_width=True)

    st.divider()
    st.subheader("해석 가이드(대시보드용)")
    st.write(
        """
- **산점도**: x축 EC가 커질수록 생중량이 어떻게 변하는지 확인하면서, **온도(마커 크기)**까지 함께 비교합니다.  
- **융합 꺾은선(이중축)**: 동일한 x축(EC) 위에서 **생중량(좌축)과 온도(우축)**를 동시에 보면,
  ‘생중량 변화가 온도 때문인지, EC 때문인지’를 직관적으로 분리해 볼 수 있습니다.
"""
    )

    with st.expander("📄 데이터 테이블 + CSV 다운로드"):
        st.dataframe(mline, use_container_width=True)
        csv_bytes = mline.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", data=csv_bytes, file_name="학교별_요약지표.csv", mime="text/csv")

st.caption("© Polar Plant Dashboard — Streamlit / Plotly")
