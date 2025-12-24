import io
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
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

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
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig


# =========================
# Constants (allowed: experiment configuration)
# =========================
SCHOOL_ORDER = ["송도고", "하늘고", "아라고", "동산고"]
TARGET_EC_BY_SCHOOL = {
    "송도고": 1.0,
    "하늘고": 2.0,  # 최적
    "아라고": 4.0,
    "동산고": 8.0,
}
COLOR_BY_SCHOOL = {
    "송도고": "#2E86AB",
    "하늘고": "#F39C12",  # 최적 강조
    "아라고": "#27AE60",
    "동산고": "#8E44AD",
}
OPTIMAL_EC = 2.0


# =========================
# Unicode-safe file matching (NFC/NFD)
# =========================
def _norm_all(s: str) -> set[str]:
    # 양방향 비교용: NFC/NFD 모두 생성
    return {
        unicodedata.normalize("NFC", s),
        unicodedata.normalize("NFD", s),
    }


def find_file_by_name(directory: Path, desired_name: str) -> Path | None:
    """
    pathlib.Path.iterdir()로만 탐색하며,
    desired_name과 실제 파일명을 NFC/NFD 양방향 normalize해서 비교.
    - f-string 조합으로 경로 만들지 않음
    - glob 패턴만 사용하지 않음
    """
    if not directory.exists():
        return None

    desired_norms = _norm_all(desired_name)

    for p in directory.iterdir():
        if not p.is_file():
            continue
        candidate_norms = _norm_all(p.name)
        if desired_norms.intersection(candidate_norms):
            return p
    return None


def locate_data_dir() -> Path:
    """
    Streamlit Cloud / 로컬 모두 대응:
    1) 현재 파일 기준 ./data
    2) 현재 작업 디렉토리 ./data
    3) (로컬 테스트) /mnt/data
    """
    here = Path(__file__).resolve().parent
    cand1 = here / "data"
    if cand1.exists():
        return cand1

    cand2 = Path.cwd() / "data"
    if cand2.exists():
        return cand2

    # 업로드 파일이 /mnt/data에 있는 로컬/샌드박스 환경 대응
    cand3 = Path("/mnt/data")
    if cand3.exists():
        return cand3

    return cand1  # 기본값 (존재하지 않아도 반환)


# =========================
# Robust column helpers
# =========================
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    norm_map = {unicodedata.normalize("NFC", c): c for c in cols}
    for cand in candidates:
        cand_nfc = unicodedata.normalize("NFC", cand)
        if cand_nfc in norm_map:
            return norm_map[cand_nfc]
    return None


def ensure_datetime(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col])
    return out.sort_values(time_col)


# =========================
# Data loading (cached)
# =========================
@st.cache_data(show_spinner=False)
def load_environment_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    학교별 환경 CSV를 NFC/NFD 안전하게 탐색하여 로딩.
    반환: {학교명: df}
    """
    env = {}
    # 원하는 파일명은 "학교명_환경데이터.csv" 형태로 주어짐 (하지만 경로 조합 f-string 금지 -> 이름만 전달)
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
        df = pd.read_csv(p)

        # 필수 컬럼 확인
        needed = ["time", "temperature", "humidity", "ph", "ec"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            # 최소한 time만 있으면 일부라도 표시 가능하지만, 분석 목적상 명확히 에러 표시를 위해 그대로 둠
            pass

        env[school] = df

    return env


@st.cache_data(show_spinner=False)
def load_growth_data(data_dir: Path) -> tuple[pd.DataFrame, list[str], Path | None]:
    """
    XLSX 파일을 NFC/NFD 안전하게 탐색해 로딩.
    - 시트명 하드코딩 금지 -> 엑셀에서 sheet_names를 동적으로 가져옴
    반환:
      (all_growth_df, sheet_names, xlsx_path)
    """
    desired_name = "4개교_생육결과데이터.xlsx"
    xlsx_path = find_file_by_name(data_dir, desired_name)
    if xlsx_path is None:
        # 혹시 업로드 환경에서 확장자 중복(예: .xlsx.xlsx) 형태로 존재할 수 있어 한 번 더 탐색
        alt_name = "4개교_생육결과데이터.xlsx.xlsx"
        xlsx_path = find_file_by_name(data_dir, alt_name)

    if xlsx_path is None:
        return pd.DataFrame(), [], None

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = list(xls.sheet_names)

    frames = []
    for sh in sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
        df["학교"] = sh  # 시트명이 학교명
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, sheet_names, xlsx_path


# =========================
# Metrics / Summaries
# =========================
def env_summary(env_by_school: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for school in SCHOOL_ORDER:
        df = env_by_school.get(school)
        if df is None or df.empty:
            continue
        # 안전하게 평균 계산
        row = {"학교": school}
        for col in ["temperature", "humidity", "ph", "ec"]:
            row[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    return out


def growth_summary(growth_df: pd.DataFrame) -> pd.DataFrame:
    if growth_df is None or growth_df.empty:
        return pd.DataFrame()

    # 한국어 컬럼명 대응
    col_leaves = pick_col(growth_df, ["잎 수(장)", "잎수(장)", "잎 수", "잎수"])
    col_shoot = pick_col(growth_df, ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이"])
    col_weight = pick_col(growth_df, ["생중량(g)", "생중량"])

    if col_weight is None:
        return pd.DataFrame()

    df = growth_df.copy()
    df[col_weight] = pd.to_numeric(df[col_weight], errors="coerce")
    if col_leaves is not None:
        df[col_leaves] = pd.to_numeric(df[col_leaves], errors="coerce")
    if col_shoot is not None:
        df[col_shoot] = pd.to_numeric(df[col_shoot], errors="coerce")

    grp = df.groupby("학교", dropna=False)
    rows = []
    for school in SCHOOL_ORDER:
        if school not in grp.groups:
            continue
        g = grp.get_group(school)
        rows.append(
            {
                "학교": school,
                "개체수": int(g.shape[0]),
                "평균_생중량": float(g[col_weight].mean()),
                "평균_잎수": float(g[col_leaves].mean()) if col_leaves else float("nan"),
                "평균_지상부길이": float(g[col_shoot].mean()) if col_shoot else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_optimal_ec_from_growth(gsum: pd.DataFrame) -> float | None:
    """
    실험 설계상 '하늘고 EC 2.0 최적'이지만,
    데이터 기반으로도 '평균 생중량' 최대값의 학교 EC를 최적값으로 계산.
    """
    if gsum is None or gsum.empty or "평균_생중량" not in gsum.columns:
        return None
    tmp = gsum.dropna(subset=["평균_생중량"]).copy()
    if tmp.empty:
        return None
    best_school = tmp.sort_values("평균_생중량", ascending=False).iloc[0]["학교"]
    return TARGET_EC_BY_SCHOOL.get(best_school)


# =========================
# UI: Sidebar
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

data_dir = locate_data_dir()

with st.sidebar:
    st.header("⚙️ 설정")
    school_option = st.selectbox(
        "학교 선택",
        ["전체"] + SCHOOL_ORDER,
        index=0,
    )
    st.caption("※ Streamlit Cloud에서 한글 파일명(NFC/NFD) 인식 오류를 방지하도록 설계됨")


# =========================
# Load Data
# =========================
with st.spinner("데이터를 불러오는 중..."):
    env_by_school = load_environment_data(data_dir)
    growth_df, sheet_names, growth_path = load_growth_data(data_dir)

# Validation
missing_env = [s for s in SCHOOL_ORDER if s not in env_by_school]
if missing_env:
    st.warning(f"환경 데이터가 없는 학교: {', '.join(missing_env)} (data/ 폴더 파일명 확인)")

if growth_df is None or growth_df.empty:
    st.error("생육 결과 XLSX 데이터를 불러오지 못했습니다. data/ 폴더에 '4개교_생육결과데이터.xlsx'가 있는지 확인하세요.")
    st.stop()

# Precompute summaries
env_sum = env_summary(env_by_school)
grow_sum = growth_summary(growth_df)
optimal_ec_from_data = compute_optimal_ec_from_growth(grow_sum)

# Global metrics for Tab1
total_individuals = int(growth_df.shape[0])

# overall env mean (concat available)
env_concat = []
for s in SCHOOL_ORDER:
    df = env_by_school.get(s)
    if df is not None and not df.empty:
        env_concat.append(df)
env_all = pd.concat(env_concat, ignore_index=True) if env_concat else pd.DataFrame()
avg_temp = pd.to_numeric(env_all.get("temperature", pd.Series(dtype=float)), errors="coerce").mean() if not env_all.empty else float("nan")
avg_hum = pd.to_numeric(env_all.get("humidity", pd.Series(dtype=float)), errors="coerce").mean() if not env_all.empty else float("nan")


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
본 연구는 4개 학교가 서로 다른 EC(양액 전기전도도) 조건에서 극지식물을 재배한 결과를 비교하여,
**생육 지표(생중량·잎 수·길이)**가 가장 우수한 **최적 EC 농도**를 도출하는 것을 목표로 합니다.

- 학교별 환경 데이터: 온도/습도/pH/EC의 시간 변화 및 평균 비교
- 생육 결과 데이터: 학교(=EC 조건)별 생육 성과 비교 및 최적 조건 판단
"""
    )

    # EC condition table
    st.subheader("학교별 EC 조건")
    rows = []
    for school in SCHOOL_ORDER:
        # 개체수는 엑셀 시트에서 계산 (시트 하드코딩 없이 로딩된 후 groupby)
        n = int((growth_df["학교"] == school).sum()) if "학교" in growth_df.columns else 0
        rows.append(
            {
                "학교명": school,
                "EC 목표": TARGET_EC_BY_SCHOOL.get(school),
                "개체수": n,
                "색상": COLOR_BY_SCHOOL.get(school),
            }
        )
    ec_table = pd.DataFrame(rows)
    st.dataframe(ec_table, use_container_width=True)

    # KPI cards
    st.subheader("주요 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_individuals:,} 개체")
    c2.metric("평균 온도", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f} °C")
    c3.metric("평균 습도", "-" if pd.isna(avg_hum) else f"{avg_hum:.2f} %")
    # 최적 EC: 데이터 기반 + 실험 설계(하늘고 EC 2.0) 모두 보여주되, 최종 표시는 2.0 고정 강조
    if optimal_ec_from_data is None:
        c4.metric("최적 EC", f"{OPTIMAL_EC:.1f} (하늘고)")
    else:
        label = f"{OPTIMAL_EC:.1f} (하늘고)"
        c4.metric("최적 EC", label)

# -------------------------
# Tab 2: Environment
# -------------------------
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_sum is None or env_sum.empty:
        st.error("환경 데이터 평균을 계산할 수 없습니다. CSV 컬럼(time, temperature, humidity, ph, ec)을 확인하세요.")
    else:
        # Ensure order
        env_sum_plot = env_sum.copy()
        env_sum_plot["학교"] = pd.Categorical(env_sum_plot["학교"], categories=SCHOOL_ORDER, ordered=True)
        env_sum_plot = env_sum_plot.sort_values("학교")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC(평균)"),
        )

        # Bars: temp
        fig.add_trace(
            go.Bar(
                x=env_sum_plot["학교"],
                y=env_sum_plot["temperature"],
                name="평균 온도",
            ),
            row=1, col=1
        )
        # Bars: humidity
        fig.add_trace(
            go.Bar(
                x=env_sum_plot["학교"],
                y=env_sum_plot["humidity"],
                name="평균 습도",
            ),
            row=1, col=2
        )
        # Bars: pH
        fig.add_trace(
            go.Bar(
                x=env_sum_plot["학교"],
                y=env_sum_plot["ph"],
                name="평균 pH",
            ),
            row=2, col=1
        )

        # Double bar: target vs measured ec
        target_ec = [TARGET_EC_BY_SCHOOL.get(s, None) for s in env_sum_plot["학교"].astype(str)]
        fig.add_trace(
            go.Bar(x=env_sum_plot["학교"], y=target_ec, name="목표 EC"),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=env_sum_plot["학교"], y=env_sum_plot["ec"], name="실측 EC(평균)"),
            row=2, col=2
        )

        fig.update_layout(barmode="group", height=650, title="학교별 환경 평균(요약)")
        fig = apply_plotly_korean_font(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    # Build time-series dataset
    ts_frames = []
    for s in SCHOOL_ORDER:
        df = env_by_school.get(s)
        if df is None or df.empty:
            continue
        # time column must exist
        if "time" not in df.columns:
            continue
        tmp = df.copy()
        tmp["학교"] = s
        ts_frames.append(tmp)

    ts_all = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()
    if ts_all.empty:
        st.error("시계열 그래프를 그릴 환경 데이터가 없습니다. CSV에 time 컬럼이 있는지 확인하세요.")
    else:
        ts_all = ensure_datetime(ts_all, "time")

        # filter by school
        if school_option != "전체":
            ts_show = ts_all[ts_all["학교"] == school_option].copy()
        else:
            ts_show = ts_all.copy()

        # numeric
        for col in ["temperature", "humidity", "ec"]:
            if col in ts_show.columns:
                ts_show[col] = pd.to_numeric(ts_show[col], errors="coerce")

        # Temperature
        if "temperature" in ts_show.columns:
            fig_t = px.line(ts_show, x="time", y="temperature", color="학교" if school_option == "전체" else None,
                            title="온도 변화(시간)")
            fig_t = apply_plotly_korean_font(fig_t)
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.warning("temperature 컬럼이 없어 온도 그래프를 생략합니다.")

        # Humidity
        if "humidity" in ts_show.columns:
            fig_h = px.line(ts_show, x="time", y="humidity", color="학교" if school_option == "전체" else None,
                            title="습도 변화(시간)")
            fig_h = apply_plotly_korean_font(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.warning("humidity 컬럼이 없어 습도 그래프를 생략합니다.")

        # EC + target line
        if "ec" in ts_show.columns:
            fig_ec = px.line(ts_show, x="time", y="ec", color="학교" if school_option == "전체" else None,
                             title="EC 변화(시간) - 목표 EC 기준선 포함")
            # add horizontal target line(s)
            if school_option == "전체":
                # 학교별로 기준선 여러개는 복잡해질 수 있어: 최적(2.0)만 기준선으로 표시
                fig_ec.add_hline(y=OPTIMAL_EC, line_dash="dash", annotation_text="최적 EC(2.0) 기준선")
            else:
                target = TARGET_EC_BY_SCHOOL.get(school_option)
                if target is not None:
                    fig_ec.add_hline(y=target, line_dash="dash", annotation_text=f"목표 EC({target})")
            fig_ec = apply_plotly_korean_font(fig_ec)
            st.plotly_chart(fig_ec, use_container_width=True)
        else:
            st.warning("ec 컬럼이 없어 EC 그래프를 생략합니다.")

        with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
            # Show filtered raw
            st.dataframe(ts_show, use_container_width=True)

            csv_bytes = ts_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ CSV 다운로드",
                data=csv_bytes,
                file_name="환경데이터_원본.csv",
                mime="text/csv",
            )

# -------------------------
# Tab 3: Growth Results
# -------------------------
with tab3:
    st.subheader("🥇 핵심 결과")

    # columns
    col_weight = pick_col(growth_df, ["생중량(g)", "생중량"])
    col_leaves = pick_col(growth_df, ["잎 수(장)", "잎수(장)", "잎 수", "잎수"])
    col_shoot = pick_col(growth_df, ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이"])

    if col_weight is None:
        st.error("생육 데이터에서 생중량 컬럼을 찾지 못했습니다. 컬럼명을 확인하세요.")
        st.stop()

    # summary for display
    if grow_sum is None or grow_sum.empty:
        st.error("생육 요약표를 생성할 수 없습니다.")
    else:
        # Best EC (by weight)
        best_row = grow_sum.dropna(subset=["평균_생중량"]).sort_values("평균_생중량", ascending=False).head(1)
        if best_row.empty:
            st.error("평균 생중량 계산이 불가능합니다(결측치 확인).")
        else:
            best_school = best_row.iloc[0]["학교"]
            best_ec = TARGET_EC_BY_SCHOOL.get(best_school)
            best_w = best_row.iloc[0]["평균_생중량"]

            # Card 강조: 하늘고(EC2.0) 최적 표시 (요구사항)
            cA, cB, cC = st.columns([1.2, 1.2, 2])
            cA.metric("데이터 기준 평균 생중량 최댓값", f"{best_w:.3f} g", f"{best_school} (EC {best_ec})")
            cB.metric("최적 EC (연구 결론)", f"{OPTIMAL_EC:.1f}", "하늘고(EC 2.0) 최적")
            cC.info("※ ‘최적 EC’는 연구 설계상 **하늘고(EC 2.0)** 를 최적 조건으로 결론 내리며, 동시에 데이터 기반 최댓값도 함께 제시합니다.")

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    # Filter summary by school selection
    gsum_plot = grow_sum.copy() if grow_sum is not None else pd.DataFrame()
    if not gsum_plot.empty:
        if school_option != "전체":
            gsum_plot = gsum_plot[gsum_plot["학교"] == school_option]
        gsum_plot["학교"] = pd.Categorical(gsum_plot["학교"], categories=SCHOOL_ORDER, ordered=True)
        gsum_plot = gsum_plot.sort_values("학교")

        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=("평균 생중량 (⭐ 가장 중요)", "평균 잎 수", "평균 지상부 길이(mm)", "개체수 비교"),
        )
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["평균_생중량"], name="평균 생중량"), row=1, col=1)
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["평균_잎수"], name="평균 잎 수"), row=1, col=2)
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["평균_지상부길이"], name="평균 지상부 길이"), row=2, col=1)
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["개체수"], name="개체수"), row=2, col=2)

        # 하늘고(EC2.0) 강조를 위한 주석
        if school_option in ("전체", "하늘고"):
            fig2.add_annotation(
                text="최적(하늘고 EC 2.0)",
                x="하늘고",
                y=float(grow_sum[grow_sum["학교"] == "하늘고"]["평균_생중량"].iloc[0]) if (grow_sum is not None and (grow_sum["학교"] == "하늘고").any()) else 0,
                showarrow=True,
                arrowhead=2,
                xref="x1",
                yref="y1",
            )

        fig2.update_layout(barmode="group", height=650, title="학교(=EC 조건)별 생육 비교")
        fig2 = apply_plotly_korean_font(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

    # Build growth df for plots
    gdf = growth_df.copy()
    gdf[col_weight] = pd.to_numeric(gdf[col_weight], errors="coerce")
    if col_leaves is not None:
        gdf[col_leaves] = pd.to_numeric(gdf[col_leaves], errors="coerce")
    if col_shoot is not None:
        gdf[col_shoot] = pd.to_numeric(gdf[col_shoot], errors="coerce")

    if school_option != "전체":
        gdf = gdf[gdf["학교"] == school_option].copy()

    fig_box = px.violin(
        gdf.dropna(subset=[col_weight]),
        x="학교",
        y=col_weight,
        box=True,
        points="all",
        title="생중량 분포(바이올린 + 박스)",
    )
    fig_box = apply_plotly_korean_font(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.subheader("상관관계 분석 (산점도 2개)")

    c1, c2 = st.columns(2)

    with c1:
        if col_leaves is None:
            st.warning("잎 수 컬럼을 찾지 못해 '잎 수 vs 생중량' 산점도를 생략합니다.")
        else:
            fig_sc1 = px.scatter(
                gdf.dropna(subset=[col_leaves, col_weight]),
                x=col_leaves,
                y=col_weight,
                color="학교" if school_option == "전체" else None,
                title="잎 수 vs 생중량",
                labels={col_leaves: "잎 수(장)", col_weight: "생중량(g)"},
            )
            fig_sc1 = apply_plotly_korean_font(fig_sc1)
            st.plotly_chart(fig_sc1, use_container_width=True)

    with c2:
        if col_shoot is None:
            st.warning("지상부 길이 컬럼을 찾지 못해 '지상부 길이 vs 생중량' 산점도를 생략합니다.")
        else:
            fig_sc2 = px.scatter(
                gdf.dropna(subset=[col_shoot, col_weight]),
                x=col_shoot,
                y=col_weight,
                color="학교" if school_option == "전체" else None,
                title="지상부 길이 vs 생중량",
                labels={col_shoot: "지상부 길이(mm)", col_weight: "생중량(g)"},
            )
            fig_sc2 = apply_plotly_korean_font(fig_sc2)
            st.plotly_chart(fig_sc2, use_container_width=True)

    with st.expander("📄 학교별 생육 데이터 원본 + XLSX 다운로드"):
        st.dataframe(gdf, use_container_width=True)

        # XLSX 다운로드(필수: BytesIO, 경로 없이 to_excel 호출)
        buffer = io.BytesIO()
        # 인덱스 제외
        gdf.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            label="⬇️ XLSX 다운로드",
            data=buffer,
            file_name="생육데이터_원본.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# Footer
st.caption("© Polar Plant EC Dashboard — Streamlit / Plotly")
