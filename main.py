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
# Experiment constants
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
# Robust path & filename (NFC/NFD safe)
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

    return cand1  # 기본


def _norm_all(s: str) -> set[str]:
    return {
        unicodedata.normalize("NFC", s),
        unicodedata.normalize("NFD", s),
    }


def canonical_filename(name: str) -> str:
    """
    비교용 표준화:
    - NFC 정규화
    - 공백 제거
    - 중복 확장자 보정(.csv.csv / .xlsx.xlsx)
    """
    n = unicodedata.normalize("NFC", str(name)).strip()
    low = n.lower()
    if low.endswith(".csv.csv"):
        n = n[:-4]  # 마지막 ".csv" 제거
    if low.endswith(".xlsx.xlsx"):
        n = n[:-5]  # 마지막 ".xlsx" 제거
    return n


def filename_match(candidate: str, desired: str) -> bool:
    c_nfc = canonical_filename(candidate)
    d_nfc = canonical_filename(desired)

    c_nfd = unicodedata.normalize("NFD", c_nfc)
    d_nfd = unicodedata.normalize("NFD", d_nfc)

    # 완전일치
    if c_nfc == d_nfc or c_nfd == d_nfd:
        return True
    # endswith 허용 (중복 확장자/경로차 흡수)
    if c_nfc.endswith(d_nfc) or c_nfd.endswith(d_nfd):
        return True
    return False


def find_file_by_name(directory: Path, desired_name: str) -> Path | None:
    """
    pathlib.Path.iterdir()로만 탐색하며,
    NFC/NFD 양방향 비교 + 중복확장자/endswith까지 흡수.
    """
    if not directory.exists():
        return None

    # desired도 NFC/NFD set로 한번 더 안전하게
    desired_norms = _norm_all(canonical_filename(desired_name))

    for p in directory.iterdir():
        if not p.is_file():
            continue

        cand_name = canonical_filename(p.name)
        cand_norms = _norm_all(cand_name)

        # 1) NFC/NFD 교집합(완전일치급)
        if desired_norms.intersection(cand_norms):
            return p

        # 2) 보강 매칭
        if filename_match(p.name, desired_name):
            return p

    return None


# =========================
# CSV read & column standardization
# =========================
def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    한글 CSV 인코딩 이슈 방지:
    utf-8-sig -> utf-8 -> cp949 -> euc-kr 순으로 시도
    """
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def normalize_colname(c: str) -> str:
    """
    컬럼명 정규화:
    - BOM 제거
    - 소문자
    - 공백 제거
    - 기호 최소화
    """
    c = unicodedata.normalize("NFC", str(c)).strip().lower()
    c = c.replace("\ufeff", "")
    c = re.sub(r"\s+", "", c)
    c = c.replace("-", "").replace(".", "")
    return c


def standardize_env_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명이 살짝 달라도 time/temperature/humidity/ph/ec로 자동 정리
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
# Growth helpers
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
# Data loading (cached)
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

    # 업로드 환경에서 확장자가 중복되는 경우 대비
    if xlsx_path is None:
        xlsx_path = find_file_by_name(data_dir, "4개교_생육결과데이터.xlsx.xlsx")

    if xlsx_path is None:
        return pd.DataFrame(), [], None

    xls = pd.ExcelFile(xlsx_path)
    sheet_names = list(xls.sheet_names)  # ✅ 시트명 하드코딩 금지

    frames = []
    for sh in sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
        df["학교"] = sh
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, sheet_names, xlsx_path


# =========================
# Summaries
# =========================
def env_summary(env_by_school: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for school in SCHOOL_ORDER:
        df = env_by_school.get(school)
        if df is None or df.empty:
            continue
        row = {"학교": school}
        for col in ["temperature", "humidity", "ph", "ec"]:
            row[col] = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def growth_summary(growth_df: pd.DataFrame) -> pd.DataFrame:
    if growth_df is None or growth_df.empty:
        return pd.DataFrame()

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
    if gsum is None or gsum.empty or "평균_생중량" not in gsum.columns:
        return None
    tmp = gsum.dropna(subset=["평균_생중량"]).copy()
    if tmp.empty:
        return None
    best_school = tmp.sort_values("평균_생중량", ascending=False).iloc[0]["학교"]
    return TARGET_EC_BY_SCHOOL.get(best_school)


# =========================
# UI
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

data_dir = get_data_dir()

with st.sidebar:
    st.header("⚙️ 설정")
    school_option = st.selectbox("학교 선택", ["전체"] + SCHOOL_ORDER, index=0)

    # ✅ 디버그: data 폴더 실제 인식 확인
    with st.expander("🧪 디버그: data 폴더/파일 확인"):
        st.write("data_dir =", str(data_dir))
        if data_dir.exists():
            st.write("files =", [p.name for p in data_dir.iterdir() if p.is_file()])
        else:
            st.error("data 폴더를 찾지 못했습니다.")


with st.spinner("데이터를 불러오는 중..."):
    env_by_school = load_environment_data(data_dir)
    growth_df, sheet_names, growth_path = load_growth_data(data_dir)

missing_env = [s for s in SCHOOL_ORDER if s not in env_by_school]
if missing_env:
    st.warning(f"환경 데이터가 없는 학교: {', '.join(missing_env)} (data/ 폴더 파일명 또는 인코딩/컬럼 확인)")

if growth_df is None or growth_df.empty:
    st.error("생육 결과 XLSX 데이터를 불러오지 못했습니다. data/ 폴더에 '4개교_생육결과데이터.xlsx'가 있는지 확인하세요.")
    st.stop()

env_sum = env_summary(env_by_school)
grow_sum = growth_summary(growth_df)
optimal_ec_from_data = compute_optimal_ec_from_growth(grow_sum)

total_individuals = int(growth_df.shape[0])

env_concat = []
for s in SCHOOL_ORDER:
    df = env_by_school.get(s)
    if df is not None and not df.empty:
        env_concat.append(df)
env_all = pd.concat(env_concat, ignore_index=True) if env_concat else pd.DataFrame()
avg_temp = pd.to_numeric(env_all.get("temperature", pd.Series(dtype=float)), errors="coerce").mean() if not env_all.empty else float("nan")
avg_hum = pd.to_numeric(env_all.get("humidity", pd.Series(dtype=float)), errors="coerce").mean() if not env_all.empty else float("nan")

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# =========================
# Tab 1
# =========================
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

    st.subheader("학교별 EC 조건")
    rows = []
    for school in SCHOOL_ORDER:
        n = int((growth_df["학교"] == school).sum()) if "학교" in growth_df.columns else 0
        rows.append(
            {
                "학교명": school,
                "EC 목표": TARGET_EC_BY_SCHOOL.get(school),
                "개체수": n,
                "색상": COLOR_BY_SCHOOL.get(school),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("주요 지표")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_individuals:,} 개체")
    c2.metric("평균 온도", "-" if pd.isna(avg_temp) else f"{avg_temp:.2f} °C")
    c3.metric("평균 습도", "-" if pd.isna(avg_hum) else f"{avg_hum:.2f} %")
    c4.metric("최적 EC", f"{OPTIMAL_EC:.1f} (하늘고)")


# =========================
# Tab 2
# =========================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    if env_sum is None or env_sum.empty:
        st.error("환경 데이터 평균을 계산할 수 없습니다. CSV 컬럼(time, temperature, humidity, ph, ec)을 확인하세요.")
    else:
        env_sum_plot = env_sum.copy()
        env_sum_plot["학교"] = pd.Categorical(env_sum_plot["학교"], categories=SCHOOL_ORDER, ordered=True)
        env_sum_plot = env_sum_plot.sort_values("학교")

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC 비교(평균)"),
        )

        fig.add_trace(go.Bar(x=env_sum_plot["학교"], y=env_sum_plot["temperature"], name="평균 온도"), row=1, col=1)
        fig.add_trace(go.Bar(x=env_sum_plot["학교"], y=env_sum_plot["humidity"], name="평균 습도"), row=1, col=2)
        fig.add_trace(go.Bar(x=env_sum_plot["학교"], y=env_sum_plot["ph"], name="평균 pH"), row=2, col=1)

        target_ec = [TARGET_EC_BY_SCHOOL.get(str(s), None) for s in env_sum_plot["학교"].astype(str)]
        fig.add_trace(go.Bar(x=env_sum_plot["학교"], y=target_ec, name="목표 EC"), row=2, col=2)
        fig.add_trace(go.Bar(x=env_sum_plot["학교"], y=env_sum_plot["ec"], name="실측 EC(평균)"), row=2, col=2)

        fig.update_layout(barmode="group", height=650, title="학교별 환경 평균(요약)")
        fig = apply_plotly_korean_font(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    ts_frames = []
    for s in SCHOOL_ORDER:
        df = env_by_school.get(s)
        if df is None or df.empty:
            continue
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
        ts_show = ts_all.copy() if school_option == "전체" else ts_all[ts_all["학교"] == school_option].copy()

        for col in ["temperature", "humidity", "ec"]:
            if col in ts_show.columns:
                ts_show[col] = pd.to_numeric(ts_show[col], errors="coerce")

        if "temperature" in ts_show.columns:
            fig_t = px.line(ts_show, x="time", y="temperature",
                            color="학교" if school_option == "전체" else None,
                            title="온도 변화(시간)")
            fig_t = apply_plotly_korean_font(fig_t)
            st.plotly_chart(fig_t, use_container_width=True)

        if "humidity" in ts_show.columns:
            fig_h = px.line(ts_show, x="time", y="humidity",
                            color="학교" if school_option == "전체" else None,
                            title="습도 변화(시간)")
            fig_h = apply_plotly_korean_font(fig_h)
            st.plotly_chart(fig_h, use_container_width=True)

        if "ec" in ts_show.columns:
            fig_ec = px.line(ts_show, x="time", y="ec",
                             color="학교" if school_option == "전체" else None,
                             title="EC 변화(시간) - 목표 EC 기준선 포함")
            if school_option == "전체":
                fig_ec.add_hline(y=OPTIMAL_EC, line_dash="dash", annotation_text="최적 EC(2.0) 기준선")
            else:
                t = TARGET_EC_BY_SCHOOL.get(school_option)
                if t is not None:
                    fig_ec.add_hline(y=t, line_dash="dash", annotation_text=f"목표 EC({t})")
            fig_ec = apply_plotly_korean_font(fig_ec)
            st.plotly_chart(fig_ec, use_container_width=True)

        with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
            st.dataframe(ts_show, use_container_width=True)
            csv_bytes = ts_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ CSV 다운로드", data=csv_bytes, file_name="환경데이터_원본.csv", mime="text/csv")


# =========================
# Tab 3
# =========================
with tab3:
    st.subheader("🥇 핵심 결과")

    col_weight = pick_col(growth_df, ["생중량(g)", "생중량"])
    col_leaves = pick_col(growth_df, ["잎 수(장)", "잎수(장)", "잎 수", "잎수"])
    col_shoot = pick_col(growth_df, ["지상부 길이(mm)", "지상부길이(mm)", "지상부 길이", "지상부길이"])

    if col_weight is None:
        st.error("생육 데이터에서 생중량 컬럼을 찾지 못했습니다. 컬럼명을 확인하세요.")
        st.stop()

    if grow_sum is not None and not grow_sum.empty:
        best_row = grow_sum.dropna(subset=["평균_생중량"]).sort_values("평균_생중량", ascending=False).head(1)
        if not best_row.empty:
            best_school = best_row.iloc[0]["학교"]
            best_ec = TARGET_EC_BY_SCHOOL.get(best_school)
            best_w = best_row.iloc[0]["평균_생중량"]

            a, b, c = st.columns([1.2, 1.2, 2])
            a.metric("데이터 기준 평균 생중량 최댓값", f"{best_w:.3f} g", f"{best_school} (EC {best_ec})")
            b.metric("최적 EC (연구 결론)", f"{OPTIMAL_EC:.1f}", "하늘고(EC 2.0) 최적")
            c.info("※ 최적 EC는 연구 설계상 **하늘고(EC 2.0)** 를 최적 조건으로 결론 내립니다.")

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    gsum_plot = grow_sum.copy() if grow_sum is not None else pd.DataFrame()
    if not gsum_plot.empty:
        if school_option != "전체":
            gsum_plot = gsum_plot[gsum_plot["학교"] == school_option]
        gsum_plot["학교"] = pd.Categorical(gsum_plot["학교"], categories=SCHOOL_ORDER, ordered=True)
        gsum_plot = gsum_plot.sort_values("학교")

        fig2 = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("평균 생중량 (⭐ 가장 중요)", "평균 잎 수", "평균 지상부 길이(mm)", "개체수 비교"),
        )
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["평균_생중량"], name="평균 생중량"), row=1, col=1)
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["평균_잎수"], name="평균 잎 수"), row=1, col=2)
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["평균_지상부길이"], name="평균 지상부 길이"), row=2, col=1)
        fig2.add_trace(go.Bar(x=gsum_plot["학교"], y=gsum_plot["개체수"], name="개체수"), row=2, col=2)

        fig2.update_layout(barmode="group", height=650, title="학교(=EC 조건)별 생육 비교")
        fig2 = apply_plotly_korean_font(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

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

    left, right = st.columns(2)

    with left:
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

    with right:
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

        buffer = io.BytesIO()
        gdf.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            label="⬇️ XLSX 다운로드",
            data=buffer,
            file_name="생육데이터_원본.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.caption("© Polar Plant EC Dashboard — Streamlit / Plotly")
