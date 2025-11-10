# -*- coding: utf-8 -*-

# app.py
# Streamlit 웹앱: OpenCap .mot/.sto → CSV 변환 + 병합
# pip install streamlit pandas numpy

import io
import re
import zipfile
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os


st.set_page_config(page_title="MOT → CSV 변환기", page_icon="🚶", layout="wide")

st.title("🚶 OpenCap MOT → CSV 변환기")
st.write("여러 개의 .mot/.sto 파일을 업로드하면 CSV로 변환합니다. ‘병합’ 기능으로 파일명(=ID) 기준 세로 병합도 지원합니다. (OpenSim 설치 불필요)")
st.write("파일을 밑에 배너에 드래그 해주세요!")
st.caption("mimic")
# ---------------------------
# 유틸: .mot/.sto 헤더 파싱 & 본문 읽기
# ---------------------------
def load_hero():
    candidates = [
        "assets/hero.png",
        "hero.png",
        "static/hero.png",
        "/mnt/data/Gemini_Generated_Image_o7yi0xo7yi0xo7yi.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return Image.open(p)
    return None

hero = load_hero()
if hero is not None:
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(hero, use_container_width=True)  # ✅ no deprecation
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
else:
    st.info("메인 이미지를 표시하려면 `assets/hero.png`(권장) 또는 `hero.png`를 앱 폴더에 두세요.")

# ---------------------------
# 유틸: .mot/.sto 헤더 파싱 & 본문 읽기
# ---------------------------
def parse_opensim_table(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, str]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    header_meta = {}
    header_end_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end_idx = i
            break
        if ":" in line:
            k, v = line.split(":", 1)
            header_meta[k.strip()] = v.strip()

    if header_end_idx is None:
        header_end_idx = 0
        for i, line in enumerate(lines):
            if re.match(r"^\s*[-+]?(\d+(\.\d+)?([eE][-+]?\d+)?)", line.strip()):
                header_end_idx = i - 1
                break

    data_text = "\n".join(lines[header_end_idx + 1 :])

    try:
        df_try = pd.read_csv(io.StringIO(data_text), delim_whitespace=True)
        if all(str(c).replace(".", "", 1).isdigit() for c in df_try.columns[:2]):
            df = pd.read_csv(io.StringIO(data_text), delim_whitespace=True, header=None)
        else:
            df = df_try
    except Exception:
        df = pd.read_csv(io.StringIO(data_text), sep=r"[\\t\\s]+", engine="python", header=None)

    if df.shape[0] > 1:
        first_row = df.iloc[0].astype(str).tolist()
        if all(re.search(r"[A-Za-z_]", s) for s in first_row):
            df.columns = first_row
            df = df.iloc[1:].reset_index(drop=True)

    for cand in ["time", "Time", "t", "Time(s)"]:
        if cand in df.columns:
            df.rename(columns={cand: "time"}, inplace=True)
            break

    if "time" in df.columns:
        with np.errstate(all="ignore"):
            df["time"] = pd.to_numeric(df["time"], errors="coerce")

    for c in df.columns:
        if c == "time":
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df, header_meta


def sanitize_id_from_filename(name: str) -> str:
    base = re.sub(r"\\.mot$|\\.sto$|\\.txt$|\\.csv$", "", name, flags=re.IGNORECASE)
    base = re.sub(r"[^\\w\\-]+", "_", base).strip("_")
    return base or "ID"


# ---------------------------
# 사이드바: 옵션
# ---------------------------
st.sidebar.header("옵션")
merge_on = st.sidebar.checkbox("여러 파일 병합(파일명=ID)", value=True)
id_column_name = st.sidebar.text_input("ID 컬럼명", value="ID")
time_round = st.sidebar.selectbox("time 반올림(옵션)", options=["그대로", "소수점 3자리", "소수점 4자리"], index=0)
zip_prefix = st.sidebar.text_input("다운로드 ZIP/CSV 이름 접두사", value="opencap")

st.sidebar.markdown("---")
st.sidebar.caption("파일명 중복 시 자동으로 `_2`, `_3`를 ID에 부여합니다.")


# ---------------------------
# 본문: 업로드
# ---------------------------
files = st.file_uploader(
    "여러 개의 .mot/.sto 파일을 업로드하세요",
    type=["mot", "sto"],
    accept_multiple_files=True
)

btn_convert = st.button("🔄 변환 실행 (CSV 생성)")
btn_merge   = st.button("📎 병합 CSV 만들기", disabled=(not merge_on))

out_individual: List[Tuple[str, bytes]] = []  # (filename, csv_bytes)
merged_df: pd.DataFrame = pd.DataFrame()


def apply_time_round(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return df
    if time_round == "소수점 3자리":
        df["time"] = df["time"].round(3)
    elif time_round == "소수점 4자리":
        df["time"] = df["time"].round(4)
    return df


# ---------------------------
# 처리 로직
# ---------------------------
if files and (btn_convert or btn_merge):
    id_counts: Dict[str, int] = {}

    merged_rows = []
    with st.spinner("파일 처리 중..."):
        for up in files:
            raw = up.read()
            try:
                df, meta = parse_opensim_table(raw)
            except Exception as e:
                st.error(f"❌ 파싱 실패: {up.name} — {e}")
                continue

            df = apply_time_round(df)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            out_individual.append((f"{sanitize_id_from_filename(up.name)}.csv", csv_bytes))

            if merge_on:
                _id = sanitize_id_from_filename(up.name)
                if _id in id_counts:
                    id_counts[_id] += 1
                    _id = f"{_id}_{id_counts[_id]}"
                else:
                    id_counts[_id] = 1

                df_ = df.copy()
                df_.insert(0, id_column_name, _id)
                merged_rows.append(df_)

    st.subheader("📥 다운로드")

    if out_individual:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, b in out_individual:
                zf.writestr(fname, b)
        zip_name = f"{zip_prefix}_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        st.download_button(
            "개별 CSV (ZIP) 다운로드",
            data=zip_buf.getvalue(),
            file_name=zip_name,
            mime="application/zip"
        )

    if merge_on and merged_rows:
        merged_df = pd.concat(merged_rows, ignore_index=True)
        if "time" in merged_df.columns:
            merged_df.sort_values([id_column_name, "time"], inplace=True)
        csv_merged = merged_df.to_csv(index=False).encode("utf-8-sig")
        csv_name = f"{zip_prefix}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        st.download_button(
            "병합 CSV 다운로드",
            data=csv_merged,
            file_name=csv_name,
            mime="text/csv"
        )

    if out_individual:
        st.markdown("---")
        st.subheader("👀 미리보기")
        try:
            preview_bytes = files[0].getvalue()
            preview_df, _ = parse_opensim_table(preview_bytes)
            st.dataframe(preview_df.head(20), use_container_width=True)
        except Exception:
            pass

elif not files:
    st.info("좌측 또는 위의 영역에서 .mot/.sto 파일을 하나 이상 업로드하세요.")
