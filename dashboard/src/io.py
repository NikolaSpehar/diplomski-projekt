import pandas as pd
import streamlit as st


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    # encoding="utf-8-sig" strips BOM if present
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalize column names:
    # - remove BOM if still present
    # - normalize non-breaking spaces
    # - strip leading/trailing whitespace
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    return df
