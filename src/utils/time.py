import pandas as pd

def add_time_columns(df: pd.DataFrame, col: str):
    ts = pd.to_datetime(df[col], errors="coerce")
    df[col + "_year"] = ts.dt.year
    df[col + "_month"] = ts.dt.month
    df[col + "_dow"] = ts.dt.dayofweek
    df[col + "_hour"] = ts.dt.hour
    return df