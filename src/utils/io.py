import pandas as pd
from pathlib import Path

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)