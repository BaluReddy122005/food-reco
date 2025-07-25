import numpy as np
import pandas as pd

def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

def pairwise_distance(loc_df: pd.DataFrame, ven_df: pd.DataFrame):
    # Not used directly; we compute per (cust_loc, vendor) pair on the fly for memory reasons.
    pass