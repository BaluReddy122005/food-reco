import pandas as pd
import numpy as np
from ..config import (
    W_CUSTOMER_FREQ, W_LOC_FREQ, W_GLOBAL_POP, W_DISTANCE
)
from ..utils.distance import euclidean_distance

def score_candidates(
    test_pair_row,
    vendors_df,
    cust_vendor_freq,
    loc_vendor_freq,
    vendor_pop,
    test_location_lat,
    test_location_lon
):
    customer_id = test_pair_row["customer_id"]
    loc_num = test_pair_row["location_number"]

    # Merge candidate scores
    df = vendors_df[["id", "latitude", "longitude"]].rename(columns={"id": "vendor_id"}).copy()

    # customer-vendor
    df = df.merge(
        cust_vendor_freq[cust_vendor_freq["customer_id"] == customer_id][
            ["vendor_id", "cust_vendor_freq_norm"]
        ],
        on="vendor_id",
        how="left",
    )

    # location-vendor
    df = df.merge(
        loc_vendor_freq[loc_vendor_freq["LOCATION_NUMBER"] == loc_num][
            ["vendor_id", "loc_vendor_freq_norm"]
        ],
        on="vendor_id",
        how="left",
    )

    # global popularity
    df = df.merge(
        vendor_pop[["vendor_id", "vendor_pop_norm"]],
        on="vendor_id",
        how="left",
    )

    # distance
    df["distance"] = euclidean_distance(
        test_location_lat, test_location_lon,
        df["latitude"], df["longitude"]
    )
    df["distance_score"] = 1.0 / (1.0 + df["distance"])

    # fill NaNs with 0
    for c in ["cust_vendor_freq_norm", "loc_vendor_freq_norm", "vendor_pop_norm"]:
        if c in df:
            df[c] = df[c].fillna(0.0)
    df["distance_score"] = df["distance_score"].fillna(0.0)

    # weighted score
    df["score"] = (
        W_CUSTOMER_FREQ * df["cust_vendor_freq_norm"] +
        W_LOC_FREQ * df["loc_vendor_freq_norm"] +
        W_GLOBAL_POP * df["vendor_pop_norm"] +
        W_DISTANCE * df["distance_score"]
    )
    return df.sort_values("score", ascending=False)