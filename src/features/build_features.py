import pandas as pd
import numpy as np

def compute_customer_vendor_freq(orders: pd.DataFrame) -> pd.DataFrame:
    """#orders per (customer_id, vendor_id)"""
    freq = (
        orders.groupby(["customer_id", "vendor_id"])
        .size()
        .reset_index(name="cust_vendor_orders")
    )
    # Normalize per customer
    freq["cust_total_orders"] = freq.groupby("customer_id")["cust_vendor_orders"].transform("sum")
    freq["cust_vendor_freq_norm"] = freq["cust_vendor_orders"] / freq["cust_total_orders"]
    return freq

def compute_location_vendor_freq(orders: pd.DataFrame) -> pd.DataFrame:
    """#orders per (location_number, vendor_id)"""
    freq = (
        orders.groupby(["LOCATION_NUMBER", "vendor_id"])
        .size()
        .reset_index(name="loc_vendor_orders")
    )
    # Normalize per location
    freq["loc_total_orders"] = freq.groupby("LOCATION_NUMBER")["loc_vendor_orders"].transform("sum")
    freq["loc_vendor_freq_norm"] = freq["loc_vendor_orders"] / freq["loc_total_orders"]
    return freq

def compute_vendor_global_popularity(orders: pd.DataFrame) -> pd.DataFrame:
    pop = (
        orders.groupby("vendor_id")
        .size()
        .reset_index(name="vendor_orders")
    )
    pop["vendor_pop_norm"] = pop["vendor_orders"] / pop["vendor_orders"].sum()
    return pop

def recency_weighted_counts(orders: pd.DataFrame, time_col="created_at", halflife_days=60):
    """Optional: apply exponential decay to recency (not used directly in baseline but you can plug it in)."""
    if time_col not in orders.columns:
        return None
    orders["_ts"] = pd.to_datetime(orders[time_col], errors="coerce")
    max_ts = orders["_ts"].max()
    decay = np.exp(-np.log(2) * (max_ts - orders["_ts"]).dt.days / halflife_days)
    orders["_decay"] = decay.fillna(1.0)
    return orders

def normalize(col):
    cmin, cmax = col.min(), col.max()
    if cmax == cmin:
        return col * 0.0
    return (col - cmin) / (cmax - cmin)