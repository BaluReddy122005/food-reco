import pandas as pd
from ..config import DATA_PROCESSED
from ..features.build_features import (
    compute_customer_vendor_freq,
    compute_location_vendor_freq,
    compute_vendor_global_popularity
)

def main():
    orders = pd.read_parquet(DATA_PROCESSED / "orders.parquet")

    cust_vendor_freq = compute_customer_vendor_freq(orders)
    loc_vendor_freq = compute_location_vendor_freq(orders)
    vendor_pop = compute_vendor_global_popularity(orders)

    cust_vendor_freq.to_parquet(DATA_PROCESSED / "cust_vendor_freq.parquet", index=False)
    loc_vendor_freq.to_parquet(DATA_PROCESSED / "loc_vendor_freq.parquet", index=False)
    vendor_pop.to_parquet(DATA_PROCESSED / "vendor_pop.parquet", index=False)

if __name__ == "__main__":
    main()