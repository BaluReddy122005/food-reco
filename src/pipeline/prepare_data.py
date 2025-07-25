import pandas as pd
from ..config import (
    DATA_RAW, DATA_PROCESSED,
    TRAIN_CUSTOMERS, TRAIN_LOCATIONS, ORDERS, VENDORS,
    TEST_CUSTOMERS, TEST_LOCATIONS
)
from ..utils.io import read_csv, ensure_dir

def main():
    ensure_dir(DATA_PROCESSED)

    # Just copy / light clean right now (parsing dates etc.)
    train_customers = read_csv(TRAIN_CUSTOMERS)
    train_locations = read_csv(TRAIN_LOCATIONS)
    orders = read_csv(ORDERS)
    vendors = read_csv(VENDORS)
    test_customers = read_csv(TEST_CUSTOMERS)
    test_locations = read_csv(TEST_LOCATIONS)

    # Optional: parse date columns that exist
    for col in ["created_at", "delivery_date", "delivered_time", "order_accepted_time"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Save to processed
    train_customers.to_parquet(DATA_PROCESSED / "train_customers.parquet", index=False)
    train_locations.to_parquet(DATA_PROCESSED / "train_locations.parquet", index=False)
    orders.to_parquet(DATA_PROCESSED / "orders.parquet", index=False)
    vendors.to_parquet(DATA_PROCESSED / "vendors.parquet", index=False)
    test_customers.to_parquet(DATA_PROCESSED / "test_customers.parquet", index=False)
    test_locations.to_parquet(DATA_PROCESSED / "test_locations.parquet", index=False)

if __name__ == "__main__":
    main()