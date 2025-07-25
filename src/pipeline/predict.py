import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from ..config import (
    DATA_PROCESSED, SUBMISSIONS_DIR, SAMPLE_SUBMISSION, TOPK
)
from ..utils.io import ensure_dir, read_csv
from ..models.baseline import score_candidates

def build_submission_row(customer_id, loc_num, vendor_ids):
    # Adjust to the exact SampleSubmission format you have.
    # If it needs a single vendor, choose vendor_ids[0].
    # If it needs multiple, join them appropriately.
    # Here we’ll assume single best vendor per row.
    best_vendor = vendor_ids[0]
    return {
        "customer_id": customer_id,
        "location_number": loc_num,
        "vendor_id": best_vendor
    }

def main(topk: int):
    ensure_dir(SUBMISSIONS_DIR)

    # Load processed artifacts
    vendors = pd.read_parquet(DATA_PROCESSED / "vendors.parquet")
    test_locations = pd.read_parquet(DATA_PROCESSED / "test_locations.parquet")

    cust_vendor_freq = pd.read_parquet(DATA_PROCESSED / "cust_vendor_freq.parquet")
    loc_vendor_freq = pd.read_parquet(DATA_PROCESSED / "loc_vendor_freq.parquet")
    vendor_pop = pd.read_parquet(DATA_PROCESSED / "vendor_pop.parquet")

    # If the sample submission has some required ID column, load it to mirror shape:
    try:
        sample_sub = read_csv(SAMPLE_SUBMISSION)
        use_sample = True
    except Exception:
        sample_sub = None
        use_sample = False

    rows = []
    test_pairs = test_locations[["customer_id", "location_number", "latitude", "longitude"]].drop_duplicates()

    for _, r in tqdm(test_pairs.iterrows(), total=len(test_pairs), desc="Scoring test pairs"):
        scored = score_candidates(
            r,
            vendors,
            cust_vendor_freq,
            loc_vendor_freq,
            vendor_pop,
            r["latitude"],
            r["longitude"]
        )

        top_vendors = scored.head(topk)["vendor_id"].tolist()
        rows.append(build_submission_row(r["customer_id"], r["location_number"], top_vendors))

    submission = pd.DataFrame(rows)

    # If sample submission exists and uses a CID_LOC_VENDOR key, map accordingly here.
    # Example (commented):
    # submission["CID X LOC_NUM X VENDOR"] = submission.apply(
    #     lambda x: f"{x.customer_id}_{x.location_number}_{x.vendor_id}", axis=1
    # )

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SUBMISSIONS_DIR / f"submission_baseline_{ts}.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=TOPK)
    args = parser.parse_args()
    main(args.topk)