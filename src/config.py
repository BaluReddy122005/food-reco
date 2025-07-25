from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Files
TRAIN_CUSTOMERS = DATA_RAW / "train_customers.csv"
TRAIN_LOCATIONS = DATA_RAW / "train_locations.csv"
ORDERS = DATA_RAW / "orders.csv"
VENDORS = DATA_RAW / "vendors.csv"
TEST_CUSTOMERS = DATA_RAW / "test_customers.csv"
TEST_LOCATIONS = DATA_RAW / "test_locations.csv"
SAMPLE_SUBMISSION = DATA_RAW / "SampleSubmission.csv"

# Hyper-parameters for baseline scoring
TOPK = 5
W_CUSTOMER_FREQ = 0.40
W_LOC_FREQ = 0.25
W_GLOBAL_POP = 0.20
W_DISTANCE = 0.15  # applied as (1 / (1 + distance))