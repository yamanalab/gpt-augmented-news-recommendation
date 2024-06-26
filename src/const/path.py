import pathlib

PROJECT_ROOT = (pathlib.Path(__file__) / ".." / ".." / "..").resolve()

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
LOG_OUTPUT_DIR = OUTPUT_DIR / "log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DATASET_DIR = PROJECT_ROOT / "dataset"

MIND_DATASET_DIR = DATASET_DIR / "mind"

CACHE_DIR = PROJECT_ROOT / ".cache"


MIND_SMALL_DATASET_DIR = MIND_DATASET_DIR / "small"
MIND_SMALL_VAL_DATASET_DIR = MIND_SMALL_DATASET_DIR / "val"
MIND_SMALL_TRAIN_DATASET_DIR = MIND_SMALL_DATASET_DIR / "train"

MIND_LARGE_DATASET_DIR = MIND_DATASET_DIR / "large"
MIND_LARGE_VAL_DATASET_DIR = MIND_LARGE_DATASET_DIR / "val"
MIND_LARGE_TEST_DATASET_DIR = MIND_LARGE_DATASET_DIR / "test"
MIND_LARGE_TRAIN_DATASET_DIR = MIND_LARGE_DATASET_DIR / "train"


MIND_GENERATED_DATASET_DIR = DATASET_DIR / "generated"
MIND_GENERATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
GPT4_GENERATED_CATEGORY_DESCRIPTION_JSON = MIND_GENERATED_DATASET_DIR / "category_description_gpt4.json"
