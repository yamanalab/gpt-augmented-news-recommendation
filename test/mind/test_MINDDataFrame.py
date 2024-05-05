from src.const.path import MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR
from src.mind.dataframe import create_user_ids_to_idx_map, read_behavior_df, read_news_df
from src.utils.list import all_match
from src.const.mind import UNKNOWN_USER_IDX


def test_read_news_df() -> None:
    small_val_news_tsv_path = MIND_SMALL_VAL_DATASET_DIR / "news.tsv"
    news_df = read_news_df(small_val_news_tsv_path)
    assert news_df.columns == ["news_id", "category", "subcategory", "title", "abstract", "url"]
    assert len(news_df) == 42416


def test_read_news_df_with_entities() -> None:
    small_val_news_tsv_path = MIND_SMALL_VAL_DATASET_DIR / "news.tsv"
    news_df = read_news_df(small_val_news_tsv_path, has_entities=True)
    assert news_df.columns == [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    assert len(news_df) == 42416


def test_read_behavior_df() -> None:
    small_val_behavior_tsv_path = MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv"
    behavior_df = read_behavior_df(small_val_behavior_tsv_path)
    assert behavior_df.columns == ["impression_id", "user_id", "time", "history", "impressions"]
    assert len(behavior_df) == 73152


def test_create_user_ids_to_idx_map() -> None:
    small_train_behavior_tsv_path, small_val_behavior_tsv_path = (
        MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv",
        MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv",
    )
    val_behavior_df = read_behavior_df(small_val_behavior_tsv_path)
    train_behavior_df = read_behavior_df(small_train_behavior_tsv_path)

    user_ids_to_idx_map = create_user_ids_to_idx_map(train_behavior_df, val_behavior_df)

    user_ids_in_train_set = set(train_behavior_df["user_id"].to_list())
    user_ids_in_val_set = set(val_behavior_df["user_id"].to_list())
    assert len(user_ids_to_idx_map) == len(user_ids_in_train_set | user_ids_in_val_set)

    user_ids_only_in_val = list(user_ids_in_val_set - user_ids_in_train_set)

    def condition(item):
        return user_ids_to_idx_map.get(item) == UNKNOWN_USER_IDX

    assert all_match(user_ids_only_in_val, condition)
