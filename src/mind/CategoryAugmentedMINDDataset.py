import json
import random
from typing import Callable

import numpy as np
import polars as pl
import torch
from const.path import (
    GPT35_GENERATED_CATEGORY_DESCRIPTION_JSON,
    GPT4_GENERATED_CATEGORY_DESCRIPTION_JSON,
    LLAMA2_13B_GENERATED_CATEGORY_DESCRIPTION_JSON,
)
from torch.utils.data import Dataset
from utils.list import uniq

from const.mind import EMPTY_IMPRESSION_IDX, EMPTY_NEWS_ID, MIND_SMALL_SUBCATEGORIES
import enum


class AugmentationMethodType(enum.StrEnum):
    NONE = "none"  # without augmentation
    GPT4 = "gpt4"
    GPT35 = "gpt35"
    LLAMA2_13B = "llama2_13b"
    TEMPLATE_BASED = "template_based"


AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP = {
    AugmentationMethodType.GPT4: GPT4_GENERATED_CATEGORY_DESCRIPTION_JSON,
    AugmentationMethodType.GPT35: GPT35_GENERATED_CATEGORY_DESCRIPTION_JSON,
    AugmentationMethodType.LLAMA2_13B: LLAMA2_13B_GENERATED_CATEGORY_DESCRIPTION_JSON,
}


class CategoryAugmentedMINDTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: dict[str, int],
        batch_transform_texts: Callable[[list[str | list[str]]], torch.Tensor],
        npratio: int,
        history_size: int,
        # TODO: add arguement: category_type: "category" | "subcategory" | "category_and_subcategory",
        augmentation_method: AugmentationMethodType,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str | list[str]]], torch.Tensor] = batch_transform_texts
        self.npratio: int = npratio
        self.history_size: int = history_size
        self.device: torch.device = device

        self.behavior_df = self.behavior_df.with_columns(
            [
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1])
                .alias("clicked_idxes"),
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0])
                .alias("non_clicked_idxes"),
            ]
        )

        if augmentation_method in [
            AugmentationMethodType.GPT35,
            AugmentationMethodType.GPT4,
            AugmentationMethodType.LLAMA2_13B,
        ]:
            with open(AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP[augmentation_method], "r") as f:
                subcategory_descriptions: list[dict[str, str]] = json.loads(f.read())
            self.__subcategory_name_to_description_map = {
                item["category"]: item["description"] for item in subcategory_descriptions
            }
        elif augmentation_method == AugmentationMethodType.TEMPLATE_BASED:
            self.__subcategory_name_to_description_map = {
                item: f"The category is {item}" for item in MIND_SMALL_SUBCATEGORIES
            }
        else:
            raise Exception(f"Unknown augmentation method: {augmentation_method}")

        self.__news_id_to_news_map: dict[str, dict[str, str]] = {
            self.news_df[i]["news_id"].item(): {
                "title": self.news_df[i]["title"].item(),
                "category_description": self.__subcategory_name_to_description_map[
                    self.news_df[i]["subcategory"].item()
                ],
            }
            for i in range(len(self.news_df))
        }
        self.__news_id_to_news_map[EMPTY_NEWS_ID] = {"title": "", "category_description": ""}
        self.__user_ids_to_idx_map = user_ids_to_idx_map

    def __getitem__(self, behavior_idx: int) -> dict:  # TODO: 一行あたりにpositiveが複数存在することも考慮した
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: labels
            torch.Tensor: user_id (idx)
        """
        # Extract Values
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )  # TODO: Consider Remove if "history" is None
        poss_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )  # NOTE: EMPTY_IMPRESSION_IDX = -1なので最後尾に追加する。

        poss_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )

        # Sampling Positive(clicked) & Negative(non-clicked) Sample
        sample_poss_idxes, sample_neg_idxes = (
            random.sample(poss_idxes, 1),
            self.__sampling_negative(neg_idxes, self.npratio),
        )

        sample_impression_idxes = sample_poss_idxes + sample_neg_idxes
        random.shuffle(sample_impression_idxes)

        sample_impressions = impressions[sample_impression_idxes]

        # Extract candidate_news & history_news based on sample idxes
        candidate_news_ids = [imp_item["news_id"] for imp_item in sample_impressions]
        labels = [imp_item["clicked"] for imp_item in sample_impressions]
        history_news_ids = history[: self.history_size]  # TODO: diverse
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

        # News ID to News Title
        candidate_news_list: list[str | list[str]] = [
            [self.__news_id_to_news_map[news_id]["title"], self.__news_id_to_news_map[news_id]["category_description"]]
            for news_id in candidate_news_ids
        ]
        history_news_list: list[str | list[str]] = [
            [self.__news_id_to_news_map[news_id]["title"], self.__news_id_to_news_map[news_id]["category_description"]]
            for news_id in history_news_ids
        ]

        # Convert to Tensor
        candidate_news_tensor, history_news_tensor = (
            self.batch_transform_texts(candidate_news_list),
            self.batch_transform_texts(history_news_list),
        )
        labels_tensor = torch.Tensor(labels).argmax()

        # user_id
        user_id = self.__user_ids_to_idx_map[behavior_item["user_id"].to_list()[0]]

        # ref: NRMS.forward in src/recommendation/nrms/NRMS.py
        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)

    def __sampling_negative(self, neg_idxes: list[int], npratio: int) -> list[int]:
        if len(neg_idxes) < npratio:
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))

        return random.sample(neg_idxes, self.npratio)

    def get_user_num(self, only_train: bool = True) -> int:
        if only_train:
            return len(uniq(self.behavior_df["user_id"].to_list()))
        return len(self.__user_ids_to_idx_map)


class CategoryAugmentedMINDValDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: dict[str, int],
        batch_transform_texts: Callable[[list[str | list[str]]], torch.Tensor],
        history_size: int,
        augmentation_method: AugmentationMethodType,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str | list[str]]], torch.Tensor] = batch_transform_texts
        self.history_size: int = history_size
        self.device: torch.device = device

        if augmentation_method in [
            AugmentationMethodType.GPT35,
            AugmentationMethodType.GPT4,
            AugmentationMethodType.LLAMA2_13B,
        ]:
            with open(AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP[augmentation_method], "r") as f:
                subcategory_descriptions: list[dict[str, str]] = json.loads(f.read())
            self.__subcategory_name_to_description_map = {
                item["category"]: item["description"] for item in subcategory_descriptions
            }
        elif augmentation_method == AugmentationMethodType.TEMPLATE_BASED:
            self.__subcategory_name_to_description_map = {
                item: f"The category is {item}" for item in MIND_SMALL_SUBCATEGORIES
            }
        else:
            raise Exception(f"Unknown augmentation method: {augmentation_method}")

        self.__news_id_to_news_map: dict[str, dict[str, str]] = {
            self.news_df[i]["news_id"].item(): {
                "title": self.news_df[i]["title"].item(),
                "category_description": self.__subcategory_name_to_description_map[
                    self.news_df[i]["subcategory"].item()
                ],
            }
            for i in range(len(self.news_df))
        }
        self.__news_id_to_news_map[EMPTY_NEWS_ID] = {"title": "", "category_description": ""}
        self.__user_ids_to_idx_map = user_ids_to_idx_map

    def __getitem__(self, behavior_idx: int) -> dict:  # TODO: 一行あたりにpositiveが複数存在することも考慮した
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: one-hot labels
        """
        # Extract Values
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )  # TODO: Consider Remove if "history" is None
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )  # NOTE: EMPTY_IMPRESSION_IDX = -1なので最後尾に追加する。

        # Extract candidate_news & history_news based on sample idxes
        candidate_news_ids = [imp_item["news_id"] for imp_item in impressions]
        labels = [imp_item["clicked"] for imp_item in impressions]
        history_news_ids = history[: self.history_size]  # TODO: diverse
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

        # News ID to News Title
        candidate_news_list: list[str | list[str]] = [
            [self.__news_id_to_news_map[news_id]["title"], self.__news_id_to_news_map[news_id]["category_description"]]
            for news_id in candidate_news_ids
        ]
        history_news_list: list[str | list[str]] = [
            [self.__news_id_to_news_map[news_id]["title"], self.__news_id_to_news_map[news_id]["category_description"]]
            for news_id in history_news_ids
        ]

        # Convert to Tensor
        candidate_news_tensor, history_news_tensor = (
            self.batch_transform_texts(candidate_news_list),
            self.batch_transform_texts(history_news_list),
        )
        one_hot_label_tensor = torch.Tensor(labels)

        # user_id
        user_id = self.__user_ids_to_idx_map[behavior_item["user_id"].to_list()[0]]

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            "target": one_hot_label_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)


if __name__ == "__main__":
    # from torch.utils.data import DataLoader
    # from transformers import AutoTokenizer

    from const.path import MIND_SMALL_VAL_DATASET_DIR
    from utils.random_seed import set_random_seed

    from src.mind.dataframe import read_news_df

    set_random_seed(42)

    news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # # logging.info()
    # def transform(texts: list[list[str] | str]) -> torch.Tensor:
    #     return tokenizer(texts, return_tensors="pt", max_length=64, padding="max_length", truncation=True)["input_ids"]

    # logging.info("Load Data")
    # behavior_df, news_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv"), read_news_df(
    #     MIND_SMALL_VAL_DATASET_DIR / "news.tsv"
    # )

    # logging.info("Init MINDTrainDataset")
    # train_dataset = CategoryAugmentedMINDTrainDataset(
    #     behavior_df, news_df, batch_transform_texts=transform, npratio=4, history_size=20
    # )
    # train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # logging.info("Start Iteration")
    # for batch in train_dataloader:
    #     logging.info(f"{batch}")
    #     logging.info(batch["candidate_news"].size())  # batch_size x (1 + npratio) x max_length
    #     break
