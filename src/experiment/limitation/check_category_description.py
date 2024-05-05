"""
ニュース記事とカテゴリの説明文をGPT-4に入力して、カテゴリの説明文が正しいかどうか？を判定してもらう。
"""

from mind.dataframe import read_news_df
from const.path import MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR
from experiment.limitation.openai_tools import OPENAI_TOOLS
from mind.CategoryAugmentedMINDDataset import (
    AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP,
    AugmentationMethodType,
)
from experiment.limitation.prompt import SYSTEM_PROMPT_FOR_ACCURATION_SCORING
from experiment.limitation.execute_fn_in_multithread import execute_fn_in_multithreading
import polars as pl
import json
import openai

if __name__ == "__main__":
    # Read dataframe
    train_news_df: pl.DataFrame = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    val_news_df: pl.DataFrame = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")

    news_df = pl.concat([train_news_df, val_news_df], how="vertical")
    grouped_news_df = news_df.groupby("subcategory").head(3)

    with open(AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP[AugmentationMethodType.GPT4], "r") as f:
        subcategory_descriptions: list[dict[str, str]] = json.loads(f.read())

    subcategory_descriptions_map = {item["category"]: item["description"] for item in subcategory_descriptions}

    news_items_array = []
    for subcategory in grouped_news_df["subcategory"].unique():
        news_items = grouped_news_df.filter(pl.col("subcategory") == subcategory)["title"].to_list()
        news_items_array.append(
            {
                "subcategory": subcategory,
                "news_titles": news_items,
                "description": subcategory_descriptions_map[subcategory],
            }
        )

    openai_client = openai.OpenAI()

    def _scoring_category_description(d: dict[str, str]) -> dict[str, str]:
        category, description, news_titles = d["subcategory"], d["description"], d["news_titles"]
        user_message = f"Category: {category}\nCategory Description: {description}\nNews Titles: {','.join([news for news in news_titles])}"
        messages: list[openai.ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_ACCURATION_SCORING},
            {"role": "user", "content": user_message},
        ]
        res = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            tools=OPENAI_TOOLS,
            temperature=0.0,
            timeout=10,
            tool_choice={"type": "function", "function": {"name": "send_score"}},
        )
        if not res.choices[0].message.tool_calls:
            raise Exception("Tool is not called")
        tool_call = res.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)

        return function_args | d

    results = execute_fn_in_multithreading(news_items_array, _scoring_category_description, 10)
    with open("./description_score_accuration.json", "w") as file:
        json.dump(results, file, indent=4)

    # Read Category Description
