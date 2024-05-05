import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import polars as pl

from const.path import MIND_GENERATED_DATASET_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR
from experiment.generation.prompt import SYSTEM_PROMPT
from mind.dataframe import read_news_df
from utils.logger import logging

if __name__ == "__main__":
    # Read dataframe
    train_news_df: pl.DataFrame = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    val_news_df: pl.DataFrame = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")

    # TODO: mainとsubの組み合わせで生成もやってみる。
    # NOTE: categoryが17種類(=len(unique_category))と思いの他、少ないので一旦subcategoryでやる

    train_unique_category: list[str] = train_news_df["category"].unique().to_list()
    train_unique_subcategory: list[str] = train_news_df["subcategory"].unique().to_list()
    val_unique_category: list[str] = val_news_df["category"].unique().to_list()
    val_unique_subcategory: list[str] = val_news_df["subcategory"].unique().to_list()

    unique_category = list(set(train_unique_category) | set(val_unique_category))
    unique_subcategory = list(set(train_unique_subcategory) | set(val_unique_subcategory))

    client = openai.OpenAI()

    def generate_description_for_category(category: str) -> dict[str, str | None]:
        # Set Parameters
        model = "gpt-4-turbo"
        temperature = 0.03

        # Generate Description
        user_message = f"The news category is {category}"
        messages: list[openai.ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        logging.info(f"generating {category}'s description ... ")
        res = client.chat.completions.create(model=model, messages=messages, temperature=temperature, timeout=10)
        logging.info(f"generated {category}'s description")

        return {"category": category, "description": res.choices[0].message.content}

    category_descriptions: list[dict[str, str | None]] = []
    task_queue = deque[str](unique_subcategory)
    while len(task_queue):
        logging.info(f"task_queue: {task_queue}")
        categories_to_process: list[str] = []
        while len(task_queue):
            categories_to_process.append(task_queue.pop())

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_category_map = {
                executor.submit(generate_description_for_category, category): category
                for category in categories_to_process
            }  # ref: https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example

            for future in as_completed(future_to_category_map):
                category = future_to_category_map[future]

                try:
                    data = future.result()
                    category_descriptions.append(data)

                except Exception as e:
                    logging.error("%r generated an exception: %s" % (category, e))
                    task_queue.appendleft(category)  # append task_queue for retry.

    # export data
    json_output_path = MIND_GENERATED_DATASET_DIR / "mind_small_subcategory_descriptions.json"
    MIND_GENERATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_output_path, "w") as file:
        json.dump(category_descriptions, file, indent=4)
