import json

import tiktoken

from const.path import MIND_GENERATED_DATASET_DIR
from utils.logger import logging

if __name__ == "__main__":
    with open(MIND_GENERATED_DATASET_DIR / "category_description_gpt4.json", "r") as f:
        s = f.read()
    generated_category_description_list = json.loads(s)

    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_string(string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(encoding.encode(string))
        return num_tokens

    category_count = len(generated_category_description_list)
    max_description_length = -1
    min_description_length = float("inf")
    avg_description_length = 0.0

    for category_description in generated_category_description_list:
        category, description = category_description["category"], category_description["description"]
        token_num = num_tokens_from_string(description)

        max_description_length = max(token_num, max_description_length)
        min_description_length = min(token_num, min_description_length)
        avg_description_length += token_num

    avg_description_length /= category_count

    logging.info(
        {
            "category_count": category_count,
            "max_description_length": max_description_length,
            "min_description_length": min_description_length,
            "avg_description_length": avg_description_length,
        }
    )
