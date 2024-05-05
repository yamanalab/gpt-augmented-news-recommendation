import openai
from experiment.limitation.prompt import SYSTEM_PROMPT_FOR_ABSTRACTION_SCORING
from experiment.limitation.openai_tools import OPENAI_TOOLS
from experiment.limitation.execute_fn_in_multithread import execute_fn_in_multithreading
import json
from mind.CategoryAugmentedMINDDataset import (
    AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP,
    AugmentationMethodType,
)


def scoring_category_description() -> None:
    openai_client = openai.OpenAI()

    def _scoring_category_description(d: dict[str, str]) -> dict[str, str]:
        category, description = d["category"], d["description"]
        user_message = f"Category: {category}"
        messages: list[openai.ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT_FOR_ABSTRACTION_SCORING},
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

    with open(AUGMENTATION_METHOD_TO_CATEGORY_DESCRIPTION_PATH_MAP[AugmentationMethodType.GPT4], "r") as f:
        subcategory_descriptions: list[dict[str, str]] = json.loads(f.read())

    results = execute_fn_in_multithreading(subcategory_descriptions, _scoring_category_description, 10)
    with open("./description_score.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    scoring_category_description()
