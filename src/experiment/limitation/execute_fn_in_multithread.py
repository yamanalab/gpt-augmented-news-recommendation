from typing import TypeVar, Callable
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import logging

# ジェネリック型変数Tを定義
T = TypeVar("T")
U = TypeVar("U")


def execute_fn_in_multithreading(items: list[T], func: Callable[[T], U], thread_num: int = 10) -> list[U]:
    results: list[U] = []
    task_queue = deque[T](items)
    while len(task_queue):
        logging.info(f"task_queue: {task_queue}")
        items_to_process: list[T] = []
        while len(task_queue):
            items_to_process.append(task_queue.pop())

        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            future_to_category_map = {
                executor.submit(func, item): item for item in items_to_process
            }  # ref: https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example

            for future in as_completed(future_to_category_map):
                category = future_to_category_map[future]

                try:
                    data = future.result()
                    results.append(data)

                except Exception as e:
                    logging.error("%r generated an exception: %s" % (category, e))
                    # raise e
                    task_queue.appendleft(category)  # append task_queue for retry.

    return results
