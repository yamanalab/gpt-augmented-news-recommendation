from typing import TypeVar, Callable

# ジェネリック型変数Tを定義
T = TypeVar("T")


def all_match(lst: list[T], condition: Callable[[T], bool]) -> bool:
    return all(condition(item) for item in lst)


def uniq(lst: list[T]) -> list[T]:
    return list(set(lst))
