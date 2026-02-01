"""CLI tools for DedeuceRL.

Keep imports lazy so `python -m dedeucerl.cli.<tool>` runs cleanly.
"""


def eval_main() -> None:
    from .eval import main

    main()


def generate_main() -> None:
    from .generate import main

    main()


def aggregate_main() -> None:
    from .aggregate import main

    main()


def selfcheck_main() -> None:
    from .selfcheck import main

    main()


def eval_parallel_main() -> None:
    from .eval_parallel import main

    main()


def train_main() -> None:
    from .train import main

    main()

__all__ = [
    "eval_main",
    "generate_main",
    "aggregate_main",
    "selfcheck_main",
    "eval_parallel_main",
    "train_main",
]
