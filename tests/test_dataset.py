from src.dataset import _limit_files


def test_limit_files_size():
    files = [
        f"img_{i}.png"
        for i in range(100)
    ]

    limited = _limit_files(
        files,
        limit=20,
        seed=42
    )

    assert len(limited) == 20


def test_limit_files_deterministic():
    files = [
        f"img_{i}.png"
        for i in range(100)
    ]

    first = _limit_files(
        files,
        limit = 20,
        seed = 42
    )

    second = _limit_files(
        files,
        limit = 20,
        seed = 42
    )

    assert first == second


def test_limit_files_different_seed():
    files = [
        f"img_{i}.png"
        for i in range(100)
    ]

    first = _limit_files(
        files,
        limit = 20,
        seed = 42
    )

    second = _limit_files(
        files,
        limit = 20,
        seed = 123
    )

    assert first != second
