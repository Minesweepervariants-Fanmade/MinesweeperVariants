from importlib.metadata import version

__version__ = version("minesweepervariants")

def tuple_version() -> tuple[int, int, int]:
    ver = __version__.split(".")
    match len(ver):
        case 1:
            return (int(ver[0]), 0, 0)
        case 2:
            return (int(ver[0]), int(ver[1]), 0)
        case 3:
            return (int(ver[0]), int(ver[1]), int(ver[2]))
        case _:
            return (0, 0, 0)


def load():
    from .scripts.generate_puzzle import main as puzzle
    from .scripts.generate_game import main as puzzle_query
    from .scripts.generate_test import main as test
    return puzzle, puzzle_query, test
