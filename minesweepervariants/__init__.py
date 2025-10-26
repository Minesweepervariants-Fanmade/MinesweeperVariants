from importlib.metadata import version

try:
    __version__ = version("minesweepervariants")
except Exception:
    __version__ = "0.0.0"

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



from .scripts.generate_puzzle import main as puzzle
from .scripts.generate_game import main as puzzle_query
from .scripts.generate_test import main as test