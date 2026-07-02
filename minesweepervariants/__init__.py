from importlib.metadata import version

__all__ = [
    "puzzle",
    "puzzle_query",
    "test",
    "hint",
    "img",
    "game",
    "ocr",
    "tuple_version",
    "__version__",
]

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
from .scripts.hint import main as hint
from .scripts.img import main as img

from .scripts.game import main as game

try:
    from .scripts.ocr import main as ocr
except ImportError as e:
    from minesweepervariants.utils.tool import LOGGER
    LOGGER.warning("can not import ocr module, please install \"rapidocr-onnxruntime\", \"opencv-python\"")
    LOGGER.warning(e)
    ocr = None
