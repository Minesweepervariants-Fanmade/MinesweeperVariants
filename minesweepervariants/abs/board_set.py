
from .boardv2 import AbstractBoard


class BoardSet(dict['str', 'AbstractBoard']):
    label: str
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = kwargs.get("label", "unknown")

    def __str__(self) -> str:
        return f"{self.label}({', '.join(self.keys())})"

if __name__ == "__main__":
    from .boardv2 import Board
    b = BoardSet(label="test")
    b["a"] = Board("a")
    b["b"] = Board("b")
    print(b)