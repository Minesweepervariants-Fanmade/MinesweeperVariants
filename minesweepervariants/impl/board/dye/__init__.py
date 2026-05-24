import pkgutil
import importlib
import pathlib

from minesweepervariants.abs.dye import AbstractDye

from .alias import aliases


# 动态递归导入当前目录所有模块和包
def _auto_import_modules():
    current_pkg = __name__
    current_path = pathlib.Path(__file__).parent

    for _finder, name, _ispkg in pkgutil.walk_packages([str(current_path)], prefix=current_pkg + "."):
        importlib.import_module(name)


_auto_import_modules()


def get_dye(name: str, alias: tuple[str, str] | None = None) -> AbstractDye | None:
    name = name[1:] if name.startswith("@") else name
    args = ""
    if ":" in name:
        index = name.index(":")
        name, args = name[:index], name[index + 1:]

    if name in aliases:
        _alias = aliases[name]
        name = _alias[2]
        args = _alias[3]
        return get_dye(f"{name}:{args}", _alias[0:2])

    for cls in AbstractDye.__subclasses__():
        if cls.name == name:
            dye = cls(args)
            if alias:
                dye.name = name
                dye.fullname = alias[0]
                dye.doc = alias[1]
            return dye

    raise ValueError(f"未知的染色规则[@{name[1:] if name.startswith('@') else name}]")


def get_all_dye():
    result = {}
    for cls in AbstractDye.__subclasses__():
        result[cls.name] = cls.fullname, cls.doc
    for name, (fullname, doc, _, _) in aliases.items():
        result[name] = fullname, doc
    return result
