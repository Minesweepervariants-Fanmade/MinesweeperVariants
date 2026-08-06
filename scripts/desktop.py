r"""
SELF: ('A1', '雷')
TARGET: OPERATIONS.DIG
Current Working Directory: Z:\Temp\扫雷-1S
Module Directory: D:\Coding\MinesweeperVariants
Position: A1, Value: 雷, File: Z:\Temp\扫雷-1S\A1 (雷).lnk
Position: A2, Value: 雷, File: Z:\Temp\扫雷-1S\A2 (雷).lnk
Position Map: {'A1': WindowsPath('Z:/Temp/扫雷-1S/A1 (雷).lnk'), 'A2': WindowsPath('Z:/Temp/扫雷-1S/A2 (雷).lnk')}
Board: {'A1': '雷', 'A2': '雷'}
Press Enter to exit...


"""

import contextlib
import io
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path
import win32process
import win32com.client

from ortools.sat.python import cp_model

from minesweepervariants.size import Size
from minesweepervariants.board import MASTER_BOARD_KEY, Board
from minesweepervariants.utils.impl_obj import MINES_TAG, VALUE_QUESS, POSITION_TAG
from minesweepervariants.utils.value_template import SingleIntValue
from minesweepervariants.utils.timer import timer
from minesweepervariants.impl.impl_obj import get_rule, get_value
from minesweepervariants.impl.summon.summon import Summon
from minesweepervariants.impl.summon.solver import board_create_constraints, get_solver
from minesweepervariants.impl.summon.game import GameSession, PUZZLE, ValueAsterisk


class OPERATIONS(Enum):
    DIG = "DIG"
    FLAG = "FLAG"
    CLEAR = "CLEAR"


# 扫描值 -> 类型映射
MINE_VALUES = ("F", "爆炸")          # 旗帜/已爆炸 视为雷(已翻开); 雷标记不合法
UNKNOWN_VALUES = ("", " ", "?")     # 未知格
_PREFIX_RE = re.compile(r"^\d+\.\s*")  # 排序前缀 "01. "
ICONS_PATH = Path(__file__).resolve().parent.parent / "minesweepervariants" / "assets" / "desktop_icons"


def strip_prefix(stem: str) -> str:
    return _PREFIX_RE.sub("", stem, count=1)


def stem_with_prefix(path: Path, body: str) -> str:
    m = _PREFIX_RE.match(path.stem)
    return m.group(0) + body if m else body


def value_icon(value: str) -> str:
    """根据格子显示值映射图标名(不含扩展名). 数字线索 -> 数字, F/爆炸 -> flag, 星标 -> star, 空格 -> transparent, 其它 -> question."""
    v = value.strip()
    if v in ("F", "爆炸"):
        return "flag"
    if v in ("", " "):
        return "transparent"
    if v in ("*", "星"):
        return "star"
    m = re.match(r"^(?:.+?-)?(\d+)$", v)
    if m:
        return m.group(1)
    return "question"


def set_icon(pos: str, target_path: Path, icon: str) -> None:
    """
    Sets the icon for a given target lnk file.
    """
    icon_path = ICONS_PATH / f"{icon}.ico"
    if not icon_path.exists():
        print(f"图标不存在, 跳过: {icon_path}")
        return

    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(target_path.absolute().as_posix())
    shortcut.IconLocation = icon_path.absolute().as_posix()
    shortcut.save()


def set_file_value(pos: str, target_path: Path, value: str) -> None:
    target_path.rename(target_path.with_stem(stem_with_prefix(target_path, f"{pos} ({value})")))


def parse_lnk_name(lnk_path: Path | None) -> tuple[str, str] | OPERATIONS:
    """
    Parses the name of a lnk file and returns it without the extension.
    """
    if lnk_path is None:
        return OPERATIONS.CLEAR
    stem = strip_prefix(lnk_path.stem)

    if stem.lower() == "dig":
        return OPERATIONS.DIG
    if stem.lower() == "flag":
        return OPERATIONS.FLAG

    pat = r"^(.+?) \((.+?)\)?$"
    result = re.match(pat, stem)
    if result:
        position, value = result.groups()
        return position.upper(), value
    else:
        raise ValueError(f"Invalid lnk name: {lnk_path.name}")


def iter_lnk_files(root: Path) -> list[Path]:
    """
    Recursively iterates over all .lnk files under the given root path.
    """
    return list(root.rglob("*.lnk"))


def create_board(root: Path):
    pos_map: dict[str, Path] = {}
    board: dict[str, str] = {}
    for file in iter_lnk_files(root):
        parsed = parse_lnk_name(file)
        if isinstance(parsed, OPERATIONS):
            continue
        position, value = parsed
        print(f"Position: {position}, Value: {value}, File: {file}")
        pos_map[position] = file
        board[position] = value
    return pos_map, board


def parse_rule_set(folder_name: str) -> list[str]:
    """
    从游戏目录名解析规则, 例如 "扫雷-1S" -> ["1S"].
    """
    name = folder_name
    if name.startswith("扫雷-"):
        name = name[len("扫雷-"):]
    if not name:
        return ["V"]
    parts = [p for p in name.split("-") if p]
    resolved = [p for p in parts if _rule_exists(p)]
    return resolved or ["V"]


def _rule_exists(rule_id: str) -> bool:
    try:
        get_rule(rule_id)
        return True
    except ValueError:
        return False


def parse_spec(spec: str):
    """
    解析 "m n V" / "m 1S 1X" 生成规格 -> (Size, 规则列表).
    开头 1~2 个整数为尺寸, 其余为规则.
    """
    tokens = spec.split()
    sizes = []
    i = 0
    while i < len(tokens) and tokens[i].isdigit() and len(sizes) < 2:
        sizes.append(int(tokens[i]))
        i += 1
    rules = tokens[i:]
    if not sizes or not rules:
        return None
    if any(not _rule_exists(r) for r in rules):
        return None
    size = Size(sizes[0], sizes[0]) if len(sizes) == 1 else Size(sizes[0], sizes[1])
    return size, rules


def make_shortcut(lnk_path: Path, target: Path, args: str, workdir: str = "") -> None:
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(str(lnk_path))
    shortcut.TargetPath = str(target)
    shortcut.Arguments = args
    shortcut.WorkingDirectory = workdir
    shortcut.save()


def generate_puzzle(size: Size, rules: list[str], base_dir: Path) -> None:
    """生成唯一解题板并写入 {base_dir}/扫雷-{规则}/ 下的 lnk 文件."""
    folder = base_dir / ("扫雷-" + "-".join(rules))
    print(f"生成题板: size={size.cols}x{size.rows} rules={rules}")
    print(f"输出目录: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("*.lnk"):
        f.unlink()

    puzzle = None
    for _ in range(5):
        try:
            summon = Summon(size=size, total=-1, rules=rules, drop_r=True)
            with contextlib.redirect_stdout(io.StringIO()):
                puzzle = summon.create_puzzle()
            if puzzle is not None:
                break
        except Exception:
            puzzle = None
    if puzzle is None:
        print("生成失败: 多次尝试未生成有效题板")
        return

    script = Path(__file__).resolve()
    python = Path(sys.executable)
    pythonw = python if python.name.lower().startswith("pythonw") else python.with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = python
    cells = sorted(
        (pos, t) for pos, t in puzzle(mode="type", key=MASTER_BOARD_KEY, special='raw')
    )
    width = len(str(len(cells) + 2))
    for i, (pos, t) in enumerate(cells, start=1):
        label = repr(pos)
        if t == "C" and puzzle[pos] is not VALUE_QUESS:
            obj = puzzle[pos]
            num = getattr(getattr(obj, "value", None), "value", None)
            value = f"{obj.id}-{num}" if isinstance(num, int) else repr(obj)
        else:
            value = " "
        cell = folder / f"{i:0{width}d}. {label} ({value}).lnk"
        make_shortcut(cell, pythonw, f'"{script}"')
        set_icon(label, cell, value_icon(value))
    dig_lnk = folder / f"{len(cells) + 1:0{width}d}. Dig.lnk"
    make_shortcut(dig_lnk, pythonw, f'"{script}"')
    set_icon("Dig", dig_lnk, "shovel")
    flag_lnk = folder / f"{len(cells) + 2:0{width}d}. Flag.lnk"
    make_shortcut(flag_lnk, pythonw, f'"{script}"')
    set_icon("Flag", flag_lnk, "flag")
    print(f"完成: {len(list(folder.glob('*.lnk')))} 个 lnk 文件")


def _safe_get_value(pos, rule_id: str, data):
    try:
        return get_value(pos, rule_id, data)
    except Exception:
        return None


def parse_value(pos, value: str, clue_rule_id: str):
    """
    将文件名中的值字符串解析为线索对象.
    支持 SingleIntValue 数字线索(纯数字或 "规则-数字" 标签格式, 如 V-1 / 1X-1),
    以及 "?"/空格 未知格, F/爆炸 标记. 雷标记不合法, 按未知格处理.
    """
    v = value.strip() if value else ""
    if v in UNKNOWN_VALUES:
        return None
    if v in MINE_VALUES:
        return MINES_TAG
    if v == "*" or v == "星":
        return ValueAsterisk(POSITION_TAG)
    obj = None
    if v.isdigit():
        obj = _safe_get_value(pos, clue_rule_id, SingleIntValue(int(v)).json())
    else:
        m = re.match(r"^(.+)-(\d+)$", v)
        if m:
            obj = _safe_get_value(pos, m.group(1), SingleIntValue(int(m.group(2))).json())
    if obj is not None:
        return obj
    print(f"未识别的格子值: {value!r}, 按未知格处理")
    return None


def solve_answer(game, summon, size: Size):
    """求解当前题板的唯一解, 重建答案板(雷位置), 不依赖任何雷标记文件."""
    board = game.clone()
    all_rules = summon.mines_rules.rules[:] + [summon.clue_rule, summon.mines_clue_rule]
    model, switch, _ = board_create_constraints(board, all_rules, drop_r=True)
    for pos, var in board("C", mode="variable", special='raw'):
        model.add(var == 0)
    for pos, var in board("F", mode="variable", special='raw'):
        model.add(var == 1)
    model.add_bool_and(switch.get_all_vars())

    solver = get_solver(False)
    if timer(solver.Solve)(model) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    answer = Board()
    answer.generate_board(MASTER_BOARD_KEY, size)
    for pos, var in board(mode="variable", special='raw'):
        if solver.Value(var):
            answer.set_value(pos, MINES_TAG)
    summon.clue_rule.fill(answer)
    return answer


def build_boards(cells: dict[str, str], rule_names: list[str]):
    """
    根据扫描出的题板构建 game_board(玩家视角) 与 answer_board(求解器唯一解).
    """
    cols = rows = 0
    parsed: dict[str, tuple[int, int]] = {}
    for label in cells:
        m = re.match(r"^([A-Za-z]+)(\d+)$", label)
        if not m:
            continue
        col = 0
        for ch in m.group(1).upper():
            col = col * 26 + (ord(ch) - 64)
        col -= 1
        row = int(m.group(2)) - 1
        parsed[label] = (row, col)
        cols = max(cols, col + 1)
        rows = max(rows, row + 1)

    size = Size(cols, rows)
    summon = Summon(size=size, total=-1, rules=rule_names, drop_r=True)
    clue_rule_id = summon.clue_rule.id

    game = Board()
    game.generate_board(MASTER_BOARD_KEY, size)
    for label, v in cells.items():
        if label not in parsed:
            continue
        row, col = parsed[label]
        pos = game.get_pos(row, col)
        obj = parse_value(pos, v, clue_rule_id)
        if obj is not None:
            game.set_value(pos, obj)

    answer = solve_answer(game, summon, size)

    label_pos = {}
    for label, (row, col) in parsed.items():
        label_pos[label] = game.get_pos(row, col)
    return answer, game, summon, label_pos


def reveal_cell(pos, path: Path) -> None:
    """成功挖开: 仅将该格标记为 星 (不连锁翻开, 不展示线索数, 不给额外信息). Windows 文件名禁止 *, 故用 星."""
    label = repr(pos)
    new_path = path.with_stem(stem_with_prefix(path, f"{label} (星)"))
    if path == new_path:
        return
    set_file_value(label, path, "星")
    set_icon(label, new_path, "star")


def wait_exit() -> None:
    """结尾等待退出: 仅控制台 python 下等待按键; pythonw 无控制台, 直接退出."""
    if Path(sys.executable).name.lower().startswith("pythonw"):
        return
    try:
        input("Press Enter to exit...")
    except EOFError:
        pass


def popup(message: str) -> None:
    """弹出消息框(不阻塞, 后台 PowerShell MessageBox)."""
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        f"[System.Windows.Forms.MessageBox]::Show('{message}', '扫雷', 'OK', 'Warning')"
    )
    subprocess.Popen(
        ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def explode(pos_label: str, message: str) -> None:
    """踩雷/标错雷: 不修改格子文件, 仅打印并弹窗提示."""
    print(f"{pos_label}: {message}")
    popup(f"{pos_label} {message}")


def dig(gs: GameSession, pos_label: str, path: Path, pos) -> None:
    """挖格: 未翻开格 → 单线索验证后标星; 已翻开线索格 → 单线索推理(chord)其邻域."""
    t = gs.board.get_type(pos, special='raw')
    if t == "F":
        print(f"{pos_label} 已是雷/旗帜, 无需操作")
        return
    if t != "N":
        chord = gs.chord_clue(pos)
        if not chord:
            print(f"{pos_label} 已翻开, 无新推理")
            return
        print(f"{pos_label} 单线索推理: {[repr(p) for p in chord]}")
        for p in chord:
            p_label = repr(p)
            p_path = POS_MAP.get(p_label)
            if p_path is None:
                continue
            if gs.answer_board.get_type(p, special='raw') == "F":
                new_path = p_path.with_stem(stem_with_prefix(p_path, f"{p_label} (F)"))
                if p_path != new_path:
                    set_file_value(p_label, p_path, "F")
                    set_icon(p_label, new_path, "flag")
            else:
                reveal_cell(p, p_path)
        return

    if gs.unbelievable(pos, 0) is not None:
        explode(pos_label, "你踩雷了!")
        return

    gs.apply(pos, 0)
    print(f"{pos_label} 合法翻开")
    reveal_cell(pos, path)


def flag(gs: GameSession, pos_label: str, path: Path, pos) -> None:
    """标雷: 按游戏单线索推理, 仅当格子可推为雷时才合法, 否则爆炸."""
    if gs.board.get_type(pos, special='raw') == "F":
        print(f"{pos_label} 已标记")
        return
    if gs.board.get_type(pos, special='raw') != "N":
        print(f"{pos_label} 已翻开, 无需操作")
        return

    if gs.unbelievable(pos, 1) is not None:
        explode(pos_label, "你标记了一个错误的雷!")
        return

    gs.apply(pos, 1)
    new_path = path.with_stem(stem_with_prefix(path, f"{pos_label} (F)"))
    if path != new_path:
        set_file_value(pos_label, path, "F")
        set_icon(pos_label, new_path, "flag")


def main():
    print("SELF:", SELF)
    print("TARGET:", TARGET)
    print("Current Working Directory:", CWD)
    print("Module Directory:", MODULE_DIR)
    global POS_MAP
    POS_MAP, board = create_board(CWD)
    print("Position Map:", POS_MAP)
    print("Board:", board)

    if not board:
        print("空题板, 无格子")
        return

    rule_names = parse_rule_set(CWD.name)
    print("Rules:", rule_names)

    answer, game, summon, label_pos = build_boards(board, rule_names)
    if answer is None:
        print("题板无解(可能线索与规则矛盾), 无法继续")
        return
    gs = GameSession(summon=summon, mode=PUZZLE, drop_r=True)
    gs.answer_board = answer
    gs.board = game

    if TARGET is OPERATIONS.CLEAR and isinstance(SELF, OPERATIONS):
        print("CLEAR: 仅展示当前题板")
        return

    if isinstance(SELF, OPERATIONS):
        print("SELF 不是格子, 忽略")
        return

    pos_label = SELF[0]
    if pos_label not in POS_MAP or pos_label not in label_pos:
        print(f"找不到格子: {pos_label}")
        return
    path = POS_MAP[pos_label]
    pos = label_pos[pos_label]

    if TARGET is OPERATIONS.FLAG:
        flag(gs, pos_label, path, pos)
    else:  # DIG 或目标为格子
        dig(gs, pos_label, path, pos)


if __name__ == "__main__":
    spec = None
    if len(sys.argv) >= 3:
        target_raw = sys.argv[1]
        self_raw = sys.argv[2]
        target_ok = bool(target_raw.strip()) and Path(target_raw).exists()
        if not target_ok:
            if self_raw:
                spec = parse_spec(strip_prefix(Path(self_raw).stem))
            if spec is None and target_raw:
                spec = parse_spec(strip_prefix(Path(target_raw).stem))
    elif len(sys.argv) == 2:
        spec = parse_spec(strip_prefix(Path(sys.argv[1]).stem))
    else:
        _title = win32process.GetStartupInfo().lpTitle
        if _title:
            spec = parse_spec(strip_prefix(Path(_title).stem))

    if spec is not None:
        size, rules = spec
        _title = win32process.GetStartupInfo().lpTitle
        base_dir = Path(_title).parent if _title else Path.cwd()
        generate_puzzle(size, rules, base_dir)
        wait_exit()
        raise SystemExit(0)

    if len(sys.argv) == 3:
        _self_path = Path(sys.argv[2])
    else:
        _title = win32process.GetStartupInfo().lpTitle
        _self_path = Path(_title) if _title else None
    SELF_PATH = _self_path

    if len(sys.argv) < 2:
        _target_path = None
    else:
        _target_path = Path(sys.argv[1])
    if _target_path is not None:
        try:
            if _target_path.resolve() == Path(__file__).resolve():
                _target_path = None
        except Exception:
            pass
    TARGET_PATH = _target_path

    if SELF_PATH is None:
        print("无法确定 SELF 路径(请通过第二个参数模拟lnk运行)")
        wait_exit()
        raise SystemExit(1)

    CWD = SELF_PATH.parent
    MODULE_DIR = Path(__file__).parent.parent

    POS_MAP: dict[str, Path] = {}

    SELF = parse_lnk_name(SELF_PATH)
    TARGET = parse_lnk_name(TARGET_PATH)

    main()
    wait_exit()
