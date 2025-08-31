from minesweepervariants.abs.board import AbstractBoard
from minesweepervariants.abs.rule import AbstractValue
from minesweepervariants.impl.summon.game import Mode, UMode, ValueAsterisk, MinesAsterisk
from minesweepervariants.utils.impl_obj import VALUE_QUESS, MINES_TAG


def init_component(data: dict, *, color: str = '--foreground-color', invalid: bool = False) -> dict:
    if data["type"] in ["col", "row"]:
        style = "display: flex; "
        if data["type"] == "col":
            # 垂直布局：主轴垂直方向，交叉轴居中
            style += "flex-direction: column;"
            # 子项高度平均分配（使用 flex-grow 实现）
            for child in data["children"]:
                child_style = init_component(child, color=color, invalid=invalid).get("style", "")
                if "flex-grow" not in child_style:
                    child_style += " flex-grow: 1;"
                child["style"] = child_style
        else:
            # 水平布局：主轴水平方向，交叉轴居中
            style += "flex-direction: row;"
            # 子项宽度平均分配（使用 flex-grow 实现）
            for child in data["children"]:
                child_style = init_component(child, color=color, invalid=invalid).get("style", "")
                if "flex-grow" not in child_style:
                    child_style += " flex-grow: 1;"
                child["style"] = child_style
        style += " align-items: center; justify-content: center; gap: 5%;"
        style += " width: 100%; height: 100%; flex-grow: 1;"
        return {
            "type": "container",
            "value": [init_component(i, color=color, invalid=invalid) for i in data["children"]],
            "style": style
        }

    elif data["type"] == "text":
        # 文本项：使用 flex 布局填充可用空间，添加溢出处理
        style = (f"color: rgb(from var({color}) r g b / "
                 f"{50 if invalid else 100}%); text-align: center;")
        style += " display: flex; justify-content: center; align-items: center;"
        style += " flex: 1; min-width: 0; max-width: 100%;"  # 关键：允许内容收缩
        style += " overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
        style += data["style"]

        return {
            "type": "text",
            "value": data.get("text", ""),
            "style": style
        }

    elif data["type"] == "image":
        path = data.get("image")
        # 图片项：保持比例，居中显示
        return {
            "type": "assets",
            "value": path,
            "style": f"fill: rgb(from var({color}) r g b / "
                     f"{50 if invalid else 100}%); flex: 1; min-width: 0;"
                     f"{data['style']}"
        }

    elif data["type"] == "placeholder":
        style = "flex-shrink: 0;"  # 防止占位符被压缩

        if "width" in data:
            # 行内容器中使用固定宽度
            style += f" width: {int(100 * data['width'])}%;"
        if "height" in data:
            # 列内容器中使用固定高度
            style += f" height: {int(100 * data['height'])}%;"

        return {
            "type": "container",
            "value": [],
            "style": style
        }
    elif data["type"] == "template":
        return data
    raise ValueError("Unknown component type")


def format_cell(_board, pos, label):
    obj = _board[pos]
    color = "--flag-color" if _board.get_type(pos) == "F" else "--foreground-color"
    invalid = False if obj is None else obj.invalid(_board)
    cell_data = init_component(
        obj.web_component(_board),
        color=color, invalid=invalid
    )
    cell_data["style"] += (f"color: rgb(from var({color}) r g b /"
                           f" {50 if invalid else 100}%);")
    if (
        _board.get_config(pos.board_key, "pos_label") and
        (
            _board.get_type(pos) == "F" or
            _board.get_type(pos) == "N"
        )
    ):
        cell_data = {
            "type": "container",
            "value": [
                cell_data,
                {
                    "type": "template",
                    "style": "",
                    "value": {
                        "name": "backgroundStr",
                        "value": _board.pos_label(pos)
                    },
                }
            ],
            "style": ""
        }
    VALUE = _board.get_config(pos.board_key, "VALUE")
    MINES = _board.get_config(pos.board_key, "MINES")
    obj: AbstractValue
    if obj in [VALUE, MINES, None]:
        overlayText = ""
    else:
        overlayText = obj.tag(_board).decode()
    # hightlight = [{
    #             "x": pos.x,
    #             "y": pos.y,
    #             "boardname": pos.board_key,
    #         }]
    hightlight = {pos.board_key: [[pos.x, pos.y]]}
    if obj is not None:
        if obj.high_light(_board) is not None:
            for h_pos in set(h_pos for h_pos in obj.high_light(_board) if _board.in_bounds(h_pos)):
                # hightlight.append({
                #     "x": h_pos.x,
                #     "y": h_pos.y,
                #     "boardname": h_pos.board_key,
                # })
                if h_pos.board_key not in hightlight:
                    hightlight[h_pos.board_key] = []
                hightlight[h_pos.board_key].append([h_pos.x, h_pos.y])
    cell_data = {
        "type": "" if obj is None else obj.type().decode("ascii"),
        "position": {
            "x": pos.x, "y": pos.y,
            "boardname": pos.board_key
        },
        "component": cell_data,
        "highlight": hightlight,
        "clickable": True,
        "overlayText": overlayText if label else ""
    }
    # import json
    # json_str = json.dumps(cell_data, separators=(",", ":"))
    # print(json_str)
    return cell_data


def format_board(_board: AbstractBoard):
    if _board is None:
        return

    boards = {}
    cells = []
    count = 0
    for key in _board.get_board_keys():
        dye_list = [
            [_board.get_dyed(pos) if _board.is_valid(pos) else False
             for pos in _board.get_row_pos(col_pos)]
            for col_pos in _board.get_col_pos(
                _board.boundary(key=key)
            )
        ]
        mask_list = [
            [not _board.is_valid(pos) for pos in _board.get_row_pos(col_pos)]
            for col_pos in _board.get_col_pos(
                _board.boundary(key=key)
            )
        ]
        boards[key] = {
            "size": _board.get_config(key, "size"),
            "position": [_board.get_board_keys().index(key), 0],
            "showLabel": _board.get_config(key, "row_col"),
            "showName": not _board.get_config(key, "row_col"),
        }
        print(mask_list)
        if any(any(i) for i in mask_list):
            boards[key].update({
                "mask": mask_list
            })
        if any(any(i) for i in dye_list):
            boards[key].update({
                "dye": dye_list,
            })
        for pos, obj in _board(key=key):
            if obj is None:
                if _board.get_config(key, "pos_label"):
                    cells.append({
                        "type": "",
                        "position": {
                            "x": pos.x, "y": pos.y,
                            "boardname": pos.board_key
                        },
                        "component": {
                            "type": "template",
                            "style": "",
                            "value": {
                                "name": "backgroundStr",
                                "value": _board.pos_label(pos)
                            },
                        },
                        "highlight": {},
                        "clickable": True,
                        "overlayText": ""
                    })
                continue
            label = obj not in [
                VALUE_QUESS, MINES_TAG,
                _board.get_config(key, "MINES"),
                _board.get_config(key, "VALUE"),
            ]
            label = (
                    _board.get_config(key, "by_mini") and
                    label and
                    not (
                            isinstance(obj, ValueAsterisk) or
                            isinstance(obj, MinesAsterisk)
                    )
            )
            cells.append(
                format_cell(_board, pos, label))
            count += 1
    return boards, cells, count


def format_gamemode(gamemode: Mode, u_gamemode: UMode):
    u_mode: list[str] = []

    match gamemode:
        case Mode.NORMAL:
            mode = "NORMAL"
        case Mode.EXPERT:
            mode = "EXPERT"
        case Mode.ULTIMATE:
            mode = "ULTIMATE"
            if u_gamemode & UMode.ULTIMATE_A:
                u_mode.append("+A")
            if u_gamemode & UMode.ULTIMATE_F:
                u_mode.append("+F")
            if u_gamemode & UMode.ULTIMATE_S:
                u_mode.append("+S")
            if u_gamemode & UMode.ULTIMATE_R:
                u_mode.append("+R")
            if u_gamemode & UMode.ULTIMATE_P:
                u_mode.append("+!")
        case Mode.PUZZLE:
            mode = "PUZZLE"
        case _:
            mode = "UNKNOWN"
    return mode, u_mode
