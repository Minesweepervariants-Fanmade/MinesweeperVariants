# 右线规则接口说明

---

## 1. 概述

右线规则（Clue）负责表示谜题中线索对象的行为和约束，是规则系统中用于表达线索对象的关键组件。

本文档详细说明右线规则的接口组成、实现要求与函数规范，适用于希望扩展或实现自定义线索逻辑的开发者。

##  目录（导航索引）

###  总览

* [1. 概述](#1-概述)
* [2. 类结构概览](#2-类结构概览)
* [3. 方法总览（带跳转表格）](#3-方法总览)

### 开发说明与结构

* [4. 设计注意事项](#4-设计注意事项)
* [5. 实现位置](#5-实现位置)

### 实例与附录

* [6. 示例](#6-示例)
* [7. 测试与调试](#7-测试与调试)
* [8. 相关文档](#8-相关文档)

---

## 2. 类结构概览

本模块包含以下两个主要类：

| 类名                  | 类型  | 继承自                           | 功能说明                    |
|---------------------|-----|-------------------------------|:------------------------|
| `AbstractClueRule`  | 抽象类 | `abs.Rrule.AbstractClueRule`  | 表示一个规则的整体，负责填充线索与整体建模逻辑 |
| `AbstractClueValue` | 抽象类 | `abs.Rrule.AbstractClueValue` | 表示单个线索格的具体数值行为（如一个数字线索） |

## 注意事项（开发者须知）

* **规则名称通过类的`name`属性指定**，该属性是一个字符串列表（至少包含一个元素），例如：`name = ["V", "Vanilla"]`
* **规则名称不能与已有规则重复**，否则加载时会冲突或覆盖
* **模块开头的字符串说明或类的`doc`属性将作为规则简介在命令行中展示**，优先使用模块开头的字符串说明
* 简介内容仅限普通文本，不得包含代码、格式符或特殊符号
* 所有规则文件需放置在统一目录中, impl/rule中会自动检测并加载所有规则
* 程序加载规则时，自动提取上述说明并用于 `run list` 展示，缺失时显示为空

## 命名规范如下：

* 对于规则主名称 `XXX`：
  * 规则类应命名为 `RuleXXX`，继承自 `AbstractClueRule`
  * 线索类应命名为 `ValueXXX`，继承自 `AbstractClueValue`
* 规则类中的`name`属性示例：
  ```python
  name = ["1A", "A", "无马步", "Anti-Knight"]  # 第一个元素是主名称
  ```

---

## 3. 方法总览

### AbstractClueRule（规则类）

| 类型 | 名称                                                           | 简介            |
|----|--------------------------------------------------------------|---------------|
| 属性 | [name](#name)                                                | 规则名称属性        |
| 属性 | [doc](#doc)                                                  | 规则介绍说明        |
| 方法 | [__init\__](#__init__board-abstractboard--none-datanone)     | 构造函数，初始化规则和题板 |
| 方法 | [fill](#fillboard-abstractboard---abstractboard)             | 规则填充题板的方法     |
| 方法 | [init\_board](#init_boardboard-abstractboard)                | 初始化题板内容       |
| 方法 | [init_clear](#init_clearboard-abstractboard)                 | 初始化清理题板       |
| 方法 | [create_constraints](#create_constraintsboard-abstractboard) | 创建约束条件        |

### AbstractClueValue（线索类）

| 类型 | 名称                                                                         | 简介                  |
|----|----------------------------------------------------------------------------|---------------------|
| 方法 | [__init\__](#__init__pos-abstractposition-code-bytes--none)                | 构造函数，初始化线索位置和代码     |
| 方法 | [__repr\__](#__repr__)                                                     | 线索的字符串表示            |
| 方法 | [compose](#composeboard-web---listtextelement--imageelement)               | 生成线索的组合表示，接收board参数 |
| 方法 | [type](#type---bytes)                                                      | 获取线索类型              |
| 方法 | [code](#code---bytes)                                                      | 获取线索编码              |
| 方法 | [high_light](#hight_lightboard-abstractboard---listpos)                    | 获取线索编码              |
| 方法 | [deduce_cells](#deduce_cellsboard-abstractboard---bool)                    | 推断相关格子状态            |
| 方法 | [create_constraints](#create_constraintsboard-abstractboard-switch-switch) | 创建线索相关的约束条件         |

---

# AbstractClueRule 方法详解

---

### name

**类型**：字符串列表（至少包含一个元素）
**说明**：规则名称属性，如 `["1A", "无马步", "Anti-Knight"]`。第一个元素是主名称，后续元素为别名。

---

### doc

**类型**：字符串
**说明**：规则的详细文档说明，用于在命令行中展示。

---

### \__init\__(board: AbstractBoard = None, data=None)

**功能说明**：
可选的初始化方法，用于变量初始化或题板操作。

**参数**：

* `board (AbstractBoard)`：题板实例。
* `data`：默认不传参

**返回值**：
无。

**备注**:

data值可以在运行命令的时候使用`|`分割来传入data值

如`V|123`, `data='123'`

---

### fill(board: AbstractBoard) -> AbstractBoard

**功能说明**：
根据已布雷的题板，填充所有线索对象。

**参数**：

* `board (AbstractBoard)`：已布雷题板。

**返回值**：
填充好线索的 `AbstractBoard`。

---

### init_clear(board: AbstractBoard)

**功能说明**：
在题板生成阶段调用，用于删除题板上必须被清除的线索或对象。
例如纸笔题目中，某些规则可能要求特定位置不能出现雷或线索。

**参数**：

* `board (AbstractBoard)`：题板实例。

**返回值**：
无。

**备注**：
此方法为可选实现，视具体规则需求决定是否重写。

---

### `init_board(board: AbstractBoard)`

**功能说明**:
用于生成answer.png 需要将题板填充至无空

**参数**：

* `board (AbstractBoard)`：题板实例

**返回值**：

* 无

---

### create_constraints(board: AbstractBoard)

**功能说明**：
可选方法，向约束模型添加规则约束。

**参数**：

* `board (AbstractBoard)`：题板对象。

**返回值**：
无。

**备注**：
该方法在父类已实现，若需自定义请手动遍历题板线索调用，仅在线索类无法表达时使用。

---

### suggest\_total(info: dict)

**功能说明**：
规则在生成阶段调用，向模型添加硬约束，并通过调用 `info` 根键的软约束函数实现软约束。

**参数**：

* `info (dict)`：上下文信息字典，包含以下关键字段：
  * `size (dict[str, tuple[int, int]])` 其键为题板的字符串索引 值为size元组
  * `total (dict[str, int])`其键为题板的字符串索引 值为该题板的所有格子的数量
  * `interactive (list[str])`：题板交互权，列表内为题板索引，所有键均为允许求解器主动交互。
  * `hard_fns (list[Callable[[CpModel, IntVar], None]])`：硬约束函数列表。
    * 规则通过定义函数的形式添加硬约束（如调用 `model.Add(...)`），
    * 需要将该函数追加到此列表，生成器后续会统一调用执行，确保所有硬约束生效。
    * 函数签名应为 `(model: CpModel, total: IntVar) -> None`，不返回值。
  * `soft_fn (Callable[[int, int], None])`：软约束函数。
    * 签名为 `(target_value: int, priority: int)`，用于表示软约束的目标值和优先级。
    * 规则调用此函数以注册软约束，具体添加到模型的逻辑由生成器统一处理。
    * 规则只需传入期望的目标值与优先级，无需关心底层实现和返回值。
    * 关于总雷数优先级:
      * -1: 提前预设 如果未实现不会有任何影响(例:[R])
      * 0:  预设值 推荐使用该值 未实现可能题板会出现未知情况(例:[2T]是0.5)
      * 1:  高优先实现 如果未实现规则有大概率不可使用
      * 2:  必须实现 如果不实现规则将一定不可用(例:[1B']失衡)

**返回值**：
无。

### 使用示例

```python
class RuleTest:
  def suggest_total_mines(self, info):

      # 定义硬约束函数，追加到 info 的硬约束列表
      def hard_constraint(m, total):
          m.AddModuloEquality(0, total, 3)
      info["hard_fns"].append(hard_constraint)

      # 调用软约束函数，传入目标值和优先级
      info["soft_fn"](40, 0)
```

**备注**：

* `interactive` 用于表示题板的等级 只有与主板同级和副板两个等级；
* 硬约束以函数形式追加，生成器统一执行；
* 软约束函数只接受目标值和优先级，由生成器负责根据上下文处理；
* 该接口为可选实现，规则可根据需求重写。

---

# AbstractClueValue 方法详解

---

### \__init\__(pos: AbstractPosition, code: bytes = None)

**功能说明**：
初始化线索位置及编码数据。

**参数**：

* `pos (AbstractPosition)`：线索坐标。
* `code (bytes, 可选)`：线索编码字节码，来自 `code()` 方法。

**返回值**： 无。

---

### \__repr\__()

**功能说明**：
返回简短字符串表示，用于调试输出。

**返回值**：
简短字符串，建议字符数不超过5。

---

### compose(board, web) -> list\[TextElement | ImageElement]

**功能说明**：
（可选）渲染线索的视觉元素。

**参数**：

* `board (AbstractBoard)`：题板实例。
* `web (Boolean)`: 是否供web渲染(如果启用web将会无视部分渲染设置)

**返回值**：
返回由 `utils.image_create` 模块内的 `get_...` 系列函数生成的渲染组件（如 `get_text()` 或 `get_image()`）。

**备注**：
如果未实现此方法，则系统默认调用 `__repr__()` 返回的字符串进行渲染。
具体组件构造方式可参阅 [utils/image_create.py](./utils.md#image_createpy)。

---

### type() -> bytes

**功能说明**：
返回规则名称的字节码表示。

**返回值**：
字节串，如 `b"V"`。

**备注**：
最好与规则类的name一致

---

### code() -> bytes

**功能说明**：
返回线索实例的编码形式。

**返回值**：
字节串编码。

---

### deduce\_cells(board: AbstractBoard) -> bool

**功能说明**：
可选方法，对题板进行快速推理并原地修改。

**参数**：

* `board (AbstractBoard)`：题板实例。

**返回值**：
布尔值，`True` 表示题板有修改。/`False` 表示题板无修改

**备注**：
相对于对着该线索进行chord/左键单击

---

### hight_light(board: AbstractBoard) -> List[Pos]

**功能说明**：
可选方法，返回该线索的高亮范围

**参数**：

* `board (AbstractBoard)`：题板实例。

**返回值**：
所有高亮格子的列表

**备注**：
会被用作判断该线索是否无效的根据
如果高亮内部无任何None值则当作该值已经无效

---

### create\_constraints(board: AbstractBoard, switch: Switch)

**功能说明**：
为该线索添加ortools模型约束，优先实现此方法。

**参数**：

* `board (AbstractBoard)`：题板实例。
* `switch (Switch)`: 该约束的开关
  * 详见: [switch.get](utils.md/#switchget)


**返回值**：
无。

---

## 4. 设计注意事项

* 优先实现 `create_constraints()`，提升解题效率和准确性。
* 若规则难以用约束表达，可退化实现 `check()` 进行解过滤。
* 规则名称 `name` 必须唯一且规范。
* 线索编码需确保能完整还原实例状态。

明确：如果你需要在运行 `run list`（假设为列出规则列表的命令）时输出规则的说明文字，**则文档中应指明这个说明必须写在模块头部 docstring 中，供系统提取展示**。

因此，这部分要求应该被归入文档第 **5. 实现位置** 章节，**作为明确的强制规范**，内容建议如下添加：

---

## 5. 实现位置

右线规则的实现文件放置于：

```text
impl/rule/Rrule/
```

每个规则文件应遵循以下规范：

* 规则类命名为 `RuleXX`，继承自 `AbstractClueRule`
* 线索类命名为 `ValueXX`，继承自 `AbstractClueValue`
* 文件名必须为 `XX.py`，其中 `XX` 为规则代号（如 `V.py`, `2E.py`）

---

### 文件头注释规范（必需）

每个规则文件必须包含模块级 docstring，用于描述规则的含义。

> 该注释将由系统在运行 `run list` 命令时读取并展示，若缺失则不会显示规则说明。

**位置要求**：**必须放在文件顶部 `import` 之前。**

**格式规范**：

```python
"""
[规则代号] 规则详细名称：
简要描述该规则的功能。
"""
```

**示例**：

```python
"""
[T] 测试规则：
所有标记为“T”的格子代表非雷.
"""
```

该注释是规则展示和文档自动生成的依据，必须完整准确撰写。

---

## 6. 示例

```python
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/07/03 03:39:12
# @Author  : 作者名
# @FileName: V.py
"""
[T]测试: 这是一个规则说明
所有标记为“T”的格子代表非雷.(这说的什么废话)
"""

from minesweepervariants.abs.Rrule import AbstractClueRule, AbstractClueValue

class RuleT(AbstractClueRule):
    name = "T"

    def fill(self, board):
        for pos, val in board(target="N"):
            board[pos] = ValueT(pos)
        return board
    
    def create_constraints(self, board, switch):
        # 规则对整体的约束, 一般适用于左线或优化(示例:QL,2E)
        ...


class ValueT(AbstractClueValue):
    def __init__(self, pos, code=None):
        self.pos = pos
        self.code = code

    def __repr__(self):
        return "T"

    def type(self):
        return RuleT.name.encode("ascii")

    def code(self):
        return b""

    def create_constraints(self, board, switch):
        model = board.get_model()
        var = board.get_variable(self.pos)
        model.Add(var == 0)
```

> 代码实现放置于 `impl/rule/Rrule/` 目录，可直接通过命令行调用对应规则。

## 7. 测试与调试

* 确认 `fill()` 正确填充线索对象。

* 验证 `create_constraints()` 限制解空间效果。

* 使用日志输出调试实现细节，确保规则逻辑正确。

* `check()` 需严格判断非法状态。

## 8. 相关文档

### [README.md](../README.md) << 入口文档

### [rule_mines.md](./rule_mines.md) << 左线规则接口

### [rule_clue_mines.md](./rule_clue_mines.md) << 中线规则接口

### [board_api.md](./board_api.md) << 题板与坐标系统接口

### [utils.md](./utils.md) << 工具模块说明
