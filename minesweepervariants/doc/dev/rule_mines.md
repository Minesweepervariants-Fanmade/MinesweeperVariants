# 左线规则接口说明

---

## 1. 概述

左线规则接口，定义地雷布局相关的约束与逻辑。

---

## 目录

* [2. 类结构概览](#2-类结构概览)
* [3. 方法总览（含跳转）](#3-方法总览)
* [4. 设计注意事项](#4-设计注意事项)
* [5. 实现位置](#5-实现位置)

---

## 2. 类结构概览

| 类名                  | 类型  | 继承自                           | 功能说明                   |
|---------------------|-----|-------------------------------|------------------------|
| `AbstractMinesRule` | 抽象类 | `abs.Lrule.AbstractMinesRule` | 表示一个地雷布局规则，控制布雷逻辑与建模约束 |


## 注意事项（开发者须知）

* **规则名称必须全大写**，例如：`V`, `1Q`, `2E`。
* **规则名称不能与已有规则重复**，否则加载时会冲突或覆盖。
* **模块开头的字符串说明将作为规则简介在命令行中展示**。
* 简介内容仅限普通文本，不得包含代码、格式符或特殊符号。
* 所有规则文件需放置在统一目录中, impl/rule中会自动检测并加载所有规则。
* 程序加载规则时，自动提取上述说明并用于 `run list` 展示，缺失时显示为空。

## **命名规范**：

* 对于规则 `XXX`：

  * 规则类应命名为 `RuleXXX`
  * 必须继承自 `AbstractMinesRule`

---

## 3. 方法总览

| 类型 | 名称                                                            | 简介            |
|----|---------------------------------------------------------------|---------------|
| 属性 | [name](#name)                                                 | 规则名称属性        |
| 属性 | [subrules](#subrules)                                         | 规则子项列表        |
| 方法 | [\_\_init\_\_](#__init__board-abstractboard--none-datanone)   | 构造函数，初始化规则和题板 |
| 方法 | [deduce\_cells](#deduce_cellsboard-abstractboard---bool)      | 推导格子状态的方法     |
| 方法 | [create\_constraints](#create_constraintsboard-abstractboard) | 创建约束条件        |
| 方法 | [init\_board](#init_boardboard-abstractboard)                 | 初始化题板内容       |
| 方法 | [init\_clear](#init_clearboard-abstractboard)                 | 初始化清理题板       |
| 方法 | [suggest\_total](#suggest_totalinfo-dict)                     | 建议总雷数的方法      |


> 每个规则类文件建议在**文件头部加注释说明规则含义**，以便 `run list` 命令扫描输出。
>
> 建议格式：
>
> ```python
> """
> [0R]：总雷数规则，控制题板中雷的总数
> """
> ```

---

## 4. 设计注意事项

* 优先实现 `create_constraints()`，提升解题效率与解唯一性
* 若规则无法用 OR-Tools 表达，应实现 `check()` 并通过 `method_choose()` 明确声明
* 所有变量必须使用 `board.get_variable(pos)` 获得
* CP 模型通过 `board.get_model()` 获取
* `name` 必须为唯一字符串常量，用于规则识别
* 若涉及交互线索或检查开关功能，应实现 `subrules` 属性

---

## 5. 实现位置

左线规则实现文件应放置于：

```text
impl/rule/Lrule/
```

实现类命名建议：

* `Rule_0R`：规则类（继承自 `AbstractMinesRule`）

---

## 6. 方法详解

---

### name

**类型**：字符串列表（至少包含一个元素）
**说明**：规则名称属性，如 `["1A", "无马步", "Anti-Knight"]`。第一个元素是主名称，后续元素为别名。

---

### subrules

**类型**：`list[list[bool, str]]`（可选）

**说明**：
定义规则的子模块开关与描述，用于交互式检查或线索提示。

**结构**：

* 第一个元素：布尔值，表示该子规则模块是否启用
* 第二个元素：字符串描述，用于显示

**示例**：

```python
subrules = [
    [True, "[1B]列平衡"],
    [True, "[1B]行平衡"]
]
```

---

## `__init__(board: AbstractBoard = None, data=None)`

**功能说明**：
可选的初始化方法，用于变量初始化或题板操作。

**参数**：

* `board (AbstractBoard)`：题板实例。
* `data`：默认不传参

**返回值**：
无。

---

## `deduce_cells(board: AbstractBoard) -> bool`

**功能说明**：
快速分析当前题板，尝试推出某些格子状态（是否必为雷或非雷），并直接修改题板。

**参数**：

* `board (AbstractBoard)`：当前题板对象

**返回值**：

* `True` 表示题板被修改（即推出了内容）
* `False` 表示无法推出（无修改）

**备注**：

* 属于优化功能，不影响解的正确性
* 不实现时可省略该函数，系统默认跳过此优化

---

## `create_constraints(board: AbstractBoard)`

**功能说明**：
为当前左线规则向 OR-Tools 模型添加约束表达式。

**参数**：

* `board (AbstractBoard)`：题板实例

**实现要点**：

* 所有变量必须通过 `board.get_variable(pos)` 获取
* 模型对象通过 `board.get_model()` 获取

**返回值**：

* 无

**示例结构**：

```python
from minesweepervariants.utils.impl_obj import get_total

class RuleR(...):
  def create_constraints(self, board):
      model = board.get_model()
      variables = [board.get_variable(pos) for pos in board.iter_all()]
      model.Add(sum(variables) == get_total())
```

---

### `init_board(board: AbstractBoard)`

**功能说明**:
用于生成answer.png 需要将题板填充至无空

**参数**：

* `board (AbstractBoard)`：题板实例

**返回值**：

* 无

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

### `init_clear(board: AbstractBoard)`

**功能说明**:
在执行完随机放雷后调用该接口 可以进行其他操作

**参数**：

* `board (AbstractBoard)`：题板实例

**返回值**：

* 无

---

## 7. 示例结构

```python
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 创建文件时间
# @Author  : 作者名
# @FileName: 0R.py
"""
[0R]：总雷数规则，控制题板中雷的总数
"""
from minesweepervariants.abs.Lrule import AbstractMinesRule
from minesweepervariants.utils.impl_obj import get_total

class Rule0R(AbstractMinesRule):
    name = "0R"
    subrules = [
      [True, "0R"]
    ]
    def create_constraints(self, board: 'AbstractBoard', switch: 'Switch'):
        if not self.subrules[0][0]:
            return
        model = board.get_model()
        all_variable = [board.get_variable(pos) for pos, _ in board()]
        model.Add(sum(all_variable) == get_total())
```

---

## 8. 相关文档

### [README.md](../README.md) << 入口文档

### [rule_clue.md](./rule_clue.md) << 右线规则接口

### [rule_clue_mines.md](./rule_clue_mines.md) << 中线规则接口

### [board_api.md](./board_api.md) << 题板与坐标系统接口

### [utils.md](./utils.md) << 工具模块接口说明
