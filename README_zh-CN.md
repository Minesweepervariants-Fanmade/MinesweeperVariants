# MinesweeperVariants

## **版本：1.3**

**14 Minesweeper Variants（简称 14mv）** 是一款扫雷变体的解谜游戏生成器+游戏服务器(前端项目请使用[Minesweeper Variants-Vue](https://koolshow.github.io/MinesweeperVariants-Vue/))。
支持多种扫雷规则组合，可生成具有**唯一解**的纸笔类谜题。

---

## 安装

### 系统要求

本项目要求安装 Python 3.13（推荐）。请从 [https://www.python.org](https://www.python.org) 下载并安装对应平台的安装包。

> 注意：如果你使用虚拟环境（venv/virtualenv）或 Poetry，也可以在虚拟环境中安装本包，推荐在隔离环境中操作以避免与系统包冲突。

### 通过 pip 安装

使用 pip 安装发布版：

```bash
python -m pip install --upgrade pip
python -m pip install minesweepervariants
```

## 运行

下面给出两种常用的运行方式：启动服务器和运行生成脚本。

启动服务器：

```bash
python -m minesweepervariants.server
# 运行后不要关闭, 请在浏览器内打开https://koolshow.github.io/MinesweeperVariants-Vue/
```

运行生成脚本（生成题板/使用命令行工具）：

```bash
python -m minesweepervariants
```

你可以在运行生成脚本时附加命令行参数，例如 `-s` 指定尺寸，`-t` 指定总雷数，更多参数见下文示例与参数说明。

---

## 开发环境配置

本项目推荐使用 [Poetry](https://python-poetry.org/) 进行依赖管理和环境配置。

### 1. 安装 Python（推荐 3.13）

Windows 可至 [https://www.python.org](https://www.python.org) 下载官方安装包。

### 2. 安装 Poetry

请参考官方文档安装 Poetry：[https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

### 3. 安装依赖并自动创建虚拟环境

在项目根目录下执行：

```bash
poetry install
```

Poetry 会自动为项目创建隔离的虚拟环境并安装所有依赖。

### 4. 运行项目

无需手动激活虚拟环境，直接使用 Poetry 运行项目脚本：

```bash
poetry run python run.py [参数列表]
```

其它命令同理，均可用 `poetry run <命令>` 方式执行。

### 4. C 扩展构建环境（Windows）

部分依赖可能需要编译 C 扩展，必须安装：

* Visual C++ Build Tools：

可从以下地址下载安装：

> [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

勾选内容包括：

* C++ 生成工具（含 MSVC、Windows SDK）
* CMake

安装完成后请重新启动终端。

---

## 使用说明

### 运行方式

```bash
run [参数列表]
```

调用主程序，生成谜题。

---

### 常用参数（run）

|参数|类型|说明|
|---|---|---|
|`-s, --size`|整数（必填）|谜题尺寸|
|`-t, --total`|整数|地雷总数|
|`-c, --rules`|字符串列表|所有规则列表（如 `2F 1Q V 1K 1F`），会自动分类为左中右线规则|
|`-E, --early-rules`|字符串列表|仅在最初生成题板阶段生效的附加规则，可多个，且必须是左线规则|
|`-d, --dye`|字符串|染色函数名（如 `@c`）|
|`-m, --mask`|字符串|掩码染色规则名称，如 `@c`|
|`-r, --used-r`|布尔开关|是否在推理中启用 R（默认不启用）|
|`-a, --attempts`|整数|生成谜题的最大尝试次数|
|`-q, --query`|字符串范围|线索数量范围过滤，如 `5-8`、`-8`、`5`|
|`-e, --early-stop`|布尔开关|query 模式下达到线索目标时提前退出（可能导致线索图不正确）|
|`-v, --vice-board`|布尔开关|允许在生成中删除副板信息|
|`-T, --test`|布尔开关|仅生成一份使用了规则的答案题板|
|`-S, --seed`|整数|随机种子（显式设置后会自动将 attempts 设为 1）|
|`-O, --onseed`|布尔开关|启用可复现种子生成（速度会下降）|
|`-L, --log-lv`|字符串|日志等级，支持 `DEBUG`、`INFO`、`WARNING` 等|
|`-B, --board-class`|字符串|底层实现的题板类名/题板名称，通常使用默认值即可|
|`-I, --no-image`|布尔开关|不生成图片文件|
|`-F, --file-name`|字符串|输出文件名前缀|
|`-D, --dynamic-dig-rounds`|整数|动态删线索迭代轮数（不填时按规则自动决定）|
|`-M, --dynamic-dig-max-batch`|整数|动态删线索每轮最大改动格数|
|`--output-path`|字符串|图片输出目录路径|
|`--log-path`|字符串|日志输出目录路径|
|`--lang`|字符串|输出语言代码，例如 en_US 或 zh_CN|
|`list`||列出当前所有实现的规则内容|

---

### 运行示例

```bash
run -s 5 -c 2F 1k 1q V -d c -r -q 2-4 -I
# 仅初始生成阶段附加左线规则示例
run -s 5 -c 1Q V -E 2F 3L -I
```

> 生成一道 5×5 题板，棋盘格染色，规则使用 2F、1Q（左线），V、1K（右线）
> 携带 R 推理，仅保留线索数在 2 到 4 之间的题板，并关闭图片输出
> 注: 规则名大小写不敏感

`list` 子命令补充参数：

|参数|类型|说明|
|---|---|---|
|`--json`|布尔开关|以 JSON 格式输出规则文档|

---

### 运行结果输出（run）

运行成功后将在 `output/` 目录中生成以下文件：

```
output/
├─ output.png   (img 图片默认输出文件)
├─ demo.txt     (历史所有可推理解密文本)
├─ demo.png     (题目图片)
└─ answer.png   (答案图片)
```

`demo.txt` 中将包含以下内容：

* 生成时间
* 线索表（仅使用 `-q` 时生效）
* 生成用时
* 总雷数：格式为 总雷数 / 总空格
* 种子 / 题号：一串整数数字
* 题板的题目内容
* 题板的答案和无问号时内容
* 题板图片的命令生成指令（以 `img` 开头）
* 答案图片的命令生成指令

---

### 图像输出方式

```bash
img [参数列表]
```

调用图像输出子命令。

---

### 参数列表（img）

| 参数                  | 类型     | 说明                             |
| --------------------- | -------- | -------------------------------- |
| `-c, --code`        | 字符串   | 题板字节码，表示固定题板内容     |
| `-r, --rule-text`   | 字符串   | 规则字符串（含空格需加引号）     |
| `-s, --size`        | 整数     | 单元格尺寸                       |
| `-o, --output`      | 字符串   | 输出文件名（不含后缀）           |
| `-w, --white-base`  | 布尔开关 | 是否使用白底                     |
| `-b, --board-class` | 字符串   | 底层实现的board类,使用默认值即可 |

---

### 运行示例

```bash
img -c ... -r "[V]-R*/15-4395498" -o demo -s 100 -w
```

> 生成图片 底部文字使用[V]-R*/15-4395498 输出保存至output/demo.png
> 每个格子的大小是100x100像素的 白底的图片

> 注: `...`需要替换为题板的代码值 其内容被保存至output/demo.txt内部

---

## 开发者文档结构

项目包含完整的开发文档，位于 [`doc/`](./doc) 目录中：

| 文档                                          | 说明               |
| --------------------------------------------- | ------------------ |
| [README.md](./doc/README.md)                     | 入口文件           |
| [dev/rule_mines.md](./doc/dev/rule_mines.md)     | 左线规则接口说明   |
| [dev/rule_clue_mines.md](./doc/dev/rule_clue.md) | 中线规则接口说明   |
| [dev/rule_clue.md](./doc/dev/rule_clue.md)       | 右线规则接口说明   |
| [dev/board_api.md](./doc/dev/board_api.md)       | 题板结构与坐标系统 |
| [dev/utils.md](./doc/dev/utils.md)               | 工具模块接口       |