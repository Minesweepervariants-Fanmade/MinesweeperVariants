"""
[&CL] Calendar (每日一题) 染色规则拆分定义
作者: NT (2201963934)
最后编辑时间: 2026-04-07 15:20:24

1) 规则对象与适用范围
- 规则类型: 染色规则 (AbstractDye 子类), 不直接生成线索/雷值。
- 作用对象: 当前 interactive 子板中的所有有效格 (board(key=...)).
- 染色语义: “扣掉”的格子为染色格(True), 非扣掉格为非染色(False)。
- 题版形状语义: “不在题板内的格子对应染色格子”在实现上等价为:
  以该月日历外接矩形为参照, 所有不属于“本月日期单元”的位置均染色。

2) 核心术语定义
- 年/月: 默认取系统当前日期对应的 year/month, 可由 args 覆盖。
- 今日日期(today): 默认取系统当前日 day, 可由 args 覆盖。
- 日历外接矩形: 7 列(周一到周日) x R 行, R 为覆盖该月所有日期所需周数。
- 本月日期单元: 在外接矩形内, 对应 1..days_in_month 的格子。
- 扣掉今日: 从“本月日期单元”中排除 day=today 的那个单元并将其染色。

3) 计数对象、边界条件、越界处理
- 无线索计数行为, 仅设置每格 dyed 布尔值。
- 若题板尺寸小于外接矩形: 仅对交集区域赋值(超出题板范围的日历格忽略)。
- 若题板尺寸大于外接矩形: 外接矩形之外位置一律染色(True)。
- 当 today 不在 1..days_in_month 时: 不执行“扣掉今日”例外(仅按是否属于本月日期单元判定)。
- 周起始约定固定为周一(与 Python calendar.monthrange 返回首日偏移一致, 0=周一)。

4) 阶段语义等价性
- dye 阶段语义: 对每个有效格 pos, set_dyed(pos, is_removed_or_outside_month_cell)。
- 约束阶段等价语义: 后续任意规则通过 board.get_dyed(pos) 读取到的值,
  必须与上述判定函数完全一致, 不依赖随机性且同参数下确定。

5) 可验证样例
- 样例A: 2026-04, today=8。
  2026-04-01 是周三(周一=0时偏移2), 30天覆盖5周。
  则日历矩形为 7x5:
  - 第1周前两格(周一、周二)为“非本月日期单元”, 应染色。
  - 4月8日对应第2周周三单元, 应被“扣掉”并染色。
  - 4月9日对应单元属于本月且非今日, 应为非染色。
"""

import calendar
import datetime
import re

from . import AbstractDye


class DyeCL(AbstractDye):
  name = "cl"
  fullname = "每日一题染色"

  def __init__(self, args: str):
    self.today = datetime.date.today()
    self.year, self.month, self.day = self._parse_date_args(args)

  def _parse_date_args(self, args: str) -> tuple[int, int, int]:
    text = (args or "").strip()
    if text == "":
      return self.today.year, self.today.month, self.today.day

    day_default = self.today.day
    patterns = [
      (r"^(\d{4})-(\d{2})-(\d{2})$", True),
      (r"^(\d{4})/(\d{2})/(\d{2})$", True),
      (r"^(\d{8})$", True),
      (r"^(\d{4})-(\d{2})$", False),
      (r"^(\d{4})/(\d{2})$", False),
      (r"^(\d{6})$", False),
    ]

    for pattern, has_day in patterns:
      match = re.fullmatch(pattern, text)
      if match is None:
        continue
      if pattern == r"^(\d{8})$":
        raw = match.group(1)
        year = int(raw[:4])
        month = int(raw[4:6])
        day = int(raw[6:8])
      elif pattern == r"^(\d{6})$":
        raw = match.group(1)
        year = int(raw[:4])
        month = int(raw[4:6])
        day = day_default
      else:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if has_day else day_default
      self._validate_year_month_day(year, month, day, has_day=has_day)
      return year, month, day

    raise ValueError(
      "Invalid date args for @cl: expected YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD, "
      "YYYY-MM, YYYY/MM, YYYYMM, or empty string"
    )

  @staticmethod
  def _validate_year_month_day(year: int, month: int, day: int, *, has_day: bool):
    if month < 1 or month > 12:
      raise ValueError(f"Invalid month for @cl: {month}. Month must be in 1..12")

    days_in_month = calendar.monthrange(year, month)[1]
    if has_day and (day < 1 or day > days_in_month):
      raise ValueError(
        f"Invalid day for @cl: {day}. Day must be in 1..{days_in_month} for {year:04d}-{month:02d}"
      )

  def dye(self, board):
    first_weekday, days_in_month = calendar.monthrange(self.year, self.month)
    rows = (first_weekday + days_in_month + 6) // 7
    day_in_range = 1 <= self.day <= days_in_month

    for key in board.get_interactive_keys():
      size_x, size_y = board.get_config(key, 'size')
      for x in range(size_x):
        for y in range(size_y):
          pos = board.get_pos(x, y, key)
          if pos is None:
            continue

          if x >= rows or y >= 7:
            board.set_dyed(pos, True)
            continue

          index = x * 7 + y
          day_num = index - first_weekday + 1
          in_month_cell = 1 <= day_num <= days_in_month
          is_today = day_in_range and day_num == self.day
          board.set_dyed(pos, (not in_month_cell) or is_today)
