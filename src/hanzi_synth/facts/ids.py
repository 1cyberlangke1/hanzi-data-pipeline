"""IDS（表意文字描述序列）解析器。

依据 findings.md §9.7 的实测结论：
- `property.ids` 是**逗号分隔的多候选串联**，每项尾部挂来源标签，例如
  `丑` = `⿼ユ十(G),⿻⿼ユ丨一(H),⿻⿼コ丨一(y)`、`一` = `#(H)(G),{一}#(T)(t)`；
- 解析结果必须保留**候选集合**（一字多解都合法），不是单值；
- `#(...)` 是叶典笔画记法（H 横 / A 竖 / P 撇 / D 点 / N 捺 / T 提 / G 钩）、
  `{X}` 表示 X 本身，两者都**不是** IDS，解析时丢弃。

IDC（表意文字描述符）按 Unicode 定义取全集：U+2FF0–U+2FFF 与 U+31EF（Unicode 15.1 新增的「减法」描述符）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 一元描述符：横镜像 / 旋转
IDC_UNARY = frozenset("\u2ffe\u2fff")
# 二元描述符：左右上下包围等 + U+2FFC/2FFD + U+31EF（减法）
IDC_BINARY = frozenset("\u2ff0\u2ff1\u2ff4\u2ff5\u2ff6\u2ff7\u2ff8\u2ff9\u2ffa\u2ffb\u2ffc\u2ffd\u31ef")
# 三元描述符：左中右 / 上中下
IDC_TERNARY = frozenset("\u2ff2\u2ff3")
IDC_ALL = IDC_UNARY | IDC_BINARY | IDC_TERNARY

# 括号内是来源标签（G/H/T/t/y/w/q1283…）或笔画记法，统一剥掉
_TAG_RE = re.compile(r"\([^()]*\)")

# 可作为 IDS 部件的码位区间（CJK 各平面 + 部首补充 + 康熙部首 + CJK 笔画）
_COMPONENT_RANGES = (
    (0x2E80, 0x2EFF),  # CJK 部首补充
    (0x2F00, 0x2FDF),  # 康熙部首
    (0x31C0, 0x31EF),  # CJK 笔画
    (0x3400, 0x4DBF),  # 扩展 A
    (0x4E00, 0x9FFF),  # 基本区
    (0xF900, 0xFAFF),  # 兼容表意文字
    (0x20000, 0x323AF),  # 扩展 B–I
)


class IdsSyntaxError(ValueError):
    """输入：不合语法的 IDS 串。输出：可读异常。预期：调用方捕获后判该候选非法。"""


def is_component_char(ch: str) -> bool:
    """输入：单个字符。输出：它能否作为 IDS 的部件（汉字/部首/笔画符号）。
    预期：片假名 `コ`、拉丁字母、数字等一律 False——它们只是叶典的记号，不是部件。
    """
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _COMPONENT_RANGES)


def is_wellformed(ids: str) -> bool:
    """输入：一个候选串。输出：它是不是合法 IDS（首字符是描述符，其余全是描述符或部件）。
    预期：不做括号剥除，调用方需先剥标签。
    """
    if not ids or ids[0] not in IDC_ALL:
        return False
    for ch in ids:
        if ch in IDC_ALL or is_component_char(ch):
            continue
        return False
    return True


def parse_candidates(raw: str | None) -> tuple[str, ...]:
    """输入：`property.ids` 原始串（可能为 None）。输出：去重后的合法 IDS 候选元组。
    预期：顶层按逗号切分 → 剥掉所有 `(...)` 标签 → 去掉 `{` `}` `#` → 保留合法项。
          全部被过滤掉时返回空元组，调用方据此判「无 IDS，不进合成池」。
    """
    if not raw:
        return ()
    seen: list[str] = []
    for chunk in raw.split(","):
        candidate = _TAG_RE.sub("", chunk).replace("{", "").replace("}", "").replace("#", "").strip()
        if is_wellformed(candidate) and candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


@dataclass(frozen=True)
class Node:
    """输入：解析后的子树。输出：IDS 语法树节点。
    预期：`op` 为 None 时是叶子，部件字符在 `char`；否则 `children` 有 1/2/3 个孩子。
    """

    op: str | None
    children: tuple["Node", ...] = ()
    char: str | None = None

    @property
    def is_leaf(self) -> bool:
        """输入：无。输出：本节点是否叶子。预期：叶子才有 char。"""
        return self.op is None


def parse_tree(ids: str) -> Node:
    """输入：一个合法 IDS 串。输出：语法树根节点。
    预期：串不合法或有多余字符时抛 IdsSyntaxError。
    """
    node, pos = _parse_at(ids, 0)
    if pos != len(ids):
        raise IdsSyntaxError(f"IDS 有未消费的尾部: {ids!r} 停在 {pos}")
    return node


def _parse_at(ids: str, index: int) -> tuple[Node, int]:
    """输入：IDS 串与当前位置。输出：(子树, 下一个位置)。
    预期：按描述符元数递归下降；越界或字符不合法时抛 IdsSyntaxError。
    """
    if index >= len(ids):
        raise IdsSyntaxError(f"IDS 提前结束: {ids!r}")
    ch = ids[index]
    if ch not in IDC_ALL:
        if not is_component_char(ch):
            raise IdsSyntaxError(f"IDS 里出现非部件字符 {ch!r}: {ids!r}")
        return Node(op=None, char=ch), index + 1
    arity = 1 if ch in IDC_UNARY else 2 if ch in IDC_BINARY else 3
    cursor = index + 1
    children: list[Node] = []
    for _ in range(arity):
        child, cursor = _parse_at(ids, cursor)
        children.append(child)
    return Node(op=ch, children=tuple(children)), cursor


def leaves(node: Node) -> tuple[str, ...]:
    """输入：IDS 树根。输出：从左到右的全部叶子部件字符（含重复）。
    预期：递归遍历，顺序与书写顺序一致。
    """
    if node.is_leaf:
        return (node.char or "",)
    out: list[str] = []
    for child in node.children:
        out.extend(leaves(child))
    return tuple(out)


def leaf_count(node: Node) -> int:
    """输入：IDS 树根。输出：叶子总数（含重复）。预期：T9/T12 用它定位部件下标。"""
    return len(leaves(node))


def render(node: Node) -> str:
    """输入：IDS 树根。输出：还原成 IDS 字符串。
    预期：与 parse_tree 互为逆运算——`render(parse_tree(x)) == x` 对合法 IDS 成立。
    """
    if node.is_leaf:
        return node.char or ""
    return (node.op or "") + "".join(render(child) for child in node.children)


def replace_leaf(node: Node, position: int, char: str) -> Node:
    """输入：IDS 树根、叶子下标（从左到右，0 起）、替换成的部件字符。输出：新树。
    预期：下标越界时抛 IdsSyntaxError；不修改原树（返回新节点），保证 TaskSpec 可复现。
    """
    if position < 0 or position >= leaf_count(node):
        raise IdsSyntaxError(f"叶子下标 {position} 越界（共 {leaf_count(node)} 个）")
    counter = [0]

    def walk(current: Node) -> Node:
        if current.is_leaf:
            index = counter[0]
            counter[0] += 1
            return Node(op=None, char=(char if index == position else current.char))
        return Node(op=current.op, children=tuple(walk(child) for child in current.children))

    return walk(node)


def top_operator(ids: str) -> str | None:
    """输入：合法 IDS 串。输出：顶层描述符（独体字返回 None）。
    预期：结构名称一律取自 IDS 顶层算子，**不用** `property.structure` 的整数码
          （那个码表在数据里没有文档，猜错就会产出错数据）。
    """
    tree = parse_tree(ids)
    return None if tree.is_leaf else tree.op
