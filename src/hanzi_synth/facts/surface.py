"""surface 层：canonical 用于判等/去重/索引，surface 用于输出，且**必须有多形态分布**。

findings.md 的结论：只用 ASCII 会砍掉多样性，所以同一份内容要同时具备
Unicode 与 ASCII 两种写法，并在数据集层面**成分布**地出现（不是二选一就固定）。

本模块只做确定性变换，不做任何语义推断：
- 读音：`xie2`（ASCII 数字调）↔ `xié`（Unicode 声调符）
- 字符：`㐂` ↔ `U+3402`
- IDS ：`⿰女子` ↔ `左右（女、子）`（描述式）
"""

from __future__ import annotations

from .ids import IDC_ALL, Node, parse_tree

# 声调符表：canonical 用 ASCII，surface 用带调元音（v 代表 ü）
_TONE_TABLE: dict[str, str] = {
    "a": "\u0101\u00e1\u01ce\u00e0",
    "e": "\u0113\u00e9\u011b\u00e8",
    "i": "\u012b\u00ed\u01d0\u00ec",
    "o": "\u014d\u00f3\u01d2\u00f2",
    "u": "\u016b\u00fa\u01d4\u00f9",
    "v": "\u01d6\u01d8\u01da\u01dc",
}
_VOWELS = "aeiouv"

# IDC 的中文名称（Unicode 名称的意译，非自造术语）
IDC_LABELS: dict[str, str] = {
    "\u2ff0": "左右",
    "\u2ff1": "上下",
    "\u2ff2": "左中右",
    "\u2ff3": "上中下",
    "\u2ff4": "全包围",
    "\u2ff5": "上三包围",
    "\u2ff6": "下三包围",
    "\u2ff7": "左三包围",
    "\u2ff8": "左上包围",
    "\u2ff9": "右上包围",
    "\u2ffa": "左下包围",
    "\u2ffb": "重叠",
    "\u2ffc": "右包围",
    "\u2ffd": "左下包围",
    "\u31ef": "减法",
}


def _tone_index(syllable: str) -> int:
    """输入：无声调音节（ASCII，v 代表 ü）。输出：该标调的元音下标，找不到返回 -1。
    预期：按汉语拼音规则——有 a 标 a；无 a 则标 o/e；再无则标最后一个元音（iu→u、ui→i）。
    """
    if "a" in syllable:
        return syllable.index("a")
    for vowel in ("o", "e"):
        if vowel in syllable:
            return syllable.index(vowel)
    for index in range(len(syllable) - 1, -1, -1):
        if syllable[index] in _VOWELS:
            return index
    return -1


def reading_surfaces(canonical: str) -> tuple[str, ...]:
    """输入：canonical 读音（如 `xie2`、`lv4`）。输出：surface 变体元组（ASCII 形式在前）。
    预期：轻声（5 调）与无声调音节只返回自身；这是确定性变换，不改变读音本身。
    """
    if not canonical:
        return ()
    if canonical[-1] not in "12345":
        return (canonical,)
    tone = int(canonical[-1])
    body = canonical[:-1]
    if tone == 5 or not body:
        return (canonical,)
    position = _tone_index(body)
    if position < 0:
        return (canonical,)
    marked = body[:position] + _TONE_TABLE[body[position]][tone - 1] + body[position + 1 :]
    return (canonical, marked)


def char_surfaces(char: str) -> tuple[str, ...]:
    """输入：单个字符。输出：(原字符, `U+XXXX`) 两种 surface。
    预期：空串返回空元组；码点按至少 4 位十六进制大写书写（补充平面自然超过 4 位）。
    """
    if not char:
        return ()
    return (char, f"U+{ord(char):04X}")


def _utf8_hex(char: str) -> str:
    """输入：单个字符。输出：UTF-8 字节的大写十六进制（空格分隔）。预期：纯确定性编码，不做任何裁剪。"""
    return " ".join(f"{byte:02X}" for byte in char.encode("utf-8"))


def _utf16_hex(char: str) -> str:
    """输入：单个字符。输出：UTF-16BE 码元的大写十六进制（空格分隔，非 BMP 字即代理对）。"""
    raw = char.encode("utf-16-be")
    return " ".join(raw[index : index + 2].hex().upper() for index in range(0, len(raw), 2))


def codepoint_surfaces(char: str) -> tuple[str, ...]:
    """输入：单个字符。输出：同一码位的全部等价写法，顺序固定。
    预期：BMP 字给 4 条（`U+XXXX` / 裸 hex / 十进制 / UTF-8 字节），
          非 BMP 字再加第 5 条 UTF-16 代理对；非单字符输入返回空元组。
          调用方可任选一条当主答案、其余当等价写法，从而让 hex / 十进制 / 字节在数据集层面成分布。
    """
    if len(char) != 1:
        return ()
    code = ord(char)
    forms = [f"U+{code:04X}", f"{code:04X}", str(code), _utf8_hex(char)]
    if code > 0xFFFF:
        forms.append(_utf16_hex(char))
    return tuple(forms)


def _describe(node: Node) -> str:
    """输入：IDS 语法树节点。输出：中文描述式（叶子就是字符本身）。
    预期：递归展开，运算符用 Unicode 名称的中文意译；未知描述符退化为原符号。
    """
    if node.is_leaf:
        return node.char or ""
    label = IDC_LABELS.get(node.op or "", node.op or "")
    parts = "、".join(_describe(child) for child in node.children)
    return f"{label}（{parts}）"


def ids_surfaces(ids: str) -> tuple[str, ...]:
    """输入：合法 IDS 串。输出：(Unicode IDC 形式, 描述式) 两种 surface。
    预期：解析失败时只返回原串——绝不因为 surface 变换失败而丢数据。
    """
    if not ids:
        return ()
    try:
        described = _describe(parse_tree(ids))
    except Exception:  # noqa: BLE001 - surface 变换失败不该影响主流程
        return (ids,)
    if described == ids:
        return (ids,)
    return (ids, described)


def has_idc(text: str) -> bool:
    """输入：任意文本。输出：其中是否含 IDC 描述符。预期：给校验器判断「这是不是 IDS」。"""
    return any(ch in IDC_ALL for ch in text)
