"""合成字库的检索索引（L2 构造器的公共底座）。

构造器里有四类需求不能靠线性扫描：
- T8 约束满足 / T10 反例：按 (部首, 总笔画, 顶层结构) 分桶，用来判「唯一解」与取「补集」；
- T5 异体解析 / T9 部件链：`字符 -> Character` 与 `IDS -> Character` 反查；
- T12 结构变换：把 IDS 的某个叶子**挖空**后建索引，于是「替换后能否命中真实收录字」
  从 0.93% 的暴力试错变成 O(1) 查表；
- T9 部件链：需要知道某个部件自己有没有 IDS，同样走 `字符 -> Character`。

本模块只做确定性索引，不做任何语义推断：字典内容按事实对待，不做交叉校验。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .ids import IdsSyntaxError, Node, leaves, parse_tree, render, replace_leaf, top_operator
from .records import Character

# 挖空叶子用的哨兵字符：它不是任何合法部件，故不会与真实 IDS 撞键
BLANK = "\x00"


def bucket_key(character: Character) -> tuple[str, int, str | None] | None:
    """输入：字记录。输出：约束桶的键 (部首, 总笔画, 顶层算子)，缺字段时 None。
    预期：顶层算子取自 IDS 本身而不是 `property.structure` 整数码——那个码表无文档。
    """
    if not character.radical or character.stroke_total <= 0 or not character.ids_candidates:
        return None
    try:
        operator = top_operator(character.ids_candidates[0])
    except IdsSyntaxError:
        return None
    return (character.radical, character.stroke_total, operator)


def blanked_key(ids: str, position: int) -> str | None:
    """输入：IDS 串 + 叶子下标。输出：把该叶子换成哨兵后的串；解析失败时 None。
    预期：两个「只差一个部件」的字会落到同一个键上，这正是 T12 要的等价类。
    """
    try:
        tree = parse_tree(ids)
    except IdsSyntaxError:
        return None
    if tree.is_leaf or position >= len(leaves(tree)):
        return None
    return render(replace_leaf(tree, position, BLANK))


class CharacterIndex:
    """输入：进字库字序列。输出：只读的检索索引。
    预期：构造是 O(池大小 × 候选数)；所有查询都是 O(1) 或 O(桶大小)，绝不线性扫全池。
    """

    def __init__(self, characters: Iterable[Character]) -> None:
        self.characters: tuple[Character, ...] = tuple(characters)
        self.by_char: dict[str, Character] = {}
        self.by_ids: dict[str, list[Character]] = defaultdict(list)
        self.by_radical: dict[str, list[Character]] = defaultdict(list)
        self.by_stroke: dict[int, list[Character]] = defaultdict(list)
        self.by_bucket: dict[tuple[str, int, str | None], list[Character]] = defaultdict(list)
        self.blank_index: dict[str, list[tuple[Character, int, str]]] = defaultdict(list)

        for character in self.characters:
            if character.zi and character.zi not in self.by_char:
                self.by_char[character.zi] = character
            if character.radical:
                self.by_radical[character.radical].append(character)
            if character.stroke_total > 0:
                self.by_stroke[character.stroke_total].append(character)
            key = bucket_key(character)
            if key is not None:
                self.by_bucket[key].append(character)
            for candidate in character.ids_candidates:
                self.by_ids[candidate].append(character)
                try:
                    count = len(leaves(parse_tree(candidate)))
                except IdsSyntaxError:
                    continue
                for position in range(count):
                    blank = blanked_key(candidate, position)
                    if blank is not None:
                        self.blank_index[blank].append((character, position, candidate))

    # ---------- 查询 ----------

    def lookup_char(self, zi: str) -> Character | None:
        """输入：单个字符。输出：池里的字记录，没有则 None。预期：T9 中间件、T5 异体解析用。"""
        return self.by_char.get(zi)

    def lookup_ids(self, ids: str) -> tuple[Character, ...]:
        """输入：IDS 串。输出：写法命中该串的全部字。预期：T12 判「变换后是不是真实收录字」。"""
        return tuple(self.by_ids.get(ids, ()))

    def same_bucket(self, character: Character) -> tuple[Character, ...]:
        """输入：字记录。输出：与它同 (部首, 笔画, 顶层结构) 的全部字（含自己）。
        预期：T8 判唯一解用——桶里只有一个元素才出题。
        """
        key = bucket_key(character)
        if key is None:
            return ()
        return tuple(self.by_bucket.get(key, ()))

    def same_radical(self, character: Character) -> tuple[Character, ...]:
        """输入：字记录。输出：同部首的其它字。预期：T5/T10 造干扰项与补集用。"""
        if not character.radical:
            return ()
        return tuple(c for c in self.by_radical.get(character.radical, ()) if c.ziid != character.ziid)

    def blank_mates(self, ids: str, position: int, exclude_zi: str) -> tuple[tuple[Character, int, str], ...]:
        """输入：IDS 串 + 叶子下标 + 要排除的字符。输出：挖空后同键的 (字, 下标, 命中的 IDS 候选) 列表。
        预期：这些字就是把该位置换成别的部件后**真实存在**的候选，T12 的答案从这里取。
        """
        blank = blanked_key(ids, position)
        if blank is None:
            return ()
        return tuple((c, pos, cand) for c, pos, cand in self.blank_index.get(blank, ()) if c.zi != exclude_zi)
