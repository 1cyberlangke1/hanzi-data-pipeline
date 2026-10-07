"""测试辅助：构造假的 Character 与索引，避免测试依赖真实数据文件（原始数据不入库）。"""

from __future__ import annotations

from typing import Iterable, Sequence

from hanzi_synth.facts.index import CharacterIndex
from hanzi_synth.facts.records import Character


def make_character(
    ziid: str = "U0597D",
    zi: str = "\u597d",
    ids: Sequence[str] = ("\u2ff0\u5973\u5b50",),
    stroke: int = 6,
    radical: str | None = "\u5973",
    standard_form: str | None = None,
    meanings: dict | None = None,
    variants: Sequence[str] = (),
    refs: dict | None = None,
) -> Character:
    """输入：各字段（都有默认值）。输出：一个 Character 实例，供构造器测试使用。
    预期：structure 一律留 None——生产代码已明确不依赖那个无文档的整数码。
    """
    return Character(
        ziid=ziid,
        zi=zi,
        ids_candidates=tuple(ids),
        stroke_total=stroke,
        radical=radical,
        structure=None,
        standard_form=standard_form,
        meanings=meanings or {},
        variants=tuple(variants),
        refs=refs or {},
    )


def make_index(characters: Iterable[Character]) -> CharacterIndex:
    """输入：字序列。输出：索引。预期：测试里显式构造，绝不读磁盘。"""
    return CharacterIndex(list(characters))
