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
def full_coverage_pool() -> list:
    """输入：无。输出：一批**能覆盖全部 16 个题型**的假字记录 + 索引。

    预期：这批字是给「每个题型都必须声明题干校验」这条不变式用的——覆盖不全，
          校验漏装也测不出来（第十轮 T12 的 `_TRANSFORM_ASK_MARKERS` 定义了却没接上，
          就是这么漏的）。各字的取材理由：
          - 㐀：有 IDS / 部首 / 标准字形 / 异体 / 页码，覆盖字形与字际关系一族；
          - 好：释义里带拼音且清洗后够长，T4 才有唯一读音可出、T13 才有识别条件；
          - 一 / 丁 / 七 / 三 / 上：同一部首、笔画互有重叠，给 T8 约束桶与 T10/T11 凑同伴；
          - 㐄：部首「一」+ 5 画在池里独一份，T8 的唯一解条件才成立。
    """
    return [
        make_character(
            zi="\u3400",
            ziid="U03400",
            ids=("\u2ff0\u4e00\u4e01",),
            stroke=3,
            radical="\u4e00",
            standard_form="\u4e00",
            variants=("\u4e01",),
            meanings={"mean_h": ("\u53e4\u540c\u300c\u4e00\u300d\u3002",)},
            refs={"zh": {"page": 12, "number": 3}, "meaning_src": {"\u5b57\u6d77": 1}},
        ),
        make_character(zi="\u4e01", ziid="U04E01", ids=("\u2ff0\u4e00\u4e01",), stroke=2, radical="\u4e00",
                       meanings={"mean_h": ("\u5929\u5e72\u7b2c\u56db\u4f4d\u3002",)}),
        make_character(zi="\u4e03", ziid="U04E03", ids=("\u2ff0\u4e00\u4e01",), stroke=2, radical="\u4e00",
                       meanings={"mean_h": ("\u6570\u76ee\u3002",)}),
        make_character(zi="\u4e09", ziid="U04E09", ids=("\u2ff0\u4e00\u4e01",), stroke=3, radical="\u4e00",
                       meanings={"mean_h": ("\u6570\u76ee\u3002",)}),
        make_character(zi="\u4e0a", ziid="U04E0A", ids=("\u2ff0\u4e00\u4e01",), stroke=3, radical="\u4e00",
                       meanings={"mean_h": ("\u6570\u76ee\u3002",)}),
        make_character(zi="\u597d", ziid="U0597D", ids=("\u2ff0\u5973\u5b50",), stroke=6, radical="\u5973",
                       meanings={"mean_h": ("\u62fc\u97f3hao3<br>\u6307\u4e8b\u7269\u4ee4\u4eba\u6ee1\u610f\u3002",)}),
        make_character(zi="\u5982", ziid="U05982", ids=("\u2ff0\u5973\u53e3",), stroke=6, radical="\u5973",
                       meanings={"mean_h": ("\u987a\u4ece\u3002",)}),
        make_character(zi="\u59d0", ziid="U059D0", ids=("\u2ff0\u5973\u4e14",), stroke=8, radical="\u5973",
                       meanings={"mean_h": ("\u59d0\u59d0\u3002",)}),
        make_character(zi="\u59b9", ziid="U059B9", ids=("\u2ff0\u5973\u672a",), stroke=8, radical="\u5973",
                       meanings={"mean_h": ("\u59b9\u59b9\u3002",)}),
        make_character(zi="\u3404", ziid="U03404", ids=("\u2ff0\u4e00\u4e01",), stroke=5, radical="\u4e00",
                       meanings={"mean_h": ("\u53e4\u5b57\u3002",)}),
    ]
