"""释义类构造器：T17 问这个字是什么意思。

答案 = 从字典原文切出的义项（`facts.gloss.gloss_items`，可逐条回查原文），
不是模型改写的一段话——所以「答案对不对」是可核对的：产出里必须原样出现义项。
模型只负责把「问什么 + 答案」改写成自然语言（L3）。
"""

from __future__ import annotations

import random

from ..facts.gloss import gloss_items
from ..facts.index import CharacterIndex
from ..facts.records import Character
from .base import Difficulty, TaskSpec, provenance as make_provenance

CONSTRUCTOR_VERSION = "0.1.0"

# 题干闸门：题干必须命中一个「问意思」的说法，否则读者不知道要答什么。
_GLOSS_ASK_MARKERS = (
    "意思",
    "含义",
    "释义",
    "解释",
    "讲的是什么",
    "表示什么",
    "说的是什么",
)


def build_meaning_gloss(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T17：问这个字是什么意思。
    输入：字记录 + 随机源 + 难度 + 索引（未用）。
    输出：TaskSpec；没有可用义项的字返回 None。
    预期：答案 = 字典里该字的**第一条义项**，其余义项进 `answer_aliases`（多义全算对）；
          `facts` 里给出全部义项与出处书名，模型据此改写措辞；原始整段释义不进载荷。
    """
    if not character.zi:
        return None
    items, source = gloss_items(character)
    if not items:
        return None
    return TaskSpec(
        task_id="T17_meaning_gloss",
        # **不给** `source`：书名一进载荷，2B 就把它写进产出——实测 3,401 条 LLM 产出里 1,174 条
        # 带「根据康熙字典 / 根据《中华字海》的释义 / 出自《中华字海》」。读者手上并没有那份资料，
        # 这是把出题人视角的话漏给了读者（与 `verify.checks._INTERNAL_TERMS` 挡的是同一类）。
        # 出处仍然留在 `facts.source` 里供溯源，只是不进提示词。
        payload_facts=("zi", "gloss_items"),
        instruction=(
            f"「{character.zi}」的意思"
        ),
        facts={
            "zi": character.zi,
            "source": source,
            "gloss_items": list(items),
            "radical": character.radical,
            "stroke_total": character.stroke_total,
        },
        answer=items[0],
        answer_aliases=tuple(items[1:]),
        visible_subject=character.zi,
        stem_asks_any=_GLOSS_ASK_MARKERS,
        proof=(f"「{character.zi}」的释义：{'；'.join(items)}",),
        provenance=make_provenance(
            character, "T17_meaning_gloss", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)
        ),
    )
