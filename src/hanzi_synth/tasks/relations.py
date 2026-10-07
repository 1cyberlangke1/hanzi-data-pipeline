"""关系与溯源类题目构造器：T5 字际关系 / T6 文献溯源 / T7 释义收录辨析。

数据来源与边界（全部是**字段直接值**，不做跨字段推导）：
- T5 ← `standard_form`（标准字形）与 `variants`（异体字），异体码若是 `U+XXXX` 形式按码位还原；
- T6 ← `property.zh_ref` 的 page/number。**只出中华字海**——`hd_ref`/`tg_ref`/`bf_ref`
  三个字段形状相同但书名在 crawler/README.md 里没有记载，给它们编书名就是幻觉；
- T7 ← `meanings` 的分库字段。只认三本**书名有据可查**的库：mean_h 中华字海 /
  mean_k 康熙字典 / mean_j 集韵；其余标签（综合/补遗/异体/古文字）不是书名，不参与出题。

T7 是原设计「跨库说法差异」的**保守降级**：真正的差异辨析需要比对两库释义文本，
而释义文本既不能照抄（侵权）又已砍掉外部交叉源，所以退化成「哪部字典收录了这个字的释义」
——存在性用字段直接判，答案不含任何释义原文。
"""

from __future__ import annotations

import random

from ..facts.index import CharacterIndex
from ..facts.records import Character
from .base import Difficulty, Distractor, TaskSpec, provenance as make_provenance
from .distractors import char_distractors, numeric_distractors

CONSTRUCTOR_VERSION = "0.1.0"

# 唯一书名有据可查的 ref 字段（crawler/README.md: zh_ref = 中华字海页码/序号）
_ZH_DICT = "中华字海"

# 释义库标签 → 书名（只取书名明确的三个）
_MEANING_SOURCES: dict[str, str] = {
    "mean_h": "中华字海",
    "mean_k": "康熙字典",
    "mean_j": "集韵",
}


# **题干闸门**：题干至少要命中一个「问那件事」的说法（判据在 L4 的 `stem_asks`）。
# 闸门空着 = 不设防：实测第十轮真机的 T1 U0340A 题干问的是读音与部首、答案却是笔画数，
# 因为 T1 原先没声明 `stem_asks_any`，那一关整条跳过。以下四条各按「答案是什么」列同义说法，
# 只收题型真正会问的那几件事，宁可漏（换了说法的产出照样发布）也不误杀。
# 代价（549 条已发布产出逐条量）：五条里只有 T5_standard_form 需要补「写法」（实测有产出写
# 「请问「㐗」的正确写法是什么？」），其余 **0 条**被拒收。
_STANDARD_ASK_MARKERS = ("标准字形", "标准字", "正体", "正字", "规范写法", "标准写法", "写法", "字形")
_VARIANT_ASK_MARKERS = ("异体字", "异体", "别体", "或体")
_PAGE_ASK_MARKERS = ("页", "页码", "收录")
_SOURCE_ASK_MARKERS = ("哪一部", "哪部", "哪一本", "哪本", "字典", "词典", "工具书", "收录", "释义")


def build_standard_form(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T5a：问这个字的标准字形（正字）是哪个字。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；无码位、无 standard_form、或标准字形与自己相同时返回 None。
    预期：答案 = `standard_form` 字段直接值；干扰项按 distractor_depth 从池里取。
    """
    if not character.zi or not character.standard_form:
        return None
    if character.standard_form == character.zi or len(character.standard_form) != 1:
        return None
    answer = character.standard_form
    return TaskSpec(
        task_id="T5_standard_form",
        instruction=(
            f"Ask the reader which character is the standard form of 「{character.zi}」."
        ),
        facts={
            "zi": character.zi,
            "standard_form": answer,
            "radical": character.radical,
            "stroke_total": character.stroke_total,
        },
        answer=answer,
        visible_subject=character.zi,
        stem_asks_any=_STANDARD_ASK_MARKERS,
        distractors=char_distractors(
            index,
            character,
            [character.zi, answer],
            rng,
            depth=difficulty.distractor_depth,
            why_wrong=lambda other: f"「{other.zi}」不是「{character.zi}」的标准字形",
        ),
        proof=(f"这个字的标准字形是「{answer}」",),
        provenance=make_provenance(character, "T5_standard_form", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def build_variant(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T5b：问这个字的异体字。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；无码位、或没有可书写的单字异体时返回 None。
    预期：答案集合 = 全部单字异体（多解全算对）；干扰项保证**不是**该字的异体。
    """
    if not character.zi:
        return None
    variants = [v for v in character.variants if len(v) == 1 and v != character.zi]
    if not variants:
        return None
    banned = [character.zi, *variants]
    return TaskSpec(
        task_id="T5_variant",
        instruction=(
            f"Ask the reader which character is a variant form of 「{character.zi}」."
        ),
        facts={
            "zi": character.zi,
            "variants": list(variants),
            "standard_form": character.standard_form,
        },
        answer=variants[0],
        visible_subject=character.zi,
        answer_aliases=tuple(variants[1:]),
        stem_asks_any=_VARIANT_ASK_MARKERS,
        distractors=char_distractors(
            index,
            character,
            banned,
            rng,
            depth=difficulty.distractor_depth,
            why_wrong=lambda other: f"「{other.zi}」与「{character.zi}」没有异体关系",
        ),
        proof=(f"「{variants[0]}」是「{character.zi}」的异体字",),
        provenance=make_provenance(character, "T5_variant", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def build_reference(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T6：问这个字在《中华字海》里收录于第几页。
    输入：字记录 + 随机源 + 难度 + 索引（未用）。
    输出：TaskSpec；无码位或没有 zh_ref 时返回 None。
    预期：答案 = `refs['zh']['page']` 字段直接值；干扰页码按 distractor_depth 取邻近值。
    """
    if not character.zi:
        return None
    ref = character.refs.get("zh")
    if not ref:
        return None
    page = int(ref.get("page", 0))
    if page <= 0:
        return None
    return TaskSpec(
        task_id="T6_zh_ref",
        instruction=(
            f"Ask the reader on which page of the dictionary 《{_ZH_DICT}》 the character "
            f"「{character.zi}」 is listed."
        ),
        facts={
            "zi": character.zi,
            "dictionary": _ZH_DICT,
            "page": page,
            "number": int(ref.get("number", 0)),
        },
        answer=str(page),
        visible_subject=character.zi,
        # 书名是题干的识别条件：题干不写它，读者不知道问的是哪部字典的页码
        must_contain=(_ZH_DICT,),
        stem_asks_any=_PAGE_ASK_MARKERS,
        distractors=numeric_distractors(
            page, rng, depth=difficulty.distractor_depth, why=lambda v: f"该字在《{_ZH_DICT}》第 {page} 页，不是第 {v} 页"
        ),
        proof=(f"这个字在《{_ZH_DICT}》里收录于第 {page} 页",),
        provenance=make_provenance(character, "T6_zh_ref", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def build_meaning_source(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T7：问这个字的释义收录在哪部字典里（跨库收录辨析）。
    输入：字记录 + 随机源 + 难度 + 索引（未用）。
    输出：TaskSpec；无码位、或三本库里**不是恰好一本**有释义时返回 None。
    预期：答案 = 书名（字段存在性直接判定，不含任何释义原文）；选项固定为这三本书名，
          并走 `options` 字段声明成**选择题**——只写进 facts 时 L4 会当成「答案不得出现在题干」，
          而模型天然会列出候选书名，好数据会被误拒（实测 16f 就是这样被拒的）。
    """
    if not character.zi:
        return None
    present = [name for label, name in _MEANING_SOURCES.items() if character.meanings.get(label)]
    if len(present) != 1:
        return None  # 0 本 = 没材料；多本 = 题目有歧义
    answer = present[0]
    distractors = tuple(
        Distractor(text=name, why_wrong=f"《{name}》没有收录「{character.zi}」的释义")
        for name in _MEANING_SOURCES.values()
        if name != answer
    )
    return TaskSpec(
        task_id="T7_meaning_source",
        instruction=(
            f"Ask the reader which of the listed candidate dictionaries contains the definition of "
            f"「{character.zi}」."
        ),
        facts={"zi": character.zi, "source": answer, "options": list(_MEANING_SOURCES.values())},
        answer=answer,
        options=tuple(_MEANING_SOURCES.values()),
        visible_subject=character.zi,
        stem_asks_any=_SOURCE_ASK_MARKERS,
        distractors=distractors,
        proof=(f"「{character.zi}」的释义收录在《{answer}》里",),
        provenance=make_provenance(character, "T7_meaning_source", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )
