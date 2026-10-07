"""题型注册表：题型 ID → 构造器，以及「一个字能出哪些题」的统一入口。

题型编号沿用 synthesis/README.md §6.2 的任务族编号：
T1 字形属性 / T3 IDS 正向 / T4 读音 / T5 字际关系 / T6 文献溯源 / T7 释义收录 /
T8 约束满足 / T9 部件链多跳 / T10 反例否定 / T11 组合排序 / T12 结构变换 / T13 释义反向造字形 /
T14 码位 / T15 UTF-8 编码 / T16 码位反查。
原 T2（两分拆解）因两分/叶码是字海私有编码，已删除。
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable

from ..facts.index import CharacterIndex
from ..facts.records import Character
from .base import Difficulty, TaskSpec
from .glyph import (
    _ASK_MARKERS as _IDS_ASK_MARKERS,
    _STROKE_ASK_MARKERS,
    build_ids_forward,
    build_ids_reverse,
    build_stroke_total,
)
from .reading import _READING_ASK_MARKERS, build_reading
from .reasoning import (
    _CHAIN_ASK_MARKERS,
    _CONSTRAINT_ASK_MARKERS,
    _EXTREME_ASK_MARKERS,
    _ODD_ASK_MARKERS,
    _TRANSFORM_ASK_MARKERS,
    build_component_chain,
    build_constraint,
    build_ids_transform,
    build_odd_one_out,
    build_stroke_extreme,
)
from .relations import (
    _PAGE_ASK_MARKERS,
    _SOURCE_ASK_MARKERS,
    _STANDARD_ASK_MARKERS,
    _VARIANT_ASK_MARKERS,
    build_meaning_source,
    build_reference,
    build_standard_form,
    build_variant,
)
from .unicode import (
    _BYTES_ASK_MARKERS,
    _CODEPOINT_ASK_MARKERS,
    _REVERSE_ASK_MARKERS,
    build_unicode_bytes,
    build_unicode_codepoint,
    build_unicode_reverse,
)

# 构造器签名：字记录 + 随机源 + 难度 + 索引 → TaskSpec 或 None（该字不适合此题）
Constructor = Callable[..., TaskSpec | None]

CONSTRUCTORS: dict[str, Constructor] = {
    "T1_stroke_total": build_stroke_total,
    "T3_ids_forward": build_ids_forward,
    "T4_reading": build_reading,
    "T5_standard_form": build_standard_form,
    "T5_variant": build_variant,
    "T6_zh_ref": build_reference,
    "T7_meaning_source": build_meaning_source,
    "T8_constraint": build_constraint,
    "T9_component_chain": build_component_chain,
    "T10_odd_one_out": build_odd_one_out,
    "T11_stroke_extreme": build_stroke_extreme,
    "T12_ids_transform": build_ids_transform,
    "T13_ids_reverse": build_ids_reverse,
    "T14_unicode_codepoint": build_unicode_codepoint,
    "T15_unicode_bytes": build_unicode_bytes,
    "T16_unicode_reverse": build_unicode_reverse,
}


# 题型 → 该题型**当前**声明的题干闸门（`TaskSpec.stem_asks_any`）。
#
# 为什么要有这张表：闸门是**生产时**随记录落盘的，`_offline_record` 写的是当时的
# `spec.stem_asks_any`。旧构造器压根没声明闸门的题型，其历史记录的该字段是**空元组**
# ——而 `stem_asks` 那一关在空元组时整条跳过。于是复验（`audit` / 回炉）会拿一个
# 「没有闸门」的 spec 去查，把「题干问读音、答案是笔画数」这种废题判成 tier A。
# 实测：597 条已发布产出里就有 2 条这样的废题（T1 U0340A / T10 U03400）。
#
# 闸门表是**题型级常量**（不随字变），所以能按 task_id 直接补上；`must_contain` /
# `structure_words` / `hidden_subject` 是**字级**的（T9 的位置路径、T13 的隐藏主语），
# 补不出来——那三项要靠按 `provenance.rng_seed` 重建 spec 才能复算，不在这张表的范围内。
#
# 表与构造器**不许漂移**：`src/tests/test_tasks.py::StemAsksGateCoverageTests` 拿一个
# 覆盖全部 16 个题型的假字池，逐题型断言「构造器实际挂上的闸门 == 这张表」。
TASK_STEM_ASKS: dict[str, tuple[str, ...]] = {
    "T1_stroke_total": _STROKE_ASK_MARKERS,
    "T3_ids_forward": _IDS_ASK_MARKERS,
    "T4_reading": _READING_ASK_MARKERS,
    "T5_standard_form": _STANDARD_ASK_MARKERS,
    "T5_variant": _VARIANT_ASK_MARKERS,
    "T6_zh_ref": _PAGE_ASK_MARKERS,
    "T7_meaning_source": _SOURCE_ASK_MARKERS,
    "T8_constraint": _CONSTRAINT_ASK_MARKERS,
    "T9_component_chain": _CHAIN_ASK_MARKERS,
    "T10_odd_one_out": _ODD_ASK_MARKERS,
    "T11_stroke_extreme": _EXTREME_ASK_MARKERS,
    "T12_ids_transform": _TRANSFORM_ASK_MARKERS,
    "T13_ids_reverse": _IDS_ASK_MARKERS,
    "T14_unicode_codepoint": _CODEPOINT_ASK_MARKERS,
    "T15_unicode_bytes": _BYTES_ASK_MARKERS,
    "T16_unicode_reverse": _REVERSE_ASK_MARKERS,
}


def build_all(
    character: Character,
    *,
    rng: random.Random,
    index: CharacterIndex,
    difficulty: Difficulty | None = None,
    only: tuple[str, ...] | None = None,
) -> list[TaskSpec]:
    """输入：字记录 + 随机源 + 索引 + 难度 + 可选题型白名单。
    输出：该字能构造出的全部 TaskSpec（不适用的题型被静默跳过）。
    预期：构造器抛异常不吞——出题阶段出错要立刻暴露，不能静默丢数据。
          `difficulty.distractor_depth == "mixed"` 时先按分布落成具体档位（每个字抽一次），
          再交给构造器，并把落定结果写进 `TaskSpec.resolved_depth`——
          这样产出里的难度注记写的是真实档位，而不是配置里的 `mixed`。
    """
    difficulty = difficulty or Difficulty()
    effective = difficulty.resolved(rng)
    specs: list[TaskSpec] = []
    for task_id, constructor in CONSTRUCTORS.items():
        if only is not None and task_id not in only:
            continue
        spec = constructor(character, rng=rng, difficulty=effective, index=index)
        if spec is not None:
            specs.append(replace(spec, resolved_depth=effective.distractor_depth))
    return specs
