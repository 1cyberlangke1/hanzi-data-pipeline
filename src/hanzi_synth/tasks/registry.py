"""题型注册表：题型 ID → 构造器，以及「一个字能出哪些题」的统一入口。

题型编号沿用任务族编号：
T1 字形属性 / T3 IDS 正向 / T4 读音 / T5 字际关系 / T6 文献溯源 /
T8 约束满足 / T9 部件链多跳 / T10 反例否定 / T11 组合排序 / T12 结构变换 / T13 释义反向造字形 /
T14 码位 / T15 UTF-8 编码 / T16 码位反查 / T17 释义（答案 = 字典原文切出的义项）。
原 T2（两分拆解）因两分/叶码是字海私有编码，已删除。
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable

from ..facts.index import CharacterIndex
from ..facts.records import Character
from .base import CONSTRAINT_TASKS, HOP_TASKS, Difficulty, TaskSpec
from .glyph import (
    _ASK_MARKERS as _IDS_ASK_MARKERS,
    _STROKE_ASK_MARKERS,
    build_ids_forward,
    build_ids_reverse,
    build_stroke_total,
)
from .reading import _READING_ASK_MARKERS, build_reading
from .meaning import _GLOSS_ASK_MARKERS, build_meaning_gloss
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
    _STANDARD_ASK_MARKERS,
    _VARIANT_ASK_MARKERS,
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
    "T8_constraint": build_constraint,
    "T9_component_chain": build_component_chain,
    "T10_odd_one_out": build_odd_one_out,
    "T11_stroke_extreme": build_stroke_extreme,
    "T12_ids_transform": build_ids_transform,
    "T13_ids_reverse": build_ids_reverse,
    "T14_unicode_codepoint": build_unicode_codepoint,
    "T15_unicode_bytes": build_unicode_bytes,
    "T16_unicode_reverse": build_unicode_reverse,
    "T17_meaning_gloss": build_meaning_gloss,
}


# 题型 → 该题型**当前**声明的题干校验（`TaskSpec.stem_asks_any`）。
#
# 为什么要有这张表：校验是**生产时**随记录写入文件的，`_offline_record` 写的是当时的
# `spec.stem_asks_any`。旧构造器压根没声明校验的题型，其历史记录的该字段是**空元组**
# ——而 `stem_asks` 那一关在空元组时整条跳过。于是复验（`audit` / 重跑）会拿一个
# 「没有校验」的 spec 去查，把「题干问读音、答案是笔画数」这种废题判成 tier A。
# 实测：597 条已发布产出里就有 2 条这样的废题（T1 U0340A / T10 U03400）。
#
# 校验表是**题型级常量**（不随字变），所以能按 task_id 直接补上；`must_contain` /
# `structure_words` / `hidden_subject` 是**字级**的（T9 的位置路径、T13 的隐藏主语），
# 补不出来——那三项要靠按 `provenance.rng_seed` 重建 spec 才能复算，不在这张表的范围内。
#
# 表与构造器**不许漂移**：`src/tests/test_tasks.py::StemAsksGateCoverageTests` 拿一个
# 覆盖全部 16 个题型的假字池，逐题型断言「构造器实际挂上的校验 == 这张表」。
TASK_STEM_ASKS: dict[str, tuple[str, ...]] = {
    "T1_stroke_total": _STROKE_ASK_MARKERS,
    "T3_ids_forward": _IDS_ASK_MARKERS,
    "T4_reading": _READING_ASK_MARKERS,
    "T5_standard_form": _STANDARD_ASK_MARKERS,
    "T5_variant": _VARIANT_ASK_MARKERS,
    "T6_zh_ref": _PAGE_ASK_MARKERS,
    "T8_constraint": _CONSTRAINT_ASK_MARKERS,
    "T9_component_chain": _CHAIN_ASK_MARKERS,
    "T10_odd_one_out": _ODD_ASK_MARKERS,
    "T11_stroke_extreme": _EXTREME_ASK_MARKERS,
    "T12_ids_transform": _TRANSFORM_ASK_MARKERS,
    "T13_ids_reverse": _IDS_ASK_MARKERS,
    "T14_unicode_codepoint": _CODEPOINT_ASK_MARKERS,
    "T15_unicode_bytes": _BYTES_ASK_MARKERS,
    "T16_unicode_reverse": _REVERSE_ASK_MARKERS,
    "T17_meaning_gloss": _GLOSS_ASK_MARKERS,
}


def _difficulty_ladder(effective: Difficulty, task_id: str) -> tuple[Difficulty, ...]:
    """输入：落定后的难度 + 题型。输出：该题型要**依次尝试**的难度序列（从抽到的档位退到最低档）。
    预期：只对真正消费该旋钮的题型展开阶梯。构造器在抽到的档位上凑不出题（链接不起来 /
          判不出唯一解）时退到下一档再试，而不是让这个字**整型消失**——覆盖不缩水，
          同时数据集里真的有多跳 / 多约束的难题。退到最低档仍凑不出才是真的不适配。
          不消费旋钮的题型只给一档，绝不把构造器白调 3 次（那会白花时间、还会多消耗随机数）。
    """
    if task_id in HOP_TASKS:
        return tuple(replace(effective, hops=level) for level in range(effective.hops, 0, -1))
    if task_id in CONSTRAINT_TASKS:
        return tuple(replace(effective, constraints=level) for level in range(effective.constraints, 0, -1))
    return (effective,)


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
          `hops` / `constraints` / `distractor_depth` 为 `mixed` 时先按分布落成具体值（每个字抽一次），
          再交给构造器，并把落定结果写进 `TaskSpec.resolved_*`——
          这样产出里的难度注记写的是真实值，而不是配置里的 `mixed`。
          抽到的档位凑不出题时按 `_difficulty_ladder` 退档重试。
    """
    difficulty = difficulty or Difficulty()
    effective = difficulty.resolved(rng)
    specs: list[TaskSpec] = []
    for task_id, constructor in CONSTRUCTORS.items():
        if only is not None and task_id not in only:
            continue
        for level in _difficulty_ladder(effective, task_id):
            spec = constructor(character, rng=rng, difficulty=level, index=index)
            if spec is None:
                continue
            specs.append(
                replace(
                    spec,
                    resolved_depth=effective.distractor_depth,
                    resolved_hops=level.hops if task_id in HOP_TASKS else 1,
                    resolved_constraints=level.constraints if task_id in CONSTRAINT_TASKS else 1,
                )
            )
            break
    return specs
