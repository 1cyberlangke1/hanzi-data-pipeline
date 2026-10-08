"""干扰项工厂：把 `Difficulty.distractor_depth` 变成真实可测的差异。

三档的语义（难度由「像不像」决定，不由措辞长度决定）：
- `near`：同部首且笔画差 <= 2 —— 最像，最难；
- `mid` ：同总笔画但不同部首 —— 中等；
- `far` ：全池随机 —— 最不像，最易。
- `mixed`：**不是第四个距离，而是分布档**——按 4:3:2 抽 near/mid/far，
  让难度在数据集层面成分布。正常路径由 `build_all` 先把 mixed 落成具体档位，
  这里的 `resolve_depth` 是直接调用本模块时的保底（不会与上游抽两次）。

字库不够时**按 near -> mid -> far 放宽**，并在 `why_wrong` 里如实说明干扰项与答案的区别；
绝不为了凑数重复选项。
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from ..facts.index import CharacterIndex
from ..facts.records import Character
from .base import Distractor, resolve_depth

# 数值干扰项的偏移量按档位取：near 只差 1~2，mid 差 3~5，far 差 6~15
_NUMERIC_OFFSETS: dict[str, tuple[int, ...]] = {
    "near": (1, -1, 2, -2),
    "mid": (3, -3, 4, -4, 5, -5),
    "far": (6, -6, 9, -9, 12, -12, 15, -15),
}

# 候选池抽样上界：far 档的池是全池（7.9 万条），整表 shuffle 太贵，改成先均匀抽样
_SAMPLE_CAP = 200


def numeric_distractors(
    answer: int,
    rng: random.Random,
    *,
    depth: str = "near",
    count: int = 3,
    why: Callable[[int], str] | None = None,
) -> tuple[Distractor, ...]:
    """输入：正确答案、随机源、难度档、条数、错误说明模板。输出：数值干扰项。
    预期：偏移量按档取且互异；结果恒 > 0；不足时自动放宽到下一档，绝不重复。
    """
    depth = resolve_depth(depth, rng)
    order = ("near", "mid", "far")
    start = order.index(depth) if depth in order else 0
    picked: list[int] = []
    for tier in order[start:]:
        offsets = list(_NUMERIC_OFFSETS[tier])
        rng.shuffle(offsets)
        for offset in offsets:
            value = answer + offset
            if value > 0 and value != answer and value not in picked:
                picked.append(value)
            if len(picked) >= count:
                break
        if len(picked) >= count:
            break
    explain = why or (lambda value: f"正确答案是 {answer}，不是 {value}")
    return tuple(Distractor(text=str(value), why_wrong=explain(value)) for value in picked[:count])


def _candidate_pool(index: CharacterIndex, target: Character, depth: str) -> list[Character]:
    """输入：索引、目标字、难度档。输出：该档的候选字列表。
    预期：只按字段做确定性筛选（部首/笔画），不做任何语义相似度推断。
    """
    if depth == "near" and target.radical:
        return [
            c
            for c in index.same_radical(target)
            if c.zi and c.ids_candidates and abs(c.stroke_total - target.stroke_total) <= 2
        ]
    if depth == "mid":
        return [
            c
            for c in index.characters
            if c.zi and c.ids_candidates and c.stroke_total == target.stroke_total and c.radical != target.radical
        ]
    return [c for c in index.characters if c.zi and c.ids_candidates]


def char_distractors(
    index: CharacterIndex,
    target: Character,
    banned: Sequence[str],
    rng: random.Random,
    *,
    depth: str = "near",
    why_wrong: Callable[[Character], str],
    text_of: Callable[[Character], str] | None = None,
    count: int = 3,
) -> tuple[Distractor, ...]:
    """输入：索引、目标字、禁用文本（答案+目标字）、随机源、难度档、错误说明模板、
             取文本函数（默认取字符本身，传 IDS 版可取 `ids_candidates[0]`）、条数。
    输出：字符类干扰项。
    预期：优先用指定档的池；不够时按 near -> mid -> far 放宽；选项文本绝不重复、绝不落在 banned 里。
    """
    depth = resolve_depth(depth, rng)
    render = text_of or (lambda c: c.zi)
    order = ("near", "mid", "far")
    start = order.index(depth) if depth in order else 0
    seen: set[str] = {text for text in banned if text}
    picked: list[Distractor] = []
    for tier in order[start:]:
        pool = _candidate_pool(index, target, tier)
        if len(pool) > _SAMPLE_CAP:
            pool = rng.sample(pool, _SAMPLE_CAP)
        rng.shuffle(pool)
        for candidate in pool:
            text = render(candidate)
            if not text or text in seen:
                continue
            seen.add(text)
            picked.append(Distractor(text=text, why_wrong=why_wrong(candidate)))
            if len(picked) >= count:
                return tuple(picked)
    return tuple(picked)
