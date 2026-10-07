"""读音类题目构造器：T4 从释义标注里抽读音。

读音在本库**不是独立字段**（叶码一族按 findings.md §9.6 已弃用），
唯一来源是释义文本里的「拼音xxx1」标注 —— 用**确定性正则**抽取，不是模型猜测。
抽取不到读音的字直接不出读音题（README 铁律 7：不能验证的宁可不做）。
"""

from __future__ import annotations

import random
import re

from ..facts.index import CharacterIndex
from ..facts.records import Character
from ..facts.surface import reading_surfaces
from .base import Difficulty, Distractor, TaskSpec, provenance as make_provenance, resolve_depth

CONSTRUCTOR_VERSION = "0.2.0"

# 释义里的读音标注形如「拼音ling2」「拼音 xing1」；声调可能缺省
_PINYIN_RE = re.compile(r"拼音\s*([a-zA-Z\u00fc\u00ea]+)([1-5])?")

# 常见韵母归一：ü / ê 在 canonical 层写成 ASCII，保证可索引、可 round-trip
_FOLD = str.maketrans({"\u00fc": "v", "\u00ea": "e"})

# far 档用的无关音节池：只是字符串素材，不是从字典里取的读音表
_FILLER_SYLLABLES: tuple[str, ...] = ("ma", "shi", "yu", "he", "lan", "qiu", "teng", "mian", "rong", "zhao")


def canonical_reading(syllable: str, tone: str | None) -> str:
    """输入：音节与声调数字（可缺省）。输出：canonical 读音，如 `hao4` / `lv4`。
    预期：canonical 层只用于判等与去重；surface 层展示时可另做还原。
    """
    return syllable.lower().translate(_FOLD) + (tone or "")


def extract_readings(text: str) -> tuple[str, ...]:
    """输入：释义文本。输出：按出现顺序去重的 canonical 读音元组。
    预期：没有标注时返回空元组——调用方据此不出读音题。
    """
    out: list[str] = []
    for match in _PINYIN_RE.finditer(text or ""):
        reading = canonical_reading(match.group(1), match.group(2))
        if reading and reading not in out:
            out.append(reading)
    return tuple(out)


def _reading_distractors(answer: str, rng: random.Random, depth: str) -> tuple[Distractor, ...]:
    """输入：正确读音、随机源、难度档。输出：3 个干扰读音。
    预期：near 只差声调（最难）、mid 同声调不同音节、far 无关音节；不足时按档放宽，绝不重复。
    """
    depth = resolve_depth(depth, rng)
    body = answer[:-1] if answer[-1:].isdigit() else answer
    tone = answer[-1] if answer[-1:].isdigit() else ""
    tiers: dict[str, list[str]] = {"near": [], "mid": [], "far": []}
    if tone:
        for delta in (1, -1, 2):
            value = int(tone) + delta
            if 1 <= value <= 4:
                tiers["near"].append(f"{body}{value}")
    for syllable in _FILLER_SYLLABLES:
        tiers["far"].append(f"{syllable}{tone or 1}")
        if tone:
            tiers["mid"].append(f"{syllable}{tone}")

    order = ("near", "mid", "far")
    start = order.index(depth) if depth in order else 0
    picked: list[str] = []
    for tier in order[start:]:
        options = list(tiers[tier])
        rng.shuffle(options)
        for value in options:
            if value != answer and value not in picked:
                picked.append(value)
            if len(picked) == 3:
                break
        if len(picked) == 3:
            break
    return tuple(Distractor(text=v, why_wrong=f"该字读 {answer}，不是 {v}") for v in picked)


# T4 的题干闸门：答案是读音，题干至少要命中一个「问读音」的说法。
# 代价：549 条已发布产出里 **0 条**被它拒收。
_READING_ASK_MARKERS = ("读音", "读作", "怎么读", "念作", "读法", "拼音", "发音")


def build_reading(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T4：问这个字怎么读。
    输入：字记录 + 随机源 + 难度 + 索引（未用）。
    输出：TaskSpec；无码位、无释义、抽不到读音、或**多音字**时返回 None。
    预期：答案 = 释义标注里抽出的唯一读音；多音会让题目有歧义，故直接不出。
    """
    if not character.zi:
        return None
    readings: list[str] = []
    for block in character.meaning_blocks():
        for reading in extract_readings(block):
            if reading not in readings:
                readings.append(reading)
    if len(readings) != 1:
        return None  # 0 个 = 没材料；多个 = 多音字，题目有歧义
    answer = readings[0]
    variants = reading_surfaces(answer)
    shown = variants[rng.randrange(len(variants))]
    return TaskSpec(
        task_id="T4_reading",
        instruction=(
            "Ask the reader how the character shown in stem_must_contain is pronounced."
        ),
        facts={"zi": character.zi, "reading": answer, "reading_display": shown, "radical": character.radical},
        # 题干闸门：答案是**读音**，题干必须问读音（同 T1 的理由——闸门空着等于不设防）
        stem_asks_any=_READING_ASK_MARKERS,
        answer=answer,
        surface_forms=variants[1:],
        visible_subject=character.zi,
        distractors=_reading_distractors(answer, rng, difficulty.distractor_depth),
        proof=(f"这个字的读音是 {shown}",),
        provenance=make_provenance(character, "T4_reading", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )
