"""码位与编码类构造器：T14 码位 / T15 UTF-8 编码 / T16 码位反查。

主人要求「如果这个字有 unicode，也要让模型学习 unicode 本身和编码」，
所以这三类题**只对 U 组（有码位）出**，并且把同一码位的多种等价写法都算对
（`surface.codepoint_surfaces`：`U+4E00` / `4E00` / `19968` / `E4 B8 80` / 代理对）。
主答案由 rng 在等价写法之间轮换，让 hex、十进制、字节三种形态在数据集层面**成分布**，
而不是所有条目都写成同一种格式。

Z/P 组无码位字（`zi` 为空）一律返回 None——它们没有可书写的码位，硬出题只能造错数据。
"""

from __future__ import annotations

import random

from ..facts.index import CharacterIndex
from ..facts.records import Character
from ..facts.surface import codepoint_surfaces
from .base import Difficulty, Distractor, TaskSpec, provenance as make_provenance
from .distractors import char_distractors

CONSTRUCTOR_VERSION = "0.1.0"

# 码位干扰项的偏移量按难度档取：near 只差 1~2，mid 差 16~32，far 差 256~4096
_CODE_OFFSETS: dict[str, tuple[int, ...]] = {
    "near": (1, -1, 2, -2),
    "mid": (0x10, -0x10, 0x20, -0x20),
    "far": (0x100, -0x100, 0x1000, -0x1000),
}

_MAX_CODE = 0x10FFFF


def _utf8_of(char: str) -> str:
    """输入：单个字符。输出：UTF-8 字节的大写十六进制（空格分隔）。
    预期：直接取 codepoint_surfaces 的第 4 条，不在这里重复实现编码。
    """
    return codepoint_surfaces(char)[3]


def _value_forms(char: str) -> tuple[str, ...]:
    """输入：单个字符。输出：**自明**的码位数值写法——带 `U+` 前缀的十六进制、十进制。

    预期：`codepoint_surfaces` 的第 2 条是**裸十六进制**（`3406`），它和十进制写法
          （`13318`）在文本里长得一模一样，读者按十进制读会读到另一个码位（3406 = U+0D4E）。
          实测全量离线数据里，T14 有 15,085 条解析写成「㐆的 Unicode 码位是 3406」、
          T16 有 45,157 条题干写成「码位3406对应的字」——同一个词「码位」在一份数据集里
          时而指十六进制时而指十进制，是**自相矛盾的数据**，所以裸十六进制不进主答案、
          也不进题干（它仍然是合法答案，见 `build_unicode_codepoint` 的 aliases）。
    """
    forms = codepoint_surfaces(char)
    return (forms[0], forms[2])


def _pick_form(
    char: str, rng: random.Random, *, forms: tuple[str, ...] | None = None
) -> tuple[str, tuple[str, ...]]:
    """输入：单个字符、随机源、可选的写法子集。输出：(主答案写法, 其余等价写法)。
    预期：主答案在给定子集里随机轮换，保证同一码位的多种写法在数据集里都出现；
          其余写法供判分接受。传 `forms` 是为了让 T14（数值写法）与 T15（字节写法）
          各自只在自己那一类里轮换，两题不会问成同一个东西。
    """
    candidates = tuple(forms) if forms is not None else codepoint_surfaces(char)
    index = rng.randrange(len(candidates))
    return candidates[index], candidates[:index] + candidates[index + 1 :]


def _codepoint_distractors(
    char: str, rng: random.Random, *, depth: str, answer_form: str, count: int = 3
) -> tuple[Distractor, ...]:
    """输入：字、随机源、难度档、主答案写法、条数。输出：错误码位干扰项。
    预期：偏移量按 near -> mid -> far 递增；不越界、不与本字码位重复；说明里给出本字的正确码位。
    """
    code = ord(char)
    order = ("near", "mid", "far")
    start = order.index(depth) if depth in order else 0
    picked: list[int] = []
    for tier in order[start:]:
        offsets = list(_CODE_OFFSETS[tier])
        rng.shuffle(offsets)
        for offset in offsets:
            value = code + offset
            if 0 < value <= _MAX_CODE and value != code and value not in picked:
                picked.append(value)
            if len(picked) >= count:
                break
        if len(picked) >= count:
            break
    return tuple(
        Distractor(text=f"U+{value:04X}", why_wrong=f"本字的码位是 {answer_form}，U+{value:04X} 是另一个码位")
        for value in picked[:count]
    )


# **题干校验**（判据在 L4 的 `stem_asks`）：校验空着 = 不设防（同 T1 的理由）。
# 代价（549 条已发布产出逐条量）：三条各 **0 条**被没通过。
_CODEPOINT_ASK_MARKERS = ("码位", "码点", "Unicode", "编码")
_BYTES_ASK_MARKERS = ("UTF-8", "UTF8", "编码", "字节")
_REVERSE_ASK_MARKERS = ("码位", "码点", "对应", "哪一个", "哪个字", "什么字", "哪个")


def build_unicode_codepoint(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T14：问这个字的 Unicode 码位（十六进制 / 十进制多种写法都算对）。
    输入：字记录 + 随机源 + 难度 + 索引（未用）。
    输出：TaskSpec；无码位字返回 None。
    预期：答案 = 本字码位的等价写法集合；题干必须写出这个字，且不得出现任何码位写法。
    """
    if not character.has_codepoint or not character.zi:
        return None
    forms = codepoint_surfaces(character.zi)
    values = _value_forms(character.zi)
    answer, aliases = _pick_form(character.zi, rng, forms=values)
    # 裸十六进制（forms[1]）仍算答对，只是不当主答案：当主答案时解析会写成「码位是 3406」，
    # 那既像十进制又像十六进制。判分集合保留它，表述不保留它。
    aliases = (*aliases, forms[1])
    return TaskSpec(
        task_id="T14_unicode_codepoint",
        payload_facts=("zi", "codepoint_forms"),
        instruction=(
            f"「{character.zi}」的 Unicode 码点"
        ),
        facts={
            "zi": character.zi,
            # 只放**码点**写法（带前缀 hex / 裸 hex / 十进制）。`codepoint_surfaces` 的后两条
            # 是 UTF-8 字节序列与 UTF-16 代理对，那是「编码」不是「码点」——混进这个键里，
            # 措辞模型会把字节串当码点写进题干（实测产出「给出码位 E3 90 8B，让读者从下列
            # 选项中选出对应的字」，主人点名过）。
            "codepoint_forms": list(values),
            "stroke_total": character.stroke_total,
            "radical": character.radical,
        },
        answer=answer,
        answer_aliases=aliases,
        visible_subject=character.zi,
        stem_asks_any=_CODEPOINT_ASK_MARKERS,
        distractors=_codepoint_distractors(
            character.zi, rng, depth=difficulty.distractor_depth, answer_form=answer
        ),
        proof=(f"「{character.zi}」的 Unicode 码位是 {forms[0]}，十进制写作 {forms[2]}",),
        provenance=make_provenance(character, "T14_unicode_codepoint", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def build_unicode_bytes(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T15：问这个字的 UTF-8 编码字节序列（补充平面字另给 UTF-16 代理对）。
    输入：字记录 + 随机源 + 难度 + 索引（用于取别的字的编码当干扰项）。
    输出：TaskSpec；无码位字返回 None。
    预期：答案 = 本字 UTF-8 字节（空格分隔大写为主，紧凑写法也算对）；
          干扰项取别的字的 UTF-8 编码，档位随 distractor_depth（near 同部首近笔画）。
    """
    if not character.has_codepoint or not character.zi:
        return None
    spaced = _utf8_of(character.zi)
    compact = spaced.replace(" ", "")
    byte_count = len(character.zi.encode("utf-8"))
    answer, aliases = _pick_form(character.zi, rng, forms=(spaced, compact))
    if compact.lower() != answer:
        aliases = (*aliases, compact.lower())
    facts: dict[str, object] = {
        "zi": character.zi,
        "codepoint": codepoint_surfaces(character.zi)[0],
        "utf8_bytes": spaced,
        "byte_count": byte_count,
        "stroke_total": character.stroke_total,
    }
    forms = codepoint_surfaces(character.zi)
    if len(forms) > 4:  # 非 BMP：补一条 UTF-16 代理对，让「编码」这一课讲全
        facts["utf16_units"] = forms[4]
    utf16_note = f"，并给出它的 UTF-16 写法 {forms[4]}" if len(forms) > 4 else ""
    distractors = char_distractors(
        index,
        character,
        [spaced, compact],
        rng,
        depth=difficulty.distractor_depth,
        why_wrong=lambda other: f"这是「{other.zi}」的 UTF-8 编码，不是本字",
        text_of=lambda other: _utf8_of(other.zi),
    )
    return TaskSpec(
        task_id="T15_unicode_bytes",
        payload_facts=("zi", "utf8_bytes", "byte_count"),
        instruction=(
            f"「{character.zi}」在 UTF-8 下的十六进制字节序列"
        ),
        facts=facts,
        answer=answer,
        answer_aliases=aliases,
        visible_subject=character.zi,
        stem_asks_any=_BYTES_ASK_MARKERS,
        distractors=distractors,
        proof=(f"「{character.zi}」的 UTF-8 编码是 {spaced}，共 {byte_count} 个字节",),
        provenance=make_provenance(character, "T15_unicode_bytes", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def build_unicode_reverse(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T16：给码位，问是哪个字（选择题）。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；无码位字返回 None，凑不满 3 个选项时也返回 None（两选一是抛硬币，不算题）。
    预期：答案 = 本字；选项 = 本字 + 至少两个别的字；题干给码位并**逐个列出候选**
          （读者要有得选，只写「下列选项」的产出是废题）；答案只是候选之一，
          因此本题不做 `no_answer_leak`，由 `options_ok` 的「每个候选只出现一次」兜住。
    """
    if not character.has_codepoint or not character.zi:
        return None
    # 「码位」只能是**自明**的数值写法（见 `_value_forms`）：`codepoint_surfaces` 的第 4 条是
    # UTF-8 字节序列、第 5 条是 UTF-16 代理对，那两条都不是码位；第 2 条是裸十六进制，与十进制
    # 写法在文本里无法区分。旧版从**全部**写法里轮换，实测产出会写成「给出码位 E3 90 8B，让读者
    # 从下列选项中选出对应的字」（主人点名过）——现在只轮换带前缀十六进制与十进制两种。
    values = _value_forms(character.zi)
    shown, _ = _pick_form(character.zi, rng, forms=values)
    distractors = char_distractors(
        index,
        character,
        [character.zi],
        rng,
        depth=difficulty.distractor_depth,
        why_wrong=lambda other: f"「{other.zi}」不是码位 {shown} 对应的字",
    )
    if len(distractors) < 2:
        return None
    options = (character.zi, *(d.text for d in distractors))
    return TaskSpec(
        task_id="T16_unicode_reverse",
        payload_facts=("codepoint", "codepoint_forms"),
        instruction=(
            # 候选写进说明（同 T10 的理由）：只写「候选里」小模型就不列候选，题干没有可选项。
            f"{'、'.join(options)} 里码点等于 {shown} 的那个字"
        ),
        facts={
            "codepoint": shown,
            # 同 T14：只留码点写法，UTF-8 字节序列不进这个键。
            "codepoint_forms": list(values),
            "zi": character.zi,
            "radical": character.radical,
            "stroke_total": character.stroke_total,
        },
        answer=character.zi,
        options=options,
        visible_subject=shown,
        stem_asks_any=_REVERSE_ASK_MARKERS,
        distractors=distractors,
        proof=(f"码位 {codepoint_surfaces(character.zi)[0]} 对应的字是「{character.zi}」",),
        provenance=make_provenance(character, "T16_unicode_reverse", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )
