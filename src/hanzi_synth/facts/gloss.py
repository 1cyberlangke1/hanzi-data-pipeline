"""释义义项的确定性抽取：把字典原文里的义项切成可以当答案的短句。

输入：`Character`（含 `meanings` 分栏原文）。
输出：短义项元组（2~30 个汉字，去重保序）+ 来源书名。
预期：全程只做确定性字符串处理，不调模型、不改写内容——义项是**事实短语**，
      可以逐条回查字典原文；原始整段释义不进产出（那是词典的表述，不是我们的措辞）。
"""

from __future__ import annotations

import re

from .records import Character, clean_meaning_block

# 有明确书名的释义栏在前（与 `tasks/relations.py::_MEANING_SOURCES` 同一套归属）：
# mean_h 中华字海 / mean_k 康熙字典 / mean_j 集韵；其余（综合/补遗/异体）只当无来源材料。
_SOURCED_LABELS: tuple[tuple[str, str], ...] = (
    ("mean_h", "中华字海"),
    ("mean_k", "康熙字典"),
    ("mean_j", "集韵"),
)
_UNSOURCED_LABELS: tuple[str, ...] = ("mean_z", "mean_b", "mean_v")

# 义项内部的句读：一个义项里再分句就该拆成两条，否则答案会长得没法当答案。
_SENTENCE_SPLIT = re.compile(r"[。；;！!？?\n]")
# 义项长度上下限（按汉字数）：字典里「归」「转身」都是正经义项，单字不许丢；
# 太长（>30 字）不是义项而是整段，那种只能当素材不能当答案。
_MIN_ITEMS = 1
_MIN_LEN = 1
_MAX_LEN = 30
# 只剩虚词、不构成义项的片段。
_STOPWORDS = frozenset({"也", "的", "之", "同", "即", "或", "又", "等", "皆"})
# 词典的元信息标注：它是**性质说明**不是词义（「音译字」= 这个字用来音译）。
# 不丢，但排在后面——「这个字什么意思」拿「音译字」当答案太虚。
_META_ITEMS = frozenset({"音译字", "部件扩展", "古字", "俗字", "异体字"})
_CJK = re.compile(r"[\u3400-\u9fff]")
# 能当答案的义项必须是**纯中文短语**：汉字 + 顿号/逗号/间隔号 + 可选句号。
# 为什么这么严：切分只按句读，字典原文里的排版标记与注记会**落单**在片段里，当答案就是错的——
# 实测全池 33,924 个可出释义题的字里 **9,821 个（29%）** 的首条义项是这类残片：
#   - 缺一半的括号：`<日乘`（原文 `【日乘】` 掉了 `【`）、`捣】的二简字`、`地名也（俗字攷`；
#   - 词头占位符：`堪～]传说中的一种鱼`、`三～]姓氏`（`～` 就是本字本身，讲的是**词**的用法）；
#   - 注音与罗马字：`读音たたみ`、`韩国读音nol`、`音itachi`；
#   - 页码 / 岁数注记：`77歳`（数字也在这条规则里一并挡掉）。
# 代价是覆盖：可出释义题的字 33,924 → 23,621。宁可这个字不出释义题
# （与「信息太少的字不做合成」同一标准），也不出答案错的题。
_PLAIN_GLOSS_RE = re.compile(r"^[\u3400-\u9fff]+(?:[，,、·][\u3400-\u9fff]+)*。?$")


def _cjk_count(text: str) -> int:
    """输入：任意文本。输出：其中的汉字个数。预期：用来判长度与「是不是纯注音」。"""
    return len(_CJK.findall(text))


def is_plain_gloss(text: str) -> bool:
    """输入：一个候选义项。输出：它是不是「纯中文短语」形态的义项。

    预期：判据就是 `_PLAIN_GLOSS_RE`（汉字 + 顿号/逗号/间隔号，可选句号）——字典排版残片
          （`～` 词头 / 括注 / 注音 / 页码）一律不算义项。抽取与事后清洗共用同一条判据，
          两边不会漂移。
    """
    return bool(_PLAIN_GLOSS_RE.match(text))


def _items_of_block(block: str) -> list[str]:
    """输入：一个释义块的原文。输出：块内可当答案的短义项（去重保序，块内顺序不变）。
    预期：先按 `clean_meaning_block` 剥掉读音标注/词头引用/互见条目，再按句读切开，
          只留**纯中文短语**（1~30 个汉字 + 顿号/逗号/间隔号）——见 `_PLAIN_GLOSS_RE`。
    """
    out: list[str] = []
    for piece in _SENTENCE_SPLIT.split(clean_meaning_block(block)):
        piece = piece.strip(" 　,，、:：·—…\"'“”‘’()（）[]〔〕【】")
        if not _MIN_LEN <= _cjk_count(piece) <= _MAX_LEN:
            continue
        if piece in _STOPWORDS:
            continue
        if not is_plain_gloss(piece):
            continue
        if piece not in out:
            out.append(piece)
    # 有内容的义项排在元信息前面：答案取第一条，不能是「音译字」这种性质说明。
    return sorted(out, key=lambda item: item in _META_ITEMS)


def plain_senses(block: str) -> tuple[str, ...]:
    """输入：一个释义块的**原文**。输出：块内可当答案 / 识别条件的纯中文义项（去重保序）。

    预期：就是 `_items_of_block` 的公开入口——调用方（构造器）据此拼识别条件，与
          `gloss_items` 用的是同一套清洗，两处不会漂移。
    """
    return tuple(_items_of_block(block))


def gloss_phrase(block: str, *, max_items: int = 2, max_chars: int = 9) -> str:
    """输入：一个释义块的**原文** + 最多取几条义项 + **单条义项**的长度上限（汉字数）。
    输出：压成一句的纯中文短语；取不到、或**首条义项本身就超长**时返回空串。

    预期：与 `gloss_items` 共用同一套清洗（`_items_of_block`）——只留纯中文义项，`～` / 括注 /
          注音 / 互见这类排版残片一个都进不来；义项之间用顿号连成**名词性短语**（出现句读就
          变成完整句子，小模型会整句照抄进题干）。
          `max_chars` 卡的是**单条义项**：校验 `no_source_copy` 数的是**连续汉字**，顿号会断开
          计数，所以只要每条义项 ≤9 字，题干照实写出释义也不会被判「照抄」；反过来，释义整句
          超过 9 字时小模型改写不了、只能照抄，同一道题反复重跑——实测短语 ≥10 字的 T13 被拒率
          94%、<10 字的 32%（已发布 703 条 vs rejects 158 条）。这种字**不出反向题**，与
          「信息太少的字不做合成」同一标准。首条义项超长时不退而取后面的义项：识别条件该是
          **主义项**，挑一条短的会问得莫名其妙。
    """
    senses = plain_senses(block)
    if not senses or _cjk_count(senses[0]) > max_chars:
        return ""
    return "、".join(item for item in senses[:max_items] if _cjk_count(item) <= max_chars)


def gloss_items(character: Character) -> tuple[tuple[str, ...], str]:
    """输入：字记录。输出：(义项元组, 来源书名)。来源为空串表示义项来自没有明确书名的分栏。
    预期：有书名的分栏（中华字海/康熙字典/集韵）优先；至少 `_MIN_ITEMS` 条才返回，
          否则返回空元组——调用方据此判定该字出不了释义题，不硬造。
    """
    for label, book in _SOURCED_LABELS:
        items: list[str] = []
        for block in character.meanings.get(label) or ():
            for item in _items_of_block(block):
                if item not in items:
                    items.append(item)
        if len(items) >= _MIN_ITEMS:
            return tuple(items), book
    for label in _UNSOURCED_LABELS:
        items = []
        for block in character.meanings.get(label) or ():
            for item in _items_of_block(block):
                if item not in items:
                    items.append(item)
        if len(items) >= _MIN_ITEMS:
            return tuple(items), ""
    return (), ""
