"""L4 验证栈：机械闸门只判「产出有没有背叛定死的载荷」，写法由 LLM 那一层判。

设计分界：
- 机械闸门（本模块）判的是**事实保真**与**数据形态**：答案串在不在解析里、题干有没有把答案
  直接送出去、候选列没列全、数字是不是凭空造的、有没有被截断、题干是不是中文自然语言、
  有没有照抄释义原文。它**不判写法**：不判句式、不判语气、不判长短、不判有没有套话——
  那些是开放集合，用枚举去追永远追不完。
- 「读起来通不通、读者能不能读懂、像不像一道题」交给 L4.6 往返检验（把题干交回模型答题）
  与 L4.5 裁判（可选开关）。这两条是 LLM，不是规则表。
- 「像不像已有的」交给 L6 去重（n-gram 相似度，是个度量，永远收敛）。

机械闸门的判据只有两类，都不是「模型该怎么写」的规定：
1. **载荷里的定死串**：答案、候选、必现串、隐藏主语、结构词——产出必须原样带上该带的、
   绝不能带上不该带的（子串 / 集合比对）；
2. **产出自身的形态**：中文自然语言、数字来源可查、没有被截断、与释义原文没有长串照抄。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..facts.surface import IDC_LABELS
from ..tasks.base import TaskSpec

if TYPE_CHECKING:  # 只在类型标注期需要，运行期不导入——`verbalizer` 反过来要 import 本模块
    from ..llm.verbalizer import Verbalized

# 比对时忽略的字符：同一串的多种书写（`E3 90 80` / `E3、90、80` / `E39080`）视为同一串。
# 实测模型写字节序列时爱用「、」分隔，不做归一就会把正确产出误判成「解析里没有答案」。
_MATCH_IGNORE = " \t\u3000、，,·．./／|｜"

# 阿拉伯数字串：用来查「产出里的数字是不是凭空造的」。
_NUM_RE = re.compile(r"\d+")

# 汉字码位区间（含扩展 A/B 及以上）：用来判「题干是不是中文自然语言」。
# 程序侧文本一律英文，所以英文题干 / 原样抄回的载荷必然一个汉字都没有，一抓一个准。
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x20000, 0x2FFFF), (0x30000, 0x3FFFF),
)

# 与释义原文连续相同的汉字数上限。只抓**照抄**（近逐字复述），不抓合理改写：
# 释义原文本身常常只有十几个汉字，阈值再低就会把正常改写误杀。
_SOURCE_COPY_RUN = 10

# 字形结构词表：`IDC_LABELS` 的值就是那 15 种结构的人话名。这是**程序侧数据**，不是手写规则表
# ——新增 IDS 描述符时词表自动跟着走。只有声明了 `structure_words` 的题（T13 反向题）才启用：
# 题干里一个结构词都不许出现（把答案那棵树用散文说一遍等于泄题），解析里的结构词必须来自答案。
_STRUCTURE_WORDS: tuple[str, ...] = tuple(sorted(set(IDC_LABELS.values())))

# 裁判否决 / 往返检验否决回灌给措辞模型时的反馈名。两条闸门分开写：
# 失败原因要能在 rejects 里一眼分开，否则「裁判说不好」和「读者答不出答案」会被混成一类。
JUDGE_FEEDBACK_KEY = "裁判否决"
ROUNDTRIP_FEEDBACK_KEY = "往返检验否决"

# 裁判的四条判据：裁判 system prompt 与措辞提示词都从这里生成，避免两边各写一份而漂移。
JUDGE_CRITERIA: tuple[tuple[str, str], ...] = (
    ("self_sufficient", "自足：只读题干就明白要回答什么、问的是哪个字或哪一项，题干自己给出识别条件；"
                        "反向题（靠释义与读音问字形）**不要求题干写出答案本身**，识别条件够就行"),
    ("unambiguous", "无歧义：按题干给出的条件，读者能定出唯一一个答案；"
                    "反向题里被隐藏的那个字**不算歧义**——识别条件唯一即可"),
    ("explanation_clear", "解析说清：解析真的把答案说出来，且与答案字段一致"),
    ("asks_a_question", "像一道题：题干真的在提问或给出作答要求；只陈述「这个字是 X」不算"),
)


def normalize_for_match(text: str) -> str:
    """输入：任意文本。输出：去掉空白与列举分隔符后的文本。
    预期：只做比对归一，不改内容、不做大小写折叠（hex 大小写另有别名覆盖）。
    """
    return "".join(ch for ch in (text or "") if ch not in _MATCH_IGNORE)


def _is_cjk(char: str) -> bool:
    """输入：单个字符。输出：它是不是汉字（含扩展区）。预期：只判码位，不判语义。"""
    point = ord(char)
    return any(low <= point <= high for low, high in _CJK_RANGES)


def _cjk_count(text: str) -> int:
    """输入：任意文本。输出：其中汉字的个数。预期：供「题干是中文自然语言」这条闸门用。"""
    return sum(1 for char in (text or "") if _is_cjk(char))


def _longest_common_run(left: str, right: str) -> int:
    """输入：两段文本。输出：它们的最长公共连续子串长度（按字符）。

    预期：只用于「题干有没有照抄释义原文」这一条。取两段文本里的**汉字**再比（标点、空白、
          引号都先去掉），因为改写时标点本来就会变。动态规划两行滚动，文本都在几百字以内。
    """
    a = [char for char in (left or "") if _is_cjk(char)]
    b = [char for char in (right or "") if _is_cjk(char)]
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    best = 0
    for char_a in a:
        current = [0] * (len(b) + 1)
        for index, char_b in enumerate(b, start=1):
            if char_a == char_b:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def _missing_in_stem(stem: str, needles: "Sequence[str]") -> list[str]:
    """输入：题干 + 必须原样出现的串。输出：其中**没出现**的串（归一化后比对）。
    预期：`visible_subject` / `must_contain` / 候选共用这一份判据——它们是「读者答得了题」的
          最低条件，只写在提示词里模型会漏（实测首轮 6 条被拒里有 5 条是必现串）。
    """
    stem_norm = normalize_for_match(stem)
    return [
        str(item)
        for item in needles
        if normalize_for_match(str(item)) and normalize_for_match(str(item)) not in stem_norm
    ]


def _numbers_in(text: str) -> set[str]:
    """输入：任意文本。输出：其中出现的阿拉伯数字串集合。预期：只切数字，不做语义判断。"""
    return set(_NUM_RE.findall(text or ""))


def allowed_numbers(spec: TaskSpec) -> set[str]:
    """输入：TaskSpec。输出：载荷里出现过的全部阿拉伯数字串。
    预期：事实、答案（含别名与写法）与候选里的数字都算合法来源——产出里的数字只要能在
          这些地方找到就算忠实，找不到就是凭空造的。
    """
    blob = json.dumps(spec.facts, ensure_ascii=False, default=str)
    allowed = _numbers_in(blob)
    for item in (*spec.accepted, *spec.options):
        allowed |= _numbers_in(str(item))
    return allowed


@dataclass(frozen=True)
class Verification:
    """输入：各道机械闸门的结果。输出：不可变验证记录，直接写进产出。
    预期：`tier == 'A'` 才允许进入 L4.5 / L4.6；发布与否由 `pipeline.run._publishable` 统一判。
    """

    schema: bool = False
    no_empty_slot: bool = False
    answer_in_explanation: bool = False
    no_answer_leak: bool = False
    subject_in_stem: bool = False
    options_listed: bool = False
    stem_asks: bool = False
    no_hidden_subject: bool = False
    structure_words: bool = False
    numbers_from_facts: bool = False
    not_truncated: bool = False
    stem_is_chinese: bool = False
    no_source_copy: bool = False
    failures: tuple[str, ...] = ()

    @property
    def tier(self) -> str:
        """输入：无。输出：'A'（七条全过）或 'C'（有任一未过）。
        预期：B 级留给「缺裁判」的场景，本层不产生。
        """
        return "A" if all(
            (
                self.schema,
                self.no_empty_slot,
                self.answer_in_explanation,
                self.no_answer_leak,
                self.subject_in_stem,
                self.options_listed,
                self.stem_asks,
                self.no_hidden_subject,
                self.structure_words,
                self.numbers_from_facts,
                self.not_truncated,
                self.stem_is_chinese,
                self.no_source_copy,
            )
        ) else "C"

    def as_dict(self) -> dict:
        """输入：无。输出：可 JSON 序列化的字典。预期：留 tier 与 failures 供审计与回炉读。"""
        return {
            "schema": self.schema,
            "no_empty_slot": self.no_empty_slot,
            "answer_in_explanation": self.answer_in_explanation,
            "no_answer_leak": self.no_answer_leak,
            "subject_in_stem": self.subject_in_stem,
            "options_listed": self.options_listed,
            "stem_asks": self.stem_asks,
            "no_hidden_subject": self.no_hidden_subject,
            "structure_words": self.structure_words,
            "numbers_from_facts": self.numbers_from_facts,
            "not_truncated": self.not_truncated,
            "stem_is_chinese": self.stem_is_chinese,
            "no_source_copy": self.no_source_copy,
            "tier": self.tier,
            "failures": list(self.failures),
        }


def verify(spec: TaskSpec, verbalized: "Verbalized") -> Verification:
    """输入：TaskSpec 与 L3 产出。输出：Verification。
    预期：任一条不过就记进 failures，绝不因为「只差一点点」而放行。
    """
    failures: list[str] = []

    schema_ok = bool(verbalized.ok and verbalized.stem and isinstance(verbalized.explanation, str))
    if not schema_ok:
        failures.append("schema: 槽位缺失或为空")
        return Verification(schema=False, failures=tuple(failures))

    stem, explanation = verbalized.stem, verbalized.explanation

    # 1. 空槽位兜底：`_slot_check` 已挡一层，这里防别处构造的 Verbalized 漏进来。
    no_empty_slot = bool(stem.strip() and explanation.strip())
    if not no_empty_slot:
        failures.append("no_empty_slot: 题干或解析是空白")

    # 2. 解析必须原样说出答案：答案由代码定死，解析说不出它，这道题就没有解。
    explanation_norm = normalize_for_match(explanation)
    answer_in_explanation = any(
        normalize_for_match(str(item)) and normalize_for_match(str(item)) in explanation_norm
        for item in spec.accepted
    )
    if not answer_in_explanation:
        failures.append(f"answer_in_explanation: 解析里没有出现答案 {spec.answer!r}")

    # 3. 题干不得泄露答案。选择题除外：候选要逐个列出，答案本来就是其中之一。
    no_answer_leak = True
    if not spec.options:
        stem_norm = normalize_for_match(stem)
        for item in spec.accepted:
            needle = normalize_for_match(str(item))
            if needle and needle in stem_norm:
                no_answer_leak = False
                failures.append(f"no_answer_leak: 题干里直接出现了答案 {item!r}")
                break

    # 4a. 必现串：`visible_subject`（问哪个字）+ `must_contain`（题型声明的其它识别条件，
    #     如反向题的读音、部件链的位置描述）必须原样出现在题干里——少了任何一个，读者都答不了。
    missing_subject = _missing_in_stem(stem, (spec.visible_subject, *spec.must_contain))
    subject_in_stem = not missing_subject
    if missing_subject:
        failures.append(f"subject_in_stem: 题干里没有写出必现串 {missing_subject}")

    # 4b. 选择题：候选必须逐个出现在题干里。**这条不能省**——往返检验的载荷里带着 options，
    #     读者永远看得见候选，所以「题干没列候选」往返检验抓不到，只有这里能抓。
    options_listed = not _missing_in_stem(stem, spec.options)
    if not options_listed:
        failures.append(f"options_listed: 题干里没有列出候选 {_missing_in_stem(stem, spec.options)}")

    # 4c. 题干问的必须是答案那件事：命中 `stem_asks_any` 里的任一种说法即可。
    #     实测 T1 的题干问读音与部首、答案却是笔画数——问错了方向，读者照着答必然错。
    stem_asks = (not spec.stem_asks_any) or any(item in stem for item in spec.stem_asks_any)
    if not stem_asks:
        failures.append(f"stem_asks: 题干没有问「{spec.stem_asks_any[0]}」那件事")

    # 4d. 隐藏主语（反向题的那个字）：题干写出它，题目就退化成查表题——读者不用看释义。
    no_hidden_subject = not spec.hidden_subject or normalize_for_match(spec.hidden_subject) not in normalize_for_match(stem)
    if not no_hidden_subject:
        failures.append(f"no_hidden_subject: 题干里写出了隐藏主语 {spec.hidden_subject!r}")

    # 4e. 字形结构：声明了 `structure_words` 的题，题干里一个结构词都不许出现（把答案那棵树
    #     用散文说一遍就是泄题），解析里出现过的结构词必须来自答案真实用到的那几个。
    structure_words = True
    if spec.structure_words:
        leaked = [word for word in _STRUCTURE_WORDS if word in stem]
        if leaked:
            structure_words = False
            failures.append(f"structure_words: 题干里出现了字形结构词 {leaked}")
        invented = [word for word in _STRUCTURE_WORDS if word in explanation and word not in spec.structure_words]
        if invented:
            structure_words = False
            failures.append(f"structure_words: 解析里的结构词 {invented} 不是答案用到的 {list(spec.structure_words)}")

    # 5. 数字必须来自载荷：产出里出现载荷没有的阿拉伯数字 = 凭空造数（笔画数、码位最容易编）。
    allowed = allowed_numbers(spec)
    invented = sorted((_numbers_in(stem) | _numbers_in(explanation)) - allowed)
    numbers_from_facts = not invented
    if invented:
        failures.append(f"numbers_from_facts: 正文里出现了载荷中没有的数字 {invented}")

    # 6. 截断：`finish_reason == 'length'` 说明这次产出被按长度砍断，不是完整的一句话。
    not_truncated = verbalized.finish_reason != "length"
    if not not_truncated:
        failures.append("not_truncated: 产出被按长度截断")

    # 7. 题干必须是**中文自然语言**：程序侧文本一律英文，题干里一个汉字都没有，只可能是
    #    模型把载荷原样抄了回来（或整段用英文作答）——两种都不是能发给读者的数据。
    stem_is_chinese = _cjk_count(stem) > 0
    if not stem_is_chinese:
        failures.append("stem_is_chinese: 题干里没有汉字（不是中文自然语言）")

    # 8. 题干不得照抄释义原文：释义是**素材**，不是可以直接搬进数据的文本（照抄既侵权，
    #    也等于没有改写）。只抓连续 `_SOURCE_COPY_RUN` 个汉字及以上的照抄，正常改写不会被误杀。
    source = ""
    if isinstance(spec.facts, Mapping):
        source = str(spec.facts.get("meaning_source") or "")
    no_source_copy = _longest_common_run(stem, source) < _SOURCE_COPY_RUN
    if not no_source_copy:
        failures.append(
            f"no_source_copy: 题干与释义原文连续相同超过 {_SOURCE_COPY_RUN - 1} 个汉字（照抄释义）"
        )

    return Verification(
        schema=True,
        no_empty_slot=no_empty_slot,
        answer_in_explanation=answer_in_explanation,
        no_answer_leak=no_answer_leak,
        subject_in_stem=subject_in_stem,
        options_listed=options_listed,
        stem_asks=stem_asks,
        no_hidden_subject=no_hidden_subject,
        structure_words=structure_words,
        numbers_from_facts=numbers_from_facts,
        not_truncated=not_truncated,
        stem_is_chinese=stem_is_chinese,
        no_source_copy=no_source_copy,
        failures=tuple(failures),
    )
