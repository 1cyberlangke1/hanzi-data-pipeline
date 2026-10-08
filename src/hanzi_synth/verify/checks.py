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
# 判的是「题干里一个汉字都没有」这件事实——整段用英文作答的产出必然一个汉字都没有，一抓一个准。
# 程序侧文本已改成中文，「照抄出题说明」不再由这条拦：出题说明是名词性短语、不是问句，
# 交给 L4.6 往返检验的 `is_question` 判（见 `llm/roundtrip.py`）。
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

# 产出必须**上下文无关**：读者手里只有这道题本身。两类词出现即说明模型在跟程序对话而不是跟读者
# 说话，或者把出题过程写进了题目：
#   1. 程序侧的说法——「题干」「载荷」「槽位」「校验器」，英文缩写 `IDS` / `payload` / `facts` /
#      `instruction`，以及闸门反馈里的行话（「必现串」）；
#   2. 指代出题过程、指代一份**题干里并不存在**的东西——「本题」「题目要求」「提供的汉字」
#      「提供的资料」。
# 实测 2B 会把 instruction 里的字段名（`stem_must_contain` / `facts.conditions`）意译成
# 「题干描述的…」「请在提供的汉字中找出…」，也会把英文夹注抄进中文题干
# （「请说明字符㐀的 IDS（ideographic description sequence）分解方式」）。
# `提供的` 单列成一条前缀：实测 2B 的固定开场是「根据提供的资料 / 信息 / 材料 / 汉字…」——
# 读者手上并没有任何「提供」的东西，这个说法本身就说明题干不自足，具体跟哪个名词无关。
# 判据是**子串比对**，判的是「产出里有没有出现这些说法」这件事实，与 `stem_is_chinese` 同类；
# 它不是措辞规则表——不规定该怎么写，只挡下这些**不该出现在读者视野里**的词。
_INTERNAL_TERMS: tuple[str, ...] = (
    "题干", "题目", "载荷", "槽位", "校验器", "本题", "题目要求", "提供的", "必现串",
    "解析", "payload", "facts", "instruction", "IDS",
    # 指代「一份并不存在的资料 / 出题过程」的说法。读者手上只有这道题本身，写出「根据资料」
    # 就是把出题人视角的话漏给了读者；实测全量 710,655 条里有 27 条这样（26 条出自加规范之前的
    # 旧提示词、1 条是当前提示词仍在漏），**全都在解析里**——机械闸门原先只认「提供的」这一类，
    # 于是「根据资料 / 根据给定 / 资料显示」整类漏网。这几条判据与 `stem_is_chinese` 同类：
    # 只判「产出里有没有出现这种说法」这件事实，不规定该怎么写。
    "根据资料", "根据给定", "根据上述", "据资料", "资料显示", "此题", "本题目",
    # 书名号形态的引用（「出自《中华字海》」「收录于《康熙字典》」）。判据仍是子串比对；题干里
    # 本来就点了书名的题型（T6 问页码）不受影响——它的题干里带着同一个书名，命中的是**引用框架**
    # 而不是书名本身，而 T6 的骨架不写框架（实测离线 707,551 条零命中）。
    "出自《", "收录于《", "见于《",
    # 引用框架的第二类漏网（实测 `llm_semantic.jsonl` 3,467 条里 900+ 条）：2B 把「出处」写进产出，
    # 形态远不止 `出自《` 一种——「根据《中华字海》的释义」「据中华字海的记载」「康熙字典中记载」
    # 「在《集韵》里」「说文解字说」。判据仍是**引用框架 / 书名**，不是「该怎么写」：
    # 下面每一条都在离线 707,551 条上量过命中 0（`字海` / `中华` / `在《` 命中 32,256 / 32,256 / 15,707
    # 条，**全是 T6 问页码**——它的题干本来就点着书名，所以那三个词不进表）。
    "据《", "按《", "按照《", "根据《", "中《", "见《",
    "据中华", "根据中华",
    "康熙", "字典", "集韵", "直音篇", "说文",
)

#
# 占位符记号：程序给生僻字发的代号（`[[1]]`），回填发生在措辞之后——回填表里没有的记号（模型
# 自己造出来的）没有任何东西能把它换掉，会原样进数据集。实测把示范写进对话轮次时，2B 会照着
# 示范的形状在句尾多补一句 `[[1]]、[[2]]`；改掉示范形状后仍要有一道闸门兜底。
_PLACEHOLDER_RE = re.compile(r"\[\[\d+\]\]")

# 两条闸门的名字，**只进 rejects / attempt-stats**（给人看）。两条分开写：
# 失败原因要能一眼分开，否则「裁判说不好」和「读者答不出答案」会被混成一类。
# 回灌给措辞模型的那份**不带这两个名字**（见 `pipeline/run.py::_roundtrip_feedback`）——
# 闸门名对模型毫无用处，只是塞进上下文的行话。
JUDGE_FEEDBACK_KEY = "裁判否决"
ROUNDTRIP_FEEDBACK_KEY = "往返检验否决"

# 裁判的五条判据：裁判 system prompt 与措辞提示词都从这里生成，避免两边各写一份而漂移。
# 合成的是中文数据，判据也一律中文——英文判据要多一道翻译，2B 会译错（真机把 `reading`
# 译成「阅读」），而且译错的词会直接进产出。
JUDGE_CRITERIA: tuple[tuple[str, str], ...] = (
    ("self_sufficient", "自足：只读题干，读者就知道要答什么、问的是哪个字或哪样东西——"
                        "全部识别条件都写在题干里。反向题（给出读音与释义、问字形）不必把答案"
                        "本身写出来，识别条件给够就行。"),
    ("unambiguous", "无歧义：题干给出的条件唯一确定一个答案。反向题里被隐藏的那个字不算歧义。"),
    ("explanation_clear", "解析真的说出了答案，并且与答案一致。"),
    ("asks_a_question", "题干真的在向读者提问或给出作答要求；只陈述一条信息不算提问。"),
    ("natural_wording", "措辞是中国人真会说的话：不要字典腔、不要生造词、不要罗列腔，"
                        "也不要出现谈论写作要求、字段、给定清单或题目本身的句子。"),
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
    """输入：任意文本。输出：其中汉字的个数。预期：供「题干是中文自然语言」与「解析不是光秃答案」两条闸门用。"""
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


def answer_leak_limit(spec: TaskSpec, item: str) -> int:
    """输入：TaskSpec + 一个答案串（主答案或别名）。输出：它在题干里允许出现的次数（0 或 1）。

    预期：**渲染层与闸门共用这一份判据**——`visible_subject` / `must_contain` / `options` 里的串
          是读者本来就要在题干里看到的（要从候选里选、要看见问的是哪个字），答案在其中时允许出现
          1 次，其余 0 次。两处算法一旦分叉，渲染就会交出注定被闸门拒掉的句子：实测全量有 38 条
          (题型, 字) 因此一条数据都出不来，其中 32 条只要换一条不撞答案的句式就能出
          （`synthesis/out/probe_leak_rescue.py`）。判据比的是**归一化后**的串（`normalize_for_match`
          会去掉空白与列举分隔符，T15 的「E3 90 8B」这类带空格的写法靠它才比得准）。
    """
    needle = normalize_for_match(str(item))
    if not needle:
        return 0
    declared = {
        normalize_for_match(str(candidate))
        for candidate in (spec.visible_subject, *spec.must_contain, *spec.options)
        if str(candidate)
    }
    return 1 if needle in declared else 0


def leaked_field_names(spec: TaskSpec, text: str) -> list[str]:
    """输入：TaskSpec + 一段正文。输出：正文里逐字出现的**载荷键名**（排序去重）。

    预期：键名 = `payload_for(spec)` 的字段名 ∪ `spec.facts` 的键。只比**键**，不比**值**——
          值本来就该出现在正文里（答案、候选、读音、笔画数都是值）。
          键名表是**从载荷本身推出来的**，不写死：新增题型字段自动被覆盖，不会变成一张追不完的
          规则表。`payload_for` 反向依赖本模块，所以这里延迟导入。
    """
    from ..llm.verbalizer import payload_for  # 延迟导入：verbalizer 反过来要 import 本模块

    names = [str(name) for name in payload_for(spec) if isinstance(name, str)]
    if isinstance(spec.facts, Mapping):
        names.extend(str(name) for name in spec.facts if isinstance(name, str))
    low = (text or "").lower()
    return sorted({name for name in names if len(name) >= 3 and name.lower() in low})


# 判据词被**连成一串**念出来多长才算「把判据词表罗列出来」。正常问句会命中一两个（那是闸门
# 本来就要求的），但不会连着念；把整张表搬进题干（「读作、怎么读、念作、读法、拼音」）才会连成串。
# 只算中间**只隔分隔符 / 并列连词**的那些——按「命中总数」判会把正常措辞误杀（实测 T16 写
# 「编码码点…请根据码位与码点的对应关系」，命中 5 个词但一个都不相邻，是正常问句）。
_MARKER_ENUMERATION_RUN = 3

# 「只隔分隔符」的判据：这些字符（含并列连词「和与及或」）之间的判据词算连成一串。
_SEPARATOR_ONLY_RE = re.compile(r"^[\s、，,。；;：:·．./／|｜（）()【】「」『』\"'“”‘’和与及或]*$")

# 正文里**允许**出现的 ASCII 字母串：中文技术写作里的标准术语——码位 / 字节序列 / 编码这几类题
# 离不开它们，写「Unicode 码位」比写「统一码码位」自然。**扩展点**：实测出现新的标准术语就加进来，
# 比「任由英文词混进中文散文」可控。
_ALLOWED_FOREIGN_WORDS: tuple[str, ...] = ("unicode", "utf", "utf8", "ascii", "hex")

# ASCII 字母串（>=2 个字母）：单个字母不算——`U+3400`、`E3 90 80` 里的字母都是单个的。
_FOREIGN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def foreign_words(spec: TaskSpec, text: str) -> list[str]:
    """输入：TaskSpec + 一段正文。输出：正文里出现的、**载荷里没有**的 ASCII 字母串。

    预期：中文散文里夹英文词就是没写中文（实测 T3 写「结构 notation 字符串」）。判据是子串 /
          集合比对：字母串要么出现在载荷文本里（`UTF-8`、`xi3`、`E3 90 80` 这类本来就是数据），
          要么是 `_ALLOWED_FOREIGN_WORDS` 里的标准术语。大小写不敏感。
    """
    haystack = json.dumps(spec.facts, ensure_ascii=False, default=str)
    haystack += " " + " ".join(str(item) for item in (*spec.accepted, *spec.options))
    haystack = haystack.lower()
    return sorted({
        word for word in _FOREIGN_WORD_RE.findall(text or "")
        if word.lower() not in _ALLOWED_FOREIGN_WORDS and word.lower() not in haystack
    })


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


def _longest_marker_run(stem: str, markers: "Sequence[str]") -> tuple[int, list[str]]:
    """输入：题干 + 判据词表（`stem_asks_any`）。输出：(最长连续串的长度, 那串里的词)。

    预期：只把**中间只隔分隔符 / 并列连词**的判据词算作「连成一串」——「读作、怎么读、念作」
          算一串；「编码码点…根据码位与码点的对应关系」不算（中间隔着别的字，是正常措辞）。
          重叠命中（「哪个」与「哪一个」同起点）只算一次；一个词都没命中时返回 (0, [])。
    """
    found: list[tuple[int, int, str]] = []
    for marker in markers:
        if not marker:
            continue
        start = stem.find(marker)
        while start != -1:
            found.append((start, start + len(marker), marker))
            start = stem.find(marker, start + 1)
    if not found:
        return 0, []
    found.sort()
    best_length, best_words = 1, [found[0][2]]
    run_length, run_words = 1, [found[0][2]]
    end = found[0][1]
    for start, stop, marker in found[1:]:
        if start < end:  # 与上一个词重叠：同一个位置只算一次
            end = max(end, stop)
            continue
        if _SEPARATOR_ONLY_RE.match(stem[end:start]):
            run_length += 1
            run_words.append(marker)
        else:
            run_length, run_words = 1, [marker]
        end = stop
        if run_length > best_length:
            best_length, best_words = run_length, list(run_words)
    return best_length, best_words


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
    no_instruction_copy: bool = False
    no_internal_terms: bool = False
    no_field_names: bool = False
    no_foreign_words: bool = False
    no_marker_enumeration: bool = False
    no_escape_artifacts: bool = False
    no_placeholder_leak: bool = False
    explanation_is_prose: bool = False
    failures: tuple[str, ...] = ()
    # 同一批失败的**模型侧**说法，只回灌给措辞模型（`pipeline/run.py` 的 reflexion）。
    # 中文，且刻意用 `_INTERNAL_TERMS` 里的词（题干 / 载荷 / 必现串）——模型把反馈原文抄进产出，
    # `no_internal_terms` 当场拦下。与 `failures` 同序同长。
    notes: tuple[str, ...] = ()
    # 答案核验的**语义那一半**的痕迹：`answer_in_explanation` 是子串比对，判得了「解析里有没有那串
    # 字」、判不了「解析说的是不是那个答案」，所以它只做快速通过，过不了才问 LLM（见
    # `llm.roundtrip.explanation_states_answer`）。空串 = 没问（快速通过）或没得问（offline /
    # 没接客户端）；仲裁缺席与仲裁判定都写在这一个字段里。
    semantic_answer: str = ""

    @property
    def tier(self) -> str:
        """输入：无。输出：'A'（下面列的闸门全过）或 'C'（有任一未过）。
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
                self.no_instruction_copy,
                self.no_internal_terms,
                self.no_field_names,
                self.no_foreign_words,
                self.no_marker_enumeration,
                self.no_escape_artifacts,
                self.no_placeholder_leak,
                self.explanation_is_prose,
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
            "no_instruction_copy": self.no_instruction_copy,
            "no_internal_terms": self.no_internal_terms,
            "no_field_names": self.no_field_names,
            "no_foreign_words": self.no_foreign_words,
            "no_marker_enumeration": self.no_marker_enumeration,
            "no_escape_artifacts": self.no_escape_artifacts,
            "no_placeholder_leak": self.no_placeholder_leak,
            "explanation_is_prose": self.explanation_is_prose,
            "tier": self.tier,
            "failures": list(self.failures),
            "notes": list(self.notes),
            "semantic_answer": self.semantic_answer,
        }


def verify(spec: TaskSpec, verbalized: "Verbalized") -> Verification:
    """输入：TaskSpec 与 L3 产出。输出：Verification。
    预期：任一条不过就记进 failures，绝不因为「只差一点点」而放行。
          `failures` 给人看（rejects / attempt-stats），`notes` 回灌给措辞模型；两份在同一个
          地方同源生成，同序同长，不会漂移。两份**都是中文**：合成的是中文数据，中英混排会让
          2B 多一道翻译并译错。`notes` 刻意用闸门会拦的词（题干 / 载荷 / 必现串）——模型把反馈
          抄进产出即被 `no_internal_terms` 拦下，抄不回数据集。
    """
    failures: list[str] = []
    notes: list[str] = []

    schema_ok = bool(verbalized.ok and verbalized.stem and isinstance(verbalized.explanation, str))
    if not schema_ok:
        failures.append("schema: 槽位缺失或为空")
        notes.append("回的不是一个带非空题干与解析的 JSON 对象。两个槽位都要填上。")
        return Verification(schema=False, failures=tuple(failures), notes=tuple(notes))

    stem, explanation = verbalized.stem, verbalized.explanation

    # 1. 空槽位兜底：`_slot_check` 已挡一层，这里防别处构造的 Verbalized 漏进来。
    no_empty_slot = bool(stem.strip() and explanation.strip())
    if not no_empty_slot:
        failures.append("no_empty_slot: 题干或解析是空白")
        notes.append("题干或解析是空白。两个都要写。")

    # 2. 解析必须原样说出答案：答案由代码定死，解析说不出它，这道题就没有解。
    explanation_norm = normalize_for_match(explanation)
    answer_in_explanation = any(
        normalize_for_match(str(item)) and normalize_for_match(str(item)) in explanation_norm
        for item in spec.accepted
    )
    if not answer_in_explanation:
        failures.append(f"answer_in_explanation: 解析里没有出现答案 {spec.answer!r}")
        notes.append(f"解析里没有说出答案「{spec.answer}」。把它写进解析。")

    # 3. 题干不得泄露答案。上限按「题干本来就得写出它几次」算：**候选名单里的字**（选择题的
    #    `options`、题干必现串 `visible_subject` / `must_contain`）允许出现**一次**——那是读者
    #    要从中选的东西；其余情况是 0 次，题干提到答案，题目就退化成抄写题。
    #    出现两次以上则无论哪种情况都算泄露：实测 T10 写「其中只有「㐀」的部首不是「風」，
    #    因此应选出「㐀」」——题干连解析一起给了，读者照抄就算答对。
    stem_norm = normalize_for_match(stem)
    no_answer_leak = True
    for item in spec.accepted:
        needle = normalize_for_match(str(item))
        if not needle:
            continue
        leak_limit = answer_leak_limit(spec, item)
        hits = stem_norm.count(needle)
        if hits <= leak_limit:
            continue
        no_answer_leak = False
        failures.append(
            f"no_answer_leak: 题干里出现了 {hits} 次答案 {item!r}（上限 {leak_limit} 次）"
        )
        if spec.options:
            notes.append(
                f"题干里出现了 {hits} 次答案「{item}」。每个候选各列一次，别说哪个是对的"
                "——题干里不能有答案。"
            )
        else:
            notes.append(f"题干把答案「{item}」直接说出来了。问这件事，但别把答案写出来。")
        break

    # 4a. 必现串：`visible_subject`（问哪个字）+ `must_contain`（题型声明的其它识别条件，
    #     如反向题的读音、部件链的位置描述）必须原样出现在题干里——少了任何一个，读者都答不了。
    missing_subject = _missing_in_stem(stem, (spec.visible_subject, *spec.must_contain))
    subject_in_stem = not missing_subject
    if missing_subject:
        failures.append(f"subject_in_stem: 题干里没有写出必现串 {missing_subject}")
        notes.append(
            f"题干里没有写出必现串 {missing_subject}。这些串必须原样出现在题干里，"
            "不许用「这个读音」「上边的部件」这类指代代替。"
        )

    # 4b. 选择题：候选必须逐个出现在题干里。**这条不能省**——往返检验的载荷里带着 options，
    #     读者永远看得见候选，所以「题干没列候选」往返检验抓不到，只有这里能抓。
    options_listed = not _missing_in_stem(stem, spec.options)
    if not options_listed:
        failures.append(f"options_listed: 题干里没有列出候选 {_missing_in_stem(stem, spec.options)}")
        notes.append(
            f"题干里没有列出候选 {_missing_in_stem(stem, spec.options)}。"
            "每个候选都要写进题干，读者才有得选。"
        )

    # 4c. 题干问的必须是答案那件事：命中 `stem_asks_any` 里的任一种说法即可。
    #     实测 T1 的题干问读音与部首、答案却是笔画数——问错了方向，读者照着答必然错。
    stem_asks = (not spec.stem_asks_any) or any(item in stem for item in spec.stem_asks_any)
    if not stem_asks:
        failures.append(f"stem_asks: 题干没有问「{spec.stem_asks_any[0]}」那件事")
        notes.append(
            f"题干没有问「{spec.stem_asks_any[0]}」这件事。问这个，别问别的。"
        )

    # 4d. 隐藏主语（反向题的那个字）：题干写出它，题目就退化成查表题——读者不用看释义。
    no_hidden_subject = not spec.hidden_subject or normalize_for_match(spec.hidden_subject) not in normalize_for_match(stem)
    if not no_hidden_subject:
        failures.append(f"no_hidden_subject: 题干里写出了隐藏主语 {spec.hidden_subject!r}")
        notes.append(
            f"题干把该藏起来的那个字「{spec.hidden_subject}」写出来了。让读者从条件里认出它，"
            "别直接点名。"
        )

    # 4e. 字形结构：声明了 `structure_words` 的题，题干里一个结构词都不许出现（把答案那棵树
    #     用散文说一遍就是泄题），解析里出现过的结构词必须来自答案真实用到的那几个。
    structure_words = True
    if spec.structure_words:
        leaked = [word for word in _STRUCTURE_WORDS if word in stem]
        if leaked:
            structure_words = False
            failures.append(f"structure_words: 题干里出现了字形结构词 {leaked}")
            notes.append(f"题干里出现了字形结构词 {leaked}，等于把答案说出来了。删掉。")
        invented = [word for word in _STRUCTURE_WORDS if word in explanation and word not in spec.structure_words]
        if invented:
            structure_words = False
            failures.append(f"structure_words: 解析里的结构词 {invented} 不是答案用到的 {list(spec.structure_words)}")
            notes.append(
                f"解析里出现了答案用不到的结构词 {invented}。只能出现 {list(spec.structure_words)}。"
            )

    # 5. 数字必须来自载荷：产出里出现载荷没有的阿拉伯数字 = 凭空造数（笔画数、码位最容易编）。
    allowed = allowed_numbers(spec)
    invented = sorted((_numbers_in(stem) | _numbers_in(explanation)) - allowed)
    numbers_from_facts = not invented
    if invented:
        failures.append(f"numbers_from_facts: 正文里出现了载荷中没有的数字 {invented}")
        notes.append(f"正文里出现了载荷中没有的数字 {invented}。只能用载荷里给的那些。")

    # 6. 截断：`finish_reason == 'length'` 说明这次产出被按长度砍断，不是完整的一句话。
    not_truncated = verbalized.finish_reason != "length"
    if not not_truncated:
        failures.append("not_truncated: 产出被按长度截断")
        notes.append("产出被截断了，没写完。两个槽位都写短一点，保证能写完。")

    # 7. 题干必须是**中文自然语言**：判据是「题干里一个汉字都没有」这件事实——整段用英文作答的
    #    产出必然一个汉字都没有，一抓一个准。（程序侧文本已改成中文，「照抄出题说明」不再由这条
    #    拦：出题说明是名词性短语、不是问句，交给 L4.6 往返检验的 `is_question` 判。）
    stem_is_chinese = _cjk_count(stem) > 0
    if not stem_is_chinese:
        failures.append("stem_is_chinese: 题干里没有汉字（不是中文自然语言）")
        notes.append("题干里一个汉字都没有。用中文把这道题写出来。")

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
        # 这是 21 个闸门里**唯一**漏了模型侧说法的一条：漏了它，`_run_one` 只能把带闸门名的
        # `failures` 原文回灌给模型（实测 T13 因此反复在同一处摔，129 条 rejects 全是这条）。
        notes.append(
            "题干把释义原文整段搬了过来，读起来像照抄字典。释义只是素材：意思要讲全，"
            "但得用你自己的话换个说法，别连续照抄原句。"
        )

    # 8b. 题干不得**原样照搬**程序侧的题目说明：说明是「要问什么」的机读描述（「「㐀」的读音」），
    #     不是写给读者的话，逐字相同 = 模型没改写（主人原话：「禁止直接抄写字段的原始内容」）。
    #     判据是**归一化后逐字相同**，不是包含——正常改写（「「㐀」的读音是什么？」）比说明长，
    #     不会被误杀（离线 705,175 条模板题干实测 0 条与说明相同；语义池实测 23/1,321 条相同）。
    no_instruction_copy = normalize_for_match(stem) != normalize_for_match(
        getattr(spec, "instruction", "") or ""
    )
    if not no_instruction_copy:
        failures.append("no_instruction_copy: 题干与题目说明逐字相同（照抄说明，没有改写）")
        notes.append(
            "题干是把题目说明原样搬过来的，读者读到的不是一句问话。把说明当成「要问什么」，"
            "用一句完整的话把它问出来。"
        )

    # 9. 产出不得出现程序侧的说法：题干或解析里冒出「题干」「载荷」「槽位」「校验器」「IDS」
    #    这类词，说明模型在跟程序对话而不是跟读者说话——产出必须上下文无关。
    #    实测 T9 写「题干描述的位置链为…」、T3 写「IDS（ideographic description sequence）」。
    blob_lower = f"{stem}\n{explanation}".lower()
    leaked_terms = [term for term in _INTERNAL_TERMS if term.lower() in blob_lower]
    no_internal_terms = not leaked_terms
    if leaked_terms:
        failures.append(f"no_internal_terms: 产出里出现了程序侧的说法 {leaked_terms}")
        # **不把那些词抄进反馈**：把泄漏的词原样回灌，2B 会再抄一遍（实测第二轮照样写
        # 「根据提供的信息」）。只说「删掉指代出题过程的说法、直接陈述事实」，不点名。
        notes.append(
            "正文里混进了出题说明的说法，不是写给读者看的话。把所有指代出题过程、字段、"
            "「提供的」清单的说法删掉，直接用一句平实的中文把事实说出来。"
        )

    # 10. 载荷的**键名**不得出现在正文里：键名是程序侧的英文标识符（`stroke_total` / `codepoint` /
    #     `stem_must_contain`），读者只该看到键对应的**值**。实测 T1 解析写「「㐀」的 stroke_total 为 5」、
    #     T4 解析写「这个字的标准拼音是 xi3」这类字段名漏进散文。判据是子串比对（键名全是 ASCII，
    #     中文自然语言里出现即泄漏），与 `stem_is_chinese` 同类。
    leaked_fields = leaked_field_names(spec, f"{stem}\n{explanation}")
    no_field_names = not leaked_fields
    if leaked_fields:
        failures.append(f"no_field_names: 正文里出现了载荷键名 {leaked_fields}")
        notes.append(
            f"正文里出现了载荷的键名 {leaked_fields}。要写的是键对应的值，不是键名本身。"
        )

    # 10b. 题干不许把 `stem_asks_any` 里的说法**连成一串**念出来：那张表是程序侧的判据词表，
    #      不是可以念给读者听的清单。实测 T4 写「读作、怎么读、念作、读法、拼音、发音这些要求
    #      里都包含读音」——把整张表搬进了题干。正常问句命中一两个词不算（那是闸门本来就要求的），
    #      连成一串才算；只隔分隔符 / 并列连词的才计入（见 `_longest_marker_run`）。
    marker_run, marker_words = _longest_marker_run(stem, spec.stem_asks_any)
    no_marker_enumeration = marker_run < _MARKER_ENUMERATION_RUN
    if not no_marker_enumeration:
        failures.append(f"no_marker_enumeration: 题干把判据词连成一串念了出来 {marker_words}")
        notes.append(
            "题干把好几种问同一件事的说法连在一起念了出来。用自己的话说一次，那张清单不要念。"
        )

    # 11. 正文必须是中文散文：夹进来的英文词要么是载荷里的数据（`UTF-8` / `xi3` / `E3 90 80`），
    #     要么是标准术语白名单里的词；其余都是「没写中文」（实测 T3 写「结构 notation 字符串」）。
    stray_words = foreign_words(spec, f"{stem}\n{explanation}")
    no_foreign_words = not stray_words
    if stray_words:
        failures.append(f"no_foreign_words: 正文里出现了载荷中没有的英文词 {stray_words}")
        notes.append(
            f"正文里出现了载荷里没有的英文词 {stray_words}。这些部分用中文写，只保留载荷给的那些值。"
        )

    # 12. 正文里不得有转义残留：实测 2B 会吐出字面的 `\n`（JSON 里写成 `\\n`），
    #     发布出去读者看到的是「…的字。\n选项包括：…」。散文里本来就不该有反斜杠，
    #     所以判据是「有没有反斜杠」这件事实，与 `stem_is_chinese` 同类。
    no_escape_artifacts = "\\" not in stem and "\\" not in explanation
    if not no_escape_artifacts:
        failures.append("no_escape_artifacts: 正文里出现了转义残留（字面的反斜杠）")
        notes.append(
            "正文里有字面的反斜杠（被转义的换行）。写成一行平实的散文，不要转义字符。"
        )

    # 13. 产出里不得残留占位符记号：`[[1]]` 是程序给生僻字发的代号，回填发生在措辞之后；
    #     回填表里没有的记号没有任何东西能把它换掉，会原样进数据集。判据是子串比对（记号
    #     全是 ASCII 括号加数字，中文散文里出现即泄漏），与 `stem_is_chinese` 同类。
    leaked_placeholders = sorted(set(_PLACEHOLDER_RE.findall(f"{stem}\n{explanation}")))
    no_placeholder_leak = not leaked_placeholders
    if leaked_placeholders:
        failures.append(f"no_placeholder_leak: 产出里残留占位符记号 {leaked_placeholders}")
        notes.append(
            "正文里残留了 [[1]] 这类记号，那是程序给字发的代号。写出字本身，或者整个删掉。"
        )

    # 14. 解析不能只是**光秃答案**：`answer_in_explanation` 只查「答案串在不在解析里」，光秃
    #     答案天然在——真机实测 T9 回一个「七」就发布了（T3 回「⿱卝一」同理）。数据要的是**自然
    #     语言**，所以解析里除了答案本身，还得有中文说明。判据与其它闸门同类（子串比对），不规定
    #     该怎么措辞。全量离线数据 739,005 条复查：**0 条**会因此被判 C（`scan_bare_answer.py`）。
    stripped_explanation = explanation_norm
    for item in (spec.answer, *spec.accepted):
        needle = normalize_for_match(str(item))
        if needle:
            stripped_explanation = stripped_explanation.replace(needle, "")
    explanation_is_prose = _cjk_count(stripped_explanation) > 0
    if not explanation_is_prose:
        failures.append("explanation_is_prose: 解析只是光秃答案，没有一句说明")
        notes.append(
            f"解析里除了答案「{spec.answer}」本身，一个字的中文说明都没有。"
            "用一句完整的话把这件事说清楚。"
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
        no_instruction_copy=no_instruction_copy,
        no_internal_terms=no_internal_terms,
        no_field_names=no_field_names,
        no_foreign_words=no_foreign_words,
        no_marker_enumeration=no_marker_enumeration,
        no_escape_artifacts=no_escape_artifacts,
        no_placeholder_leak=no_placeholder_leak,
        explanation_is_prose=explanation_is_prose,
        failures=tuple(failures),
        notes=tuple(notes),
    )
