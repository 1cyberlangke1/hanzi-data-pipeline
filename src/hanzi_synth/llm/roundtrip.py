"""L4.6 读者检验（round-trip）：把题干交回模型，让它**只当读者**读一遍。

为什么要这一层：L4 的机械闸门只会做子串 / 集合比对（答案在不在解析里、候选列没列全、数字是不是
凭空造的），它判不了「这段话读起来像不像一道给读者的题」。而「像不像题」是开放集合，用枚举措辞
形状去追追不完——规则表会无限增长，永远收不了口。换成一个**闭集检验**：把题干交回模型当读者读，
读出「它是在向读者提问吗」「它说清了问哪个字吗」「它自己有没有冒出写给写题人的话」。

为什么这条判据不会变成「模型说了算」：三件事都是**读得出来的东西**，不需要汉字知识、不需要知道
答案，所以 2B 也做得到（实测 13/13 标定通过，见 findings.md）；而这三件事机械闸门都判不了。

**为什么不再让读者「答题」**：汉字题里有一大类是知识题（几画 / 读音 / 码位 / 第几页），读者本来
就得靠字典知识才能答，2B 更不可能知道生僻字的知识。早先的版本因此把「题干说清了是哪个字」的好题
全判死（实测 16 题型探针里 10 条被否，绝大多数是这种误杀）。现在读者答不出来时**自报是哪一种**：
「题干说清了，但我没这个知识」= 知识题（不否决）；「题干没说清问的是哪个字」= 缺条件（否决）。
只有当读者**信心十足地答出一个与载荷定死答案不符的东西**时，才按「题干把读者带偏了」否决。

判据（四条全过才算过）：
1. `is_question`：题干把作答要求交给了读者（陈述句 / 复述出题说明 → 否）；
2. `identifies_subject`：题干说清了问的是哪一个字 / 哪几个候选（机械闸门已确认识别串一个不缺时
   按 true 处理，见 `subject_identified_by_gates`——2B 在这条上会误报，实测 T13 / T8）；
3. `writer_phrases` 为空：题干里没有夹着写给写题人的话（「请列出所有候选」这类）；
4. `answer` 不是「缺条件」，且要么自报「知识题」、要么与载荷里定死的答案对得上。

**答案比对为什么不能只有子串**：读者答出的是一句**自然语言**（「读作 xié」「第二声」），标准答案
是程序定死的串（`xie2`）。子串比对判得了「读者那句话里有没有这串字」，判不了「读者说的与标准答案
是不是同一件事」——把自然语言那一层当字符串，写对了写法会被判错，写错了（否定 / 举例 / 另一个值）
也会被放过。所以子串只做**快速通过**（确定、不花配额）：过了就不再花配额；没过就交给 LLM 判
「读者写的这句自然语言与标准答案是不是同一件事」——**判定权在 LLM，规则不再单独否决自然语言**。

唯一的例外是**机器值**（编码串 / 码位 / 十进制数）：那类值没有语义层，仲裁只负责把读者写的自然语言
**解析成标准答案的写法**（见 `_is_machine_value`），逐位比对由程序做——2B 判「写法不同」不可靠
（实测把 `E4 B1 91` 判成与 `E3 90 89` 等价），让它「照抄出读者说的那个值」才是它做得到的。

读者自报「缺条件」时还有一道**旁证**：机械闸门 `subject_in_stem` / `options_listed` 确认题面里
该有的识别串一个不缺（那个字本身、读音、释义、部首、笔画，或整份候选名单），读者还说「没说清问
哪个字」就是误报（实测 T13 反向题 / T8 约束题两条好题因此被丢掉）——见 `subject_identified_by_gates`。

读者缺席（请求失败 / 输出不是合法 JSON / 字段缺席）**一律不算否决**：那是工具问题，不是数据问题。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..tasks.base import TaskSpec
from .provider import ChatClient
from .verbalizer import extract_json

# 读者自报「答不出来」的两种说法。它们是**读者视角的分类**，不是对写法的规定：
#   知识题 = 题干说清了问哪个字，只是读者没有这个字的知识（汉字题的主流形态，不否决）；
#   缺条件 = 题干根本没说是哪个字（这正是「题干不自足」的样子，否决）。
_KNOWLEDGE = "知识题"
_MISSING = "缺条件"

ROUNDTRIP_SYSTEM_PROMPT = """你是一个读者，手里只有下面这道汉字题本身，没有别的资料。

请判四件事，逐件只看题干（选项算题干的一部分）。

第一件 `is_question`：题干有没有把**作答要求**交给读者——读者读完知道「要回答什么」。
- 「…是什么」「哪一个…」「写出…」「…是多少」「…读作什么」→ true；
- 只是把一条信息陈述出来（「码位 U+3400 对应的字是「㐀」」），读者没有要回答的东西 → false。

第二件 `identifies_subject`：题干有没有说清问的是**哪一个字**。
- 题干里写出了一个具体的字（「㐀」「㜽」），或者列出了几个候选字，读者就知道问的是哪些字 → true；
- 题干只写「这个字」「下面这个字」而那个字根本没在题干里出现 → false。
- 注意：题干有没有列出候选**不影响**这一条；只要题干里说清问哪个字就是 true。
- 注意：靠识别条件认字的题（题干给出读音 / 释义 / 部首 / 笔画 / 码位，读者据此能认出是哪个字）
  也算说清了 → true。

第三件 `answer`：请只凭题干作答。
- 能只凭题干推出答案 → 把答案原样写出来（问字写那个字，问读音写拼音，问码位写码位，问部件写部件）；
- 题干已经说清了问哪个字，但你不知道这个字的知识（几画 / 读什么 / 码位 / 第几页）→ 写「知识题」三个字；
- 题干没说清问的是哪个字 → 写「缺条件」三个字。
- **不许猜**：题干里没写出来的知识（这个字读什么、几画、什么码位、什么字节）你是不知道的，
  这时一律写「知识题」，不要凭感觉编一个答案——编出来的答案会让这道题被误判成坏题。
- 编码串 / 码位 / 字节序列 / 笔画数 / 页码这类**定死的值**，题干里没写你就写不出来：
  一律「知识题」，**不要**随手编一个看起来像编码的串（例如「E1A0」）——那是猜。

第四件 `writer_phrases`：把题干里**写给写题人、而不是写给读者**的话**原样摘出来**，放进数组；一句都没有就写空数组 []。
- 这类话是**命令写题人怎么写题**的，例如「请列出所有候选」「不要出现数字」「必须原样引用」
  「请用自己的话解释」「让读者…」「题目要求…」「要求：」「题干必须…」；
- 题干让**读者**做的事（选出、写出、说出、算出）**不是**这类话，不要摘。

只输出一个 JSON 对象，不要 markdown 围栏，不要任何前后缀说明：
{"is_question": true 或 false, "identifies_subject": true 或 false, "answer": "...", "writer_phrases": [], "note": "一句话说明"}
"""

# 归一化时要剥掉的包装字符：模型常把答案写成「㐑」或者 "㐑" 或者 答案是：㐑
_WRAPPERS = "「」『』“”\"\'‘’（）()[]【】《》<>：: ，,。.、\n\t "


def normalize_answer(text: str) -> str:
    """输入：一段答案文本。输出：剥掉包装字符与空白后的归一化文本。
    预期：只做**去包装**，不做同义改写——同义改写会把「答成别的字」也放过。
    """
    return (text or "").strip().strip(_WRAPPERS).strip()


def answer_matches(spec: TaskSpec, answer: str) -> bool:
    """输入：TaskSpec + 模型答出的文本。输出：这次答题算不算对。

    预期：判据是**答案出现在模型输出里**（不是反过来）——模型常常多写一句
          「答案是㐑」，只要那句里带着定死的答案就算答对；反过来（答案里含模型输出）
          会把「模型只答了半个 IDS 串」也放过。
          载荷里的 `accepted` 是同一件事的多个合法写法，逐个试。
          再补一条**机器值的写法归一**（见 `_ascii_key`）：`E3 90 89` / `E39089` / `e3 90 89`、
          `U+3408` / `3408` 是同一个值，不必为此问模型。中文答案在这条路上取不到键，自动跳过。
    """
    got = normalize_answer(answer)
    if not got:
        return False
    for expected in (spec.answer, *spec.accepted):
        needle = normalize_answer(str(expected))
        if needle and needle in got:
            return True
    got_key = _ascii_key(got)
    if got_key:
        for expected in (spec.answer, *spec.accepted):
            key = _ascii_key(str(expected))
            if key and key == got_key:
                return True
    return False


# 机器值（编码串 / 码位 / 字节序列 / 数字）的比对键：只留 ASCII 字母数字。
_ASCII_ONLY_RE = re.compile(r"[^0-9a-z]+")


def _ascii_key(text: str) -> str:
    """输入：一段答案文本。输出：机器值的比对键（只留 ASCII 字母数字、小写、去掉 `U+` / `0x` 前缀）。

    预期：编码串 / 码位 / 字节序列这类值的「写法不同」只有**分隔符、大小写、前缀**三种，
          归一化后直接比即可。文本里带任何非 ASCII 字符时，非 ASCII 部分被丢掉——
          纯中文答案因此取到空键（调用方按「这条不适用」处理），不会把「答成别的字」放过。
    """
    lowered = re.sub(r"^(?:u(?=\+?[0-9a-f])|0x)", "", (text or "").strip().lower())
    return _ASCII_ONLY_RE.sub("", lowered)


# 机器值（编码串 / 码位 / 十进制数）的形状：十六进制串（可带 `U+` / `0x` 前缀、可带分隔符）或纯数字。
# 判据是**形状**，不是「纯 ASCII」：拼音 `xie2` 是 ASCII 却有语义层（`xié` 是同一个音），
# 把它当机器值逐位比会误杀；`E3 90 89` / `U+3408` / `13320` 没有语义层，只有写法。
_HEX_VALUE_RE = re.compile(r"^(?:u\+?|0x)?[0-9a-f](?:[\s\-:·．.,、]*[0-9a-f])*$")
_DECIMAL_VALUE_RE = re.compile(r"^\d+$")


def _is_machine_value(text: str) -> bool:
    """输入：一段答案文本。输出：它是不是**定死的机器值**（编码串 / 码位 / 十进制数）。

    预期：机器值没有语义层——`E3 90 89` 与 `E39089` 是同一个值，与 `E2 82 AC` 是两个值；
          `U+3408` 与 `3408` 是同一个码位，与 `U+3409` 是两个。这类值的仲裁只需要**解析**
          （把读者写的自然语言照抄成标准答案的写法），逐位比对由程序做。
    """
    stripped = (text or "").strip().lower()
    if not stripped:
        return False
    return bool(_HEX_VALUE_RE.match(stripped) or _DECIMAL_VALUE_RE.match(stripped))


def _traceable_to_stem(stem: str, answer: str) -> bool:
    """输入：题干 + 读者答出的文本。输出：这个答案能不能在题干里找到出处。

    预期：读者答错时用它区分两件事——**题干把读者带偏了**（答案的料就在题干里，读者照着推错了）
          与**读者在猜**（题干里根本没有这个值，是它自己编的）。只有前者是题目的毛病：
          汉字题一大类是知识题（几画 / 读音 / 码位 / 第几页），2B 读者没有那些知识，
          编一个值出来是常态，否决它等于把好题丢掉（实测 T14「码位是多少」被 2B 编出 `E1A0`
          而误杀）。判据是机械的：答案（机器值按 `_ascii_key` 归一）是否出现在题干里。
    """
    text = normalize_answer(answer)
    if not text:
        return False
    stem_text = stem or ""
    if text in stem_text:
        return True
    key = _ascii_key(text)
    return bool(key) and key in _ascii_key(stem_text)


def reader_answer_ok(spec: TaskSpec, answer: str) -> bool:
    """输入：TaskSpec + 读者答出的文本。输出：这条答案算不算「没把读者带偏」。

    预期：读者自报「缺条件」= 题干没说清问哪个字 → 否；
          自报「知识题」= 题干说清了、只是读者没有那个字的知识 → 是（汉字题的主流形态）；
          其余按**载荷里定死的答案**比对，对不上才算把读者带偏了。
    """
    text = normalize_answer(answer)
    if _MISSING in text:
        return False
    if _KNOWLEDGE in text:
        return True
    return answer_matches(spec, answer)


# 读者答出的是一句**自然语言**，标准答案是程序定死的串。子串比对只是**快速通道**（快、确定），
# 对不上时交给 LLM 仲裁——只做子串比对等于把自然语言那一层当字符串处理，读者的合理写法
# （「读作 xié」「第二声」）会被误杀。仲裁只在快速通道失败时发生，不浪费配额。
# 仲裁分两步：先把读者那句自然语言**解析**成标准答案的写法（`reader_value`），再判等价。
# 机器值（编码串 / 码位 / 十进制数）的第二步由程序做（逐位比 `reader_value`）——2B 判「写法不同」
# 不可靠，让它「照抄出读者说的那个值」才是它做得到的。
_EQUIV_SYSTEM_PROMPT = """你在核对一道汉字题的答案。

我会给你一个 JSON：`reader_answer`（读者读完题干后自己写出的答案，一句自然语言）、`expected`
（标准答案）、`accepted`（标准答案的其它合法写法，可能为空）。

第一步，把读者写出的答案**改写成标准答案的那种写法**，放进 `reader_value`：
- 读者说的与标准答案是同一件事，只是写法不同 → `reader_value` 写成标准答案的写法
  （标准答案是 `xie2`、读者写「读作 xié」→ `reader_value` 写 `xie2`）；
- 读者说的是另一个字 / 另一个数 / 另一件事、或者根本没说出答案 → `reader_value` 写空串 ""。

第二步，判 `equivalent`：读者写出的答案与标准答案**说的是不是同一件事**？
- 只是**写法**不同（拼音带声调符号与数字声调、大小写、空格与分隔符、同义说法）→ true；
- 说的是另一个字 / 另一个数 / 另一件事、或者读者没说出答案 → false。

编码串、码位、笔画数、页码这类**定死的值**要逐位比：
- `E3 90 89` 与 `E39089`、`e3 90 89` 是同一个值（只差分隔符与大小写）→ 等价；
- `E3 90 89` 与 `E2 82 AC`、`U+3402` 与 `U+3403`、`7` 与 `17` 是**不同的值** → 不等价；
  只要有一位数字或字母不同就是不等价，不要因为「看起来都是编码」就判等价；
- 拿不准时判 false：判错成等价会把错的答案留在数据里，判成不等价只是让程序再写一遍。

只输出一个 JSON 对象，不要 markdown 围栏，不要任何前后缀说明：
{"reader_value": "把读者答案改写成标准答案的写法（改不出来写空串）", "equivalent": true 或 false, "reason": "一句话理由"}
"""


def answer_equivalent(
    client: ChatClient, spec: TaskSpec, reader_answer: str, *, temperature: float = 0.0
) -> tuple[bool | None, str, str]:
    """输入：仲裁客户端 + TaskSpec + 读者写出的答案 + 温度。输出：(是否等价 or None, 理由, 错误)。

    预期：温度默认 0——仲裁要的是稳定复现；`None` + 非空错误 = 仲裁**缺席**（请求失败 /
          输出不合法 / 机器值给不出可比的写法），调用方退回快速通道的结论，绝不因为仲裁崩了
          就把好题判死。
          **机器值的第二步由程序做**（见 `_is_machine_value`）：标准答案是编码串 / 码位 / 十进制数时，
          仲裁只负责解析出 `reader_value`，逐位比对由这里做，且要求那个写法**在读者原话里找得到**
          （挡「把标准答案抄一遍当解析」）。解析不出来按缺席处理——退回快速通道，宁可重写一遍。
    """
    payload = {
        "reader_answer": reader_answer,
        "expected": spec.answer,
        "accepted": list(spec.accepted),
    }
    completion = client.complete(
        [
            {"role": "system", "content": _EQUIV_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=temperature,
    )
    if not completion.ok:
        return None, "", completion.error or "答案仲裁请求失败"
    data = extract_json(completion.text)
    if not isinstance(data, dict) or not isinstance(data.get("equivalent"), bool):
        return None, "", "答案仲裁输出不是合法 JSON（缺少布尔字段 equivalent）"
    equivalent = bool(data["equivalent"])
    reason = str(data.get("reason") or "")
    expected = str(spec.answer)
    if equivalent and _is_machine_value(expected):
        parsed = _ascii_key(str(data.get("reader_value") or ""))
        key = _ascii_key(expected)
        if not parsed or parsed != key or parsed not in _ascii_key(reader_answer):
            return None, "", "机器值的仲裁没给出可比对的写法（reader_value 不可用）"
    return equivalent, reason, ""


# L4 的 `answer_in_explanation` 是**子串比对**——它判得了「解析里有没有那串字」，判不了「解析说的是不是
# 那个答案」：「这个字的读音不是 xié」「除了 qiu1 之外都对」这类句子带着答案串，却把答案说反了。
# 所以子串只做**快速通过**：过了不花配额；没过才让 LLM 读一遍解析，判「解析陈述出来的答案是不是标准答案」。
_STATES_ANSWER_SYSTEM_PROMPT = """你在核对一道汉字题的解析。

我会给你一个 JSON：`explanation`（解析，一句自然语言）、`answer`（标准答案）、`accepted`
（标准答案的其它合法写法，可能为空）。

请判：解析里**实际陈述出来的答案**，与标准答案是不是同一个？

- 解析说「…是 X」「答案是 X」「因此选 X」而 X 与标准答案说的是同一件事（同义写法 / 另一种记法）→ true；
- 解析说 X 是**另一个字 / 另一个数 / 另一件事** → false；
- 解析把标准答案**排除掉**（「不是 X」「除了 X 都不对」）、或者**根本没说出答案是什么** → false；
- 不要只看解析里有没有出现那串字，要看解析**实际说出来的结论**：那串字被否定、被当成反例
  → false。

只输出一个 JSON 对象，不要 markdown 围栏，不要任何前后缀说明：
{"states_answer": true 或 false, "stated": "解析实际说出的答案（原文片段）", "reason": "一句话理由"}
"""


def build_states_answer_messages(spec: TaskSpec, verbalized: Any) -> list[dict]:
    """输入：TaskSpec + L3 产出。输出：OpenAI 格式 messages。预期：载荷里不含内部字段。"""
    return [
        {"role": "system", "content": _STATES_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "explanation": getattr(verbalized, "explanation", ""),
                    "answer": spec.answer,
                    "accepted": list(spec.accepted),
                },
                ensure_ascii=False,
            ),
        },
    ]


def explanation_states_answer(
    client: ChatClient, spec: TaskSpec, verbalized: Any, *, temperature: float = 0.0
) -> tuple[bool | None, str, str]:
    """输入：仲裁客户端 + TaskSpec + L3 产出 + 温度。输出：(解析是否说出了标准答案 or None, 理由, 错误)。

    预期：这是 `answer_in_explanation`（子串比对）的**语义那一半**——子串比对过了就不问它（不花配额），
          没过才问。`None` + 非空错误 = 仲裁**缺席**（请求失败 / 输出不合法），调用方维持机械结论，
          绝不因为仲裁崩了就把记录判死。载荷里只放**读者能看到的东西**（解析本身 + 标准答案），
          不放 instruction / facts 这些内部字段。
    """
    completion = client.complete(
        build_states_answer_messages(spec, verbalized), temperature=temperature
    )
    if not completion.ok:
        return None, "", completion.error or "解析核验请求失败"
    data = extract_json(completion.text)
    if not isinstance(data, dict) or not isinstance(data.get("states_answer"), bool):
        return None, "", "解析核验输出不是合法 JSON（缺少布尔字段 states_answer）"
    stated = str(data.get("stated") or "")
    reason = str(data.get("reason") or "")
    detail = f"解析说出的答案是「{stated}」" if stated else ""
    return bool(data["states_answer"]), "；".join(part for part in (detail, reason) if part), ""


def subject_identified_by_gates(verification: Any) -> bool | None:
    """输入：机械闸门结论（`Verification` 对象，或落盘后的 `verification` 字典）。
    输出：题干是否已经写全识别条件；拿不到那两个字段时返回 None。

    预期：这是「读者自报缺条件」的**旁证**——`subject_in_stem`（那个字本身 / 读音 / 释义 /
          部首 / 笔画等识别串一个不缺）与 `options_listed`（整份候选名单都在）两条闸门都过，
          说明题面确实说清了问的是哪些字，读者还答「缺条件」就是误报。实测 T13 反向题、T8 约束题
          两条好题就是这样被丢进 rejects 的（读者读得出条件，但自己认不出那个字，于是自报缺条件）。
          None = 没有旁证（老记录 / 缺字段），此时按老规矩否决——旁证只用来推翻误报，不用来放行。
    """
    if verification is None:
        return None
    if isinstance(verification, Mapping):
        subject, options = verification.get("subject_in_stem"), verification.get("options_listed")
    else:
        subject = getattr(verification, "subject_in_stem", None)
        options = getattr(verification, "options_listed", None)
    if isinstance(subject, bool) and isinstance(options, bool):
        return subject and options
    return None


@dataclass(frozen=True)
class RoundTrip:
    """输入：一次读者检验的结果。输出：不可变结果对象。
    预期：`error` 非空 = 这次没跑成（请求失败 / 输出不合法），调用方按**缺席**处理；
          `error` 为空且 `ok=False` 才是真正的「这段文字不合格」——四条判据任一条不过都算。
    """

    ok: bool
    is_question: bool = True
    identifies_subject: bool = True
    subject_checked: str = ""
    answer: str = ""
    answer_checked: str = ""
    writer_phrases: tuple[str, ...] = ()
    note: str = ""
    error: str = ""
    provider: str = ""
    model: str = ""

    @property
    def absent(self) -> bool:
        """输入：无。输出：这次检验是不是缺席。预期：缺席不参与发布决策。"""
        return bool(self.error)

    def as_dict(self) -> dict:
        """输入：无。输出：可 JSON 序列化的字典。预期：留 provider / model 便于事后标定。"""
        return {
            "ok": self.ok, "is_question": self.is_question,
            "identifies_subject": self.identifies_subject,
            "subject_checked": self.subject_checked,
            "answer": self.answer, "answer_checked": self.answer_checked,
            "writer_phrases": list(self.writer_phrases),
            "note": self.note, "error": self.error,
            "provider": self.provider, "model": self.model,
        }


def roundtrip_payload(verbalized: Any, spec: TaskSpec) -> dict:
    """输入：L3 产出 + TaskSpec。输出：给读者的载荷。

    预期：**只放题干与选项**，绝不放解析。解析里写着答案，放了它读者照抄就算「答对」，
          这道检验立刻退化成恒真——实测：带解析跑 6 条，4 条题干不自足
          （「请问这个汉字对应的 Unicode 编码是什么？」——读者不知道是哪个汉字）全被判过。
          选项是读者本来就看得到的东西，保留。
    """
    return {"stem": verbalized.stem, "options": list(spec.options)}


def build_roundtrip_messages(verbalized: Any, spec: TaskSpec) -> list[dict]:
    """输入：L3 产出 + TaskSpec。输出：OpenAI 格式 messages。预期：载荷里不含答案。"""
    return [
        {"role": "system", "content": ROUNDTRIP_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(roundtrip_payload(verbalized, spec), ensure_ascii=False)},
    ]


def roundtrip_answer(
    client: ChatClient,
    spec: TaskSpec,
    verbalized: Any,
    *,
    temperature: float = 0.0,
    answer_judge: ChatClient | None = None,
    subject_identified: bool | None = None,
) -> RoundTrip:
    """输入：客户端 + TaskSpec + L3 产出 + 温度 + 可选的答案仲裁客户端。输出：RoundTrip。

    预期：温度默认 0——这条检验要的是稳定复现，不是发挥；
          请求失败 / JSON 不合法，一律记 `error`（缺席），不判否决；
          四个判据字段里，布尔字段缺席按 true、`writer_phrases` 缺席按空、`answer` 缺席
          按「不比对」处理——**缺席不否决**是这一层的总规矩（工具问题不判数据死刑）。
          答案比对**快速通过 → LLM 判定**：子串比对（含机器值的写法归一）过了就是过了，不再花
          配额；没过且给了 `answer_judge` 时，**判定权交给 LLM**——读者写的是自然语言，只有它
          判得了「是不是同一个答案」。读者自报「知识题 / 缺条件」是分类不是答案，不送仲裁。
          机器值（编码串 / 码位 / 十进制数）的仲裁只负责解析 `reader_value`，逐位比对由程序做
          （实测 2B 判「写法不同」会把 `E4 B1 91` 说成与 `E3 90 89` 等价）。
          仲裁缺席时退回机械结论：读者答出的值在题干里没有出处（在猜）不否决，有出处
          （题干把读者带偏了）否决。
          读者答错时还要过一道**出处**关（`_traceable_to_stem`）：答案在题干里找得到出处
          （题干把读者带偏了）才否决；找不到出处是读者在猜，不否决。
          `subject_identified` 是「题干已经写全识别条件」的旁证（由机械闸门给出，见
          `subject_identified_by_gates`）：为 True 时读者自报的「缺条件」按误报处理，不否决；
          None（拿不到旁证）时按老规矩否决。
    """
    completion = client.complete(build_roundtrip_messages(verbalized, spec), temperature=temperature)
    if not completion.ok:
        return RoundTrip(
            ok=False, error=completion.error or "读者检验请求失败",
            provider=completion.provider, model=completion.model,
        )
    data = extract_json(completion.text)
    if not isinstance(data, dict):
        return RoundTrip(
            ok=False, error="读者检验输出不是合法 JSON",
            provider=completion.provider, model=completion.model,
        )
    is_question = data.get("is_question")
    is_question = True if not isinstance(is_question, bool) else is_question
    identifies_subject = data.get("identifies_subject")
    identifies_subject = True if not isinstance(identifies_subject, bool) else identifies_subject
    subject_checked = ""
    if not identifies_subject and subject_identified is True:
        # 旁证推翻读者的自报：机械闸门确认识别串一个不缺（那个字本身 / 读音 / 释义 / 部首 /
        # 笔画，或整份候选名单），读者还说「认不出是哪个字」就是误报——实测 T13 反向题、
        # T8 约束题两条好题因此被丢进 rejects。
        identifies_subject = True
        subject_checked = "规则：题干已含识别条件，读者自报认不出主语不予否决"
    raw_phrases = data.get("writer_phrases")
    writer_phrases = tuple(str(item) for item in raw_phrases) if isinstance(raw_phrases, list) else ()
    raw_answer = data.get("answer")
    answer, answer_ok, answer_checked = "", True, ""
    if isinstance(raw_answer, str) and raw_answer.strip():
        answer = raw_answer
        self_report = normalize_answer(raw_answer)
        self_reported = _MISSING in self_report or _KNOWLEDGE in self_report
        if _MISSING in self_report and subject_identified is True:
            # 旁证推翻读者的自报：机械闸门确认识别串一个不缺，那「没说清问哪个字」就是读者误报。
            answer_ok = True
            answer_checked = "规则：题干已含识别条件，读者自报缺条件不予否决"
        else:
            # **快速通过**：答案串（含机器值的写法归一）在读者那句话里找得到就算过。这一步只用来
            # 省配额——它判不了自然语言那一层（读者写「读作 xié」而标准答案是 `xie2`，子串里没有
            # `xie2`），判定权在下面那一步。
            answer_ok = reader_answer_ok(spec, raw_answer)
            answer_checked = "规则：快速通道"
            if not answer_ok and not self_reported:
                # 读者照着题干里的料答错 = 题干把读者带偏了（这是题目的毛病）；读者凭空编一个题干里
                # 没有的值 = 它在猜（汉字题一大类是知识题，2B 读者没有那些知识，编值是常态，不算
                # 题目的毛病）。这条机械旁证只在**仲裁缺席**时用来兜底。
                traceable = _traceable_to_stem(getattr(verbalized, "stem", ""), raw_answer)
                if answer_judge is not None:
                    equivalent, reason, error = answer_equivalent(answer_judge, spec, raw_answer)
                    if equivalent is None:
                        answer_ok = not traceable
                        fallback = (
                            "读者答出的值在题干里有出处（题干把读者带偏了）" if traceable
                            else "读者答出的值在题干里没有出处（在猜），不予否决"
                        )
                        answer_checked = f"仲裁缺席（{error}）；规则：{fallback}"
                    else:
                        answer_ok = equivalent
                        answer_checked = f"LLM 仲裁（{reason or '未给理由'}）"
                elif traceable:
                    answer_checked = "规则：读者答出的值在题干里有出处（题干把读者带偏了）"
                else:
                    answer_ok = True
                    answer_checked = "规则：读者答出的值在题干里没有出处（在猜），不予否决"
    return RoundTrip(
        ok=is_question and identifies_subject and not writer_phrases and answer_ok,
        is_question=is_question,
        identifies_subject=identifies_subject,
        subject_checked=subject_checked,
        answer=answer,
        answer_checked=answer_checked,
        writer_phrases=writer_phrases,
        note=str(data.get("note") or ""),
        provider=completion.provider,
        model=completion.model,
    )
