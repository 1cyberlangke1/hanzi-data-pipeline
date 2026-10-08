"""L3 自然语言化：LLM 唯一出场的地方。

契约：
- 输入是**已确定的事实 JSON**，LLM 只填题面（`Q`）与回答（`A`）两个位置；
- 输出必须是**单个 JSON 对象**，不得带 markdown 围栏或解释性前后缀；
- 数值 / 字符 / IDS 必须**原样引用** facts，不得换算、不得改写；
- 题干**不得包含答案**（这条由 L4 校验器保底，不只靠提示词）。

思考段在 provider 层已被无条件剥离，这里拿到的 text 一定是剥干净的。

提示词分两处，**都不给输入/输出贴标签**：SYSTEM_PROMPT（两句：一句任务 + 一句格式要求）与
FEW_SHOT_TURNS（四条**真实对话轮次**）——短、正向、只讲这一个任务，写法靠示例给，不靠规则墙。

每一处都是实测出来的（6 题型，本地小模型，温度 0.6，写入文件 synthesis/out/probe_prompt_v9 10
11 13 14 15）：
- 示范**留两条是下限**：砍成一条时 A 写成光秃答案的从 5/12 涨到 11/12；
- 格式要求写成**正向**一句「A 写成一句完整的话」：光秃答案 7/12 → 0/12（禁令它不执行）；
- instruction 用**名词性短语**（「「㐀」的总笔画数」）：整句照抄出题说明 9/12 → 0/12，
  且 12/12 都落成了问句。写成完整问句时小模型会把它原样搬进题干，数据集里就是程序回声；
- 系统提示**再砍一句**（删掉「数字、字符、[[1]] 一律原样照抄」）：光秃答案 3/12 → 0/12、
  tier A 6/12 → 7/12——多出来的那半句小模型根本没执行，只是白占上下文；
- 示范**加到三条**（第三条是反向认字）+ 提示内容去掉与顶层同值的 options / nswer_zi：
  6 型 tier A 5/12 → 8/12，T13 的 subject_in_stem 失败 2/12 → 0~1/12。
- 校验要求的串（候选 / 位置链 / 读音 + 释义）**写进 `instruction`**：它们是「要问什么」的一部分，
  只放在 facts 里小模型不捞——四个选择题型候选列全 0/8、tier A 0/8；写进说明后 8/8 列全、
  tier A 11/12（probe_weave.jsonl）。T9 的必现位置词 0/2 → 2/2，T13 的读音 3/3、释义 3/3
  （probe_t13.jsonl）。

`instruction` 是题型自己给的一句「要问什么」，不带「问读者」这类程序口吻——带了口吻，
小模型会连口吻一起抄进产出。说明里的生僻字与候选一样换成占位符（`payload_for` 统一做），
模型照抄说明时不必自己敲生僻字。
另外重试时会把上一轮被拒的原因作为 `feedback` 追加一轮用户消息（reflexion），
把「猜哪里错了」变成「按原因改」。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from json_repair import loads as repair_json_loads

from ..tasks.base import TaskSpec
from ..verify.checks import JUDGE_FEEDBACK_KEY
from .provider import ChatClient


# 两句话：一句任务 + 一句输出格式，**一个标签都不贴**：「素材 / 产出」这类标签是元语言，小模型会连
# 口吻一起抄进产出。哪份是输入、哪份是输出由 `FEW_SHOT_TURNS` 的**角色**说明——用户消息就是
# 输入 JSON，助手消息就是该回的 JSON，不需要任何前缀。
# 「A 写成一句完整的话」这半句是实测加的：不加时 12 次里 7 次的 A 是光秃答案（「5」「⿱卝一」），
# 加了之后同一批题 0 次——小模型只执行**正向的格式要求**，禁令（「别只写答案」）它不执行。
# 两句新增的要求都是**正向**的（禁令小模型不执行，实测），各自对应一个量出来的毛病：
#   「换一种问法开头」：产出数据里 T13 有 67% 的题干都以「有一个字读作」开头、T17 只有 84 种句式
#     （最常见那句占 34%）——void_openings 只是把用过的题干列出来，小模型不会自己去比，得明说；
#   「事实照原意写」：实测 T13 把 䢂 的释义「粤语读音lip1、为英语"lift"的音译」写成了「意思与机械相关」，
#     既丢原意又和另一个字（䡃「绞车机械」）撞成逐字相同的题干。
# 原先还有半句「数字、字符、[[1]] 一律原样照抄」，实测多余：删掉后光秃答案 3/12 → 0/12、
# tier A 6/12 → 7/12（probe_prompt_v13），数值照样照抄——示范已经把这件事演出来了。
SYSTEM_PROMPT = "把给你的 JSON 写成一道自然的汉字问答：换一种问法开头，别和已给的那些重复；事实照原意写，别换成更笼统的说法。只回一个 JSON 对象：{\"Q\": \"...\", \"A\": \"...\"}，A 写成一句完整的话。"

# 四条示范轮次（输入, 输出），都与真实提示内容**同形**：
# 1. 说明是**名词性短语**（「「木」的总笔画数」），跟构造器给的写法一致——说明写成完整问句时
#    小模型会整句照抄（实测 12 次里 9 次），改成名词性短语后 0 次；
# 2. 选择题的说明**把候选写在句首**（`[[1]]、[[2]]、[[3]]、[[4]] 里总笔画数最少的那个字`）：
#    校验要求题干逐个列出候选，而只写「候选里」时小模型一个都不列（4 个选择题型 × 2 条：候选
#    列全 0/8、tier A 0/8；写成说明的主语后 8/8 列全，probe_weave.jsonl）。示范必须与真提示内容
#    同形，否则小模型学到的是另一套形状。
# 3. / 4. 两条是**反向认字**（靠读音 + 释义认出那个字，再问它的部件结构式）：前两条都是正向题，
#    小模型没见过「题干要把识别条件写出来」这种形状，T13 的题干只写「这个字…」——补上示范后
#    subject_in_stem 失败从 2/12 降到 0~1/12（probe_prompt_v14 / v15）。
#    **为什么是两条、且两条都问「部件结构式」**（实测：本地小模型，温度 0.6，同一批计划逐臂只换示范，
#    写入文件 synthesis/out/probe_t13_style.log 与 probe_t13_style_ef.log）：
#    | 反向示范 | 唯一句式 | 最常见句式占比 | 开头种数 | tier A |
#    | 1 条，问「它是哪个字」 | 22/32 | 34.4% | 3 | 29/32 |
#    | 1 条，问「部件结构式」 | 38/48 | 16.7% | 4 | 41/48 |
#    | 2 条，都问「部件结构式」 | 46/48 | 4.2% | 14 | 39/48 |
#    两个量出来的毛病：①**示范的问法会被整句搬走**——问「它是哪个字」时 T13 题干真的写
#    「它是哪个字？」，而 T13 的答案不是那个字、是它的部件结构式，校验判 tier C（3/32）；
#    问「部件结构式」时同一批 39/48 落 tier A。②**只有一条示范时小模型只学得到一个形状**，
#    最常见句式占 34.4%、开头只有 3 种；给两条不同开头的（「哪一个字…」/「意思是…，读音是…」）
#    才有第二个形状可挑——最常见句式掉到 4.2%、开头 14 种。
#    T17 同批对照（20 条一臂）：tier A 14/20 → 18/20，句式多样性不降——加这两条不伤别的题型。
FEW_SHOT_TURNS: tuple[tuple[dict[str, Any], dict[str, str]], ...] = (
    (  # 1. 正向题：说明是名词性短语，题干写成问句
        {"instruction": "「木」的总笔画数", "facts": {"zi": "木", "stroke_total": 4}, "answer": "4"},
        {"Q": "「木」这个字总共几画？", "A": "「木」一共 4 画。"},
    ),
    (  # 2. 选择题：候选写在说明句首，题干逐个列出
        {
            "instruction": "[[1]]、[[2]]、[[3]]、[[4]] 里总笔画数最少的那个字",
            "facts": {"strokes": {"[[1]]": 5, "[[2]]": 24, "[[3]]": 12, "[[4]]": 9}},
            "answer": "[[1]]",
            "options": ["[[1]]", "[[2]]", "[[3]]", "[[4]]"],
        },
        {
            "Q": "[[1]]、[[2]]、[[3]]、[[4]] 这四个字里，哪一个的笔画最少？",
            "A": "笔画最少的是 [[1]]，它只有 5 画。",
        },
    ),
    (  # 3. 反向认字：问句以「哪一个字」开头
        {
            "instruction": "读作 hǎo、释义是「美好。\n友爱。」的那个字的部件结构式",
            "facts": {
                "meaning_source": "美好。\n友爱。",
                "ids": ["⿰女子"],
                "radical": "女",
                "stroke_total": 6,
                "reading_display": ["hǎo"],
            },
            "answer": "⿰女子",
            "hidden_subject": "好",
        },
        {
            "Q": "哪一个字读作 hǎo、意思是美好和友爱？它的部件结构式是什么？",
            "A": "它的部件结构式写作 ⿰女子。",
        },
    ),
    (  # 4. 反向认字：同一件事的另一种开头（从释义切入）
        {
            "instruction": "读作 míng、释义是「明亮。明白。」的那个字的部件结构式",
            "facts": {
                "meaning_source": "明亮。\n明白。",
                "ids": ["⿰日月"],
                "radical": "日",
                "stroke_total": 8,
                "reading_display": ["míng"],
            },
            "answer": "⿰日月",
            "hidden_subject": "明",
        },
        {
            "Q": "意思是明亮、明白，读音是 míng 的字，它的部件结构式怎么写？",
            "A": "它的部件结构式写作 ⿰日月。",
        },
    ),
)


# 示范提示内容的**原文**（与 `_few_shot_messages` 里发给模型的一模一样）：`payload_of` 靠它把示范
# 与「这一次真要写的那道题」分开——两边形状相同，不排掉示范就会把示范当成真提示内容。
_FEW_SHOT_BLOBS = frozenset(
    json.dumps(item, ensure_ascii=False, indent=2) for item, _reply in FEW_SHOT_TURNS
)


def _few_shot_messages() -> list[dict[str, str]]:
    """输入：无。输出：示范轮次的 messages（用户消息=输入 JSON，助手消息=输出 JSON）。
    预期：单题与批量共用的静态前缀；角色本身就说明了哪份是输入、哪份是输出，所以示范里一个
          标签都不写——标签会被小模型抄进产出。
    """
    out: list[dict[str, str]] = []
    for demo_in, demo_out in FEW_SHOT_TURNS:
        out.append({"role": "user", "content": json.dumps(demo_in, ensure_ascii=False, indent=2)})
        out.append({"role": "assistant", "content": json.dumps(demo_out, ensure_ascii=False)})
    return out


# 重试反馈的开头与结尾，**一律中文**：合成的是中文数据，程序侧文本也一律中文——
# 中英混排会让小模型多一道翻译并译错（真机把 `reading` 译成「阅读」）。
# 反馈正文里的原因来自 L4 校验；模型若把反馈原文抄进产出，会被 `no_internal_terms` 拦下。
FEEDBACK_HEADER = "刚才那版没通过，按下面的意见重写。"
# 结尾必须再写一次输出格式：反馈很长，小模型读完就丢掉了系统提示里的格式要求（实测第二轮
# 整段输出散文 / 把提示内容原样回显）。放在**最后**一行是刻意的——离生成最近的那句最管用。
FEEDBACK_FOOTER = '只回一个 JSON 对象：{"Q": "...", "A": "..."}，A 写成一句完整的话'

# 批量模式的补充系统提示：用户给的 JSON 从「一道题」变成「一个 items 数组」。
# 只有这一段的差别，每道题的写作要求完全不变：批量不该成为放松要求的借口，质量对比才有意义。
BATCH_SUFFIX = """

这一次给你的是一个 items 数组，每项带 id。给每一项各写一份 Q/A，按 id 原样返回；每一项换一种句式开头，别让整批读起来一个样：
{"results": [{"id": 1, "Q": "...", "A": "..."}, ...]}
"""


def _feedback_block(feedback: Sequence[str]) -> str:
    """输入：上一轮被拒的原因列表。输出：反馈正文（逐条列出）。
    预期：只把**原因本身**回灌，不附规则表里的「改法」——模型看到的是「上一轮为什么被拒」，
          它自己会改；规则表已经删掉了。
    """
    return "\n".join(f"- {item}" for item in feedback)


def _batch_feedback_block(feedback: Mapping[str, Sequence[str]]) -> str:
    """输入：按题分组的失败原因（键是「第 N 题（题型）」这类标签）。输出：反馈正文。
    预期：逐题分组列出——批量模式下模型必须知道哪条原因属于哪一道题，否则会拿 A 题的
          原因去改 B 题。空映射返回空串（调用方只在非空时才追加这条消息）。
    """
    lines: list[str] = []
    for label, reasons in feedback.items():
        lines.append(f"{label}：")
        lines.extend(f"- {item}" for item in reasons)
    return "\n".join(lines)


@dataclass(frozen=True)
class Verbalized:
    """输入：一次措辞调用的结果。输出：不可变结果对象。
    预期：`ok=False` 时 `stem` 为空且 `error` 非空；
          `finish_reason` 原样带出引擎的收尾原因——`length` 说明产出被按长度截断，
          L4 的 `not_truncated` 靠它判「这次产出根本不完整」。
    """

    ok: bool
    stem: str = ""
    explanation: str = ""
    provider: str = ""
    model: str = ""
    error: str | None = None
    finish_reason: str = ""


def _is_rare_char(char: str) -> bool:
    """输入：单个字符。输出：它是不是「生僻」到不该让模型自己敲的字。
    预期：基本区（U+4E00–U+9FFF）之外、又不是 ASCII 可见字符的，都算生僻——
          扩展 A 区（㐀 一类）与扩展 B 区以上（𬚮 一类）小模型经常写错或整个写丢。
    """
    if not isinstance(char, str) or len(char) != 1:
        return False
    point = ord(char)
    return point > 0x9FFF or 0x3400 <= point <= 0x4DBF


def _alias_map(spec: TaskSpec) -> dict[str, str]:
    """输入：TaskSpec。输出：{占位符: 真字} 对照表（只收生僻字，顺序稳定）。
    预期：提示内容里凡是要求模型**写出某个生僻字**的位置（选项、隐藏主语）都换成占位符，
          模型只写 `[[1]]` 这类记号，真字由程序回填——产出里的生僻字一定是程序放的，
          模型不必自己敲，也就不会敲错、敲丢。
    """
    chars: list[str] = []
    for char in ([spec.visible_subject] if spec.visible_subject else []) + list(spec.options) + ([spec.hidden_subject] if spec.hidden_subject else []):
        if _is_rare_char(char) and char not in chars:
            chars.append(char)
    return {f"[[{index}]]": char for index, char in enumerate(chars, start=1)}


def _alias(text: str, aliases: Mapping[str, str]) -> str:
    """输入：一段文本 + 占位符对照表。输出：把表里的真字换成占位符后的文本。"""
    for token, char in aliases.items():
        text = text.replace(char, token)
    return text


# 记号写松了的形态：`[[4] ]` / `[[ 4 ]]`。只认「[[]] 里包一个整数」这一种形状，
# 不碰正文里别的方括号。
_ALIAS_SPACING = re.compile(r"\[\[\s*(\d+)\s*\]\s*\]")


def _restore_aliases(text: str, aliases: Mapping[str, str]) -> str:
    """输入：模型产出的文本 + 占位符对照表。输出：把占位符换回真字后的文本。
    预期：只替换对照表里的记号；模型若自己敲了生僻字，这里不动它，交给校验判。
    """
    # 模型偶尔把记号里的方括号转义（`[\[1\]]`）：转义残留本来就是 `no_escape_artifacts` 要拦的
    # 形态（实测 T11 出现过一次），先还原成记号本身再回填，免得「明明写了却被判成没写」。
    text = text.replace("\\[", "[").replace("\\]", "]")
    # 记号内部也偶尔被写松（`[[4] ]`、`[[ 4 ]]`）：先把记号内部的空白收掉再回填。
    # 实测（probe_noinstr.jsonl，T10 U03402）漏一个空格就让占位符原样落进题干、被校验判 tier C——
    # 那条数据本来是对的，白扔一次重试。
    text = _ALIAS_SPACING.sub(r"[[\1]]", text)
    for token, char in aliases.items():
        text = text.replace(token, char)
    return text


def _with_restored_aliases(result: Verbalized, aliases: Mapping[str, str]) -> Verbalized:
    """输入：一次措辞结果 + 占位符对照表。输出：把 stem / explanation 里占位符换回真字的结果。
    预期：失败结果、空对照表一律原样返回。
    """
    if not result.ok or not aliases:
        return result
    return replace(
        result,
        stem=_restore_aliases(result.stem, aliases),
        explanation=_restore_aliases(result.explanation, aliases),
    )




# 发给模型的 `facts` 要做两件事：只留这道题要引用的键，并去掉顶层已经给过的字段。
#
# 一、**只留 `spec.payload_facts` 声明的键**。构造器的 `facts` 是给校验 / 离线渲染用的全集
#     （判分要读它、干扰项要读它、模板要读它），措辞模型只需要「写这道题要引用的那几个值」。
#     全发时模型会拿别的题的材料当素材——实测 T16 的 `codepoint_forms` 里混着 UTF-8 字节序列，
#     产出写成「给出码位 E3 90 8B，让读者从下列选项中选出对应的字」（那条已在构造器侧收窄）；
#     T1 问笔画却带着 `radical` 与 `ids`。示范（`FEW_SHOT_TURNS`）的 facts 形状就是裁剪后的样子。
#     实测（probe_payload_ab，本地小模型，11 型 × 4 字 × 2 臂）：tier A 不降、不同题干数不降，
#     平均提示内容 206 → 172 字符。
#     裁**过头**会掉质量：T16 一开始只发 `codepoint`，产出写十进制「13314」时
#     `numbers_from_facts` 判「这个数字提示内容里没有」——所以 T16 的键里有 `codepoint_forms`。
#
# 二、**去掉顶层已经给过的字段**：`options`（顶层那份是占位符化过的）与 `answer_zi`
#     （与顶层 `answer` 同值）。留着它们不只是白占上下文——实测小模型会照 facts 里那份
#     **原始生僻字**写题干，绕开占位符回填（probe_prompt_v13 里 T11/T16 的题干直接把生僻字敲了出来）。
def _payload_facts(spec: TaskSpec) -> dict[str, Any]:
    """输入：TaskSpec。输出：给模型的 facts。
    预期：只留 `spec.payload_facts` 声明的键（声明为空 = 发全部，只给手工构造的 spec 用）；
          与顶层同值的 `options` / `answer_zi` 去掉。
    """
    facts = dict(spec.facts)
    if spec.payload_facts:
        facts = {key: value for key, value in facts.items() if key in spec.payload_facts}
    if spec.options and facts.get("options") == list(spec.options):
        del facts["options"]
    if facts.get("answer_zi") == spec.answer:
        del facts["answer_zi"]
    return facts


def _alias_deep(value: Any, aliases: Mapping[str, str]) -> Any:
    """输入：提示内容里的任意 JSON 值 + 占位符对照表。输出：把其中所有生僻字换成记号后的同形值。
    预期：**整份提示内容用同一套记号**——模型看到的是一个自洽的世界（说明、候选、答案、facts 里的
          字都是同一个记号），它才分得清「笔画最少的是哪一个」；只给候选取记号、facts 里留真字时
          小模型无从对照，会写出自相矛盾的解析（实测 T11 产出「笔画最少的是𡲗，对应字是㐂」——
          「𡲗」是它瞎挑的候选，「㐂」是从 answer 抄的）。
    """
    if isinstance(value, str):
        return _alias(value, aliases)
    if isinstance(value, dict):
        return {_alias_deep(key, aliases): _alias_deep(item, aliases) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_alias_deep(item, aliases) for item in value]
    return value


def payload_for(spec: TaskSpec, avoid: Sequence[str] = ()) -> dict[str, Any]:
    """输入：TaskSpec + 已经用过的题干开头。输出：给模型的事实提示内容（单题与批量共用同一份字段）。
    预期：只放写这道题**真正需要**的字段——要问什么、素材、答案、候选、依据；
          空值字段一律不出现在提示内容里（小模型的上下文很贵，塞进去只会干扰它）。
          题干里必须原样出现的串（哪个字、哪些候选、位置链、读音）已经写在 instruction 与
          facts 里，不单列成字段：实测单列之后小模型会把「必须出现这些串」这条**要求本身**
          抄进题干。
    """
    aliases = _alias_map(spec)
    payload: dict[str, Any] = {
        # **不给** `task_id`：实测小模型会把「T9」原样抄进题干（16y 的 T9 写「请辨识T9编码链
        # 的最内层部件」）——那是内部编号，对读者毫无意义。
        # 说明是题干的主要来源（候选 / 位置链 / 读音本来就写在说明里），里面的生僻字同样换记号：
        # 模型照抄说明时不该自己去敲生僻字——敲错就是另一道题。
        "instruction": _alias_deep(spec.instruction, aliases),
        "facts": _alias_deep(_payload_facts(spec), aliases),
        "answer": _alias_deep(spec.answer, aliases),
    }
    if spec.options:
        payload["options"] = _alias_deep(list(spec.options), aliases)
    if spec.proof:
        payload["proof"] = _alias_deep(list(spec.proof), aliases)
    # 反向题的隐藏主语：题干要靠释义与读音让读者认出它，不能直接写出来
    if spec.hidden_subject:
        payload["hidden_subject"] = _alias_deep(spec.hidden_subject, aliases)
    # 问字形结构的题（T13）：题干里不许出现这里列出的位置词
    if spec.structure_words:
        payload["structure_words"] = list(spec.structure_words)
    # 已经用过的题干开头：这是**负例**——限制重复，不是限制表达。
    if avoid:
        payload["avoid_openings"] = list(avoid)
    return payload


def build_messages(
    spec: TaskSpec, feedback: Sequence[str] = (), avoid: Sequence[str] = ()
) -> list[dict[str, str]]:
    """输入：TaskSpec + 上一轮被拒的原因 + 已用过的开头（都可空）。输出：OpenAI 格式 messages（单题）。
    预期：示范走 `FEW_SHOT_TURNS` 的真实轮次（角色即输入/输出，不贴标签），真实提示内容排在**最后**
          ——离生成最近的那条消息最管用；有 feedback 时再追加一条用户消息把失败原因回灌（reflexion）。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_few_shot_messages(),
        {"role": "user", "content": json.dumps(payload_for(spec, avoid), ensure_ascii=False, indent=2)},
    ]
    if feedback:
        reasons = _feedback_block(feedback)
        messages.append({"role": "user", "content": f"{FEEDBACK_HEADER}\n{reasons}\n{FEEDBACK_FOOTER}"})
    return messages


def build_batch_messages(
    specs: Sequence[TaskSpec], feedback: Mapping[str, Sequence[str]] | None = None
) -> list[dict[str, str]]:
    """输入：多个 TaskSpec + 按题分组的失败原因（可空）。输出：OpenAI 格式 messages（批量）。
    预期：所有题的事实放进一个 `items` 数组，`id` 从 1 开始（顺序 = 传入顺序）；
          示范轮次与单题模式同一份，真实提示内容仍排在**最后**；批量只改请求打包方式，不改每道题的写法。
    """
    items: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        payload = payload_for(spec)
        items.append({"id": index, **payload})
    body: dict[str, Any] = {"items": items}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + BATCH_SUFFIX},
        *_few_shot_messages(),
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, indent=2)},
    ]
    if feedback:
        reasons = _batch_feedback_block(feedback)
        messages.append({"role": "user", "content": f"{FEEDBACK_HEADER}\n{reasons}\n{FEEDBACK_FOOTER}"})
    return messages


def extract_json(text: str) -> dict | None:
    """输入：模型输出文本。输出：解析出的 dict，失败时 None。
    预期：整串交给 `json_repair.loads`——它自己先试标准解析（正常输出走快路径），
    失败才动手修：裸换行、结构位多余逗号、缺引号、markdown 围栏、前后夹的散文它都认
    （社区成熟实现，不自造轮子）。解析层容错**不改语义、不放松校验**：
    修出来不是 JSON 对象（散文、数组、空串）一律返回 None，交给上层重试。
    """
    if not text:
        return None
    try:
        data = repair_json_loads(text)
    except Exception:  # noqa: BLE001 - 修不出来就是修不出来，不能让解析层把管线崩掉
        return None
    return data if isinstance(data, dict) else None


def payload_of(messages: Sequence[Mapping[str, Any]]) -> dict:
    """输入：`build_messages` 产出的 messages。输出：其中的事实提示内容（含 facts / answer 的那一条）。
    预期：从后往前找第一条能解析成 JSON 对象的 user 消息——重试时会追加反馈消息，
          所以「提示内容永远是 messages[-1]」**不再成立**，回放 / 审计 / 假客户端都要用这个函数取提示内容。
          取不到时返回空字典，不抛异常。
    """
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if content in _FEW_SHOT_BLOBS:  # 示范轮次：形状与真提示内容一样，但它不是这一次要写的题
            continue
        data = extract_json(content)
        if isinstance(data, dict) and "answer" in data:
            return data
    return {}


def batch_items_of(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    """输入：`build_batch_messages` 产出的 messages。输出：`items` 数组（取不到时返回空列表）。
    预期：与 `payload_of` 同源的取法——从后往前找第一条带 `items` 的 user 消息；
          批量模式没有单题提示内容，回放 / 假客户端要用这个函数。
    """
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        data = extract_json(str(message.get("content", "")))
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return [item for item in data["items"] if isinstance(item, dict)]
    return []


def _slot_check(
    data: Mapping[str, Any], *, provider: str, model: str, label: str = "",
    finish_reason: str = "",
) -> Verbalized:
    """输入：模型输出解析出的对象 + 语体/provider/模型 + 可选的题目标签。输出：Verbalized。
    预期：题面与回答缺一不可、必须是非空字符串。单题与批量共用这一份校验——
          批量路径**不允许**悄悄放宽位置要求，否则 A/B 就变成「两套标准比质量」。
          模型按提示词回的是 `Q` / `A`（`stem` / `explanation` 是产出记录里的叫法），两边都收。
    """
    prefix = f"{label}：" if label else ""
    stem = data.get("Q") or data.get("stem")
    explanation = data.get("A") or data.get("explanation")
    if not isinstance(stem, str) or not stem.strip():
        return Verbalized(ok=False, provider=provider, model=model,
                          error=f"{prefix}缺少 Q 位置（题面）", finish_reason=finish_reason)
    if not isinstance(explanation, str):
        return Verbalized(ok=False, provider=provider, model=model,
                          error=f"{prefix}缺少 A 位置（回答）", finish_reason=finish_reason)
    return Verbalized(
        ok=True,
        stem=stem.strip(),
        explanation=explanation.strip(),
        provider=provider,
        model=model,
        finish_reason=finish_reason,
    )


def verbalize(
    client: ChatClient,
    spec: TaskSpec,
    *,
    temperature: float = 0.6,
    feedback: Sequence[str] = (),
    avoid: Sequence[str] = (),
) -> Verbalized:
    """输入：客户端 + TaskSpec + 温度 + 上一轮失败原因 + 已用过的开头。输出：Verbalized。
    预期：JSON 解析失败、缺位置、位置非字符串一律判失败，绝不半信半疑地放行；
          feedback 非空时把它作为额外一轮用户消息传给模型（reflexion）。
    """
    completion = client.complete(build_messages(spec, feedback, avoid), temperature=temperature)
    if not completion.ok:
        return Verbalized(ok=False, provider=completion.provider,
                          model=completion.model, error=completion.error)

    data = extract_json(completion.text)
    if data is None:
        # 把模型**实际吐了什么**带进错误里：光写「不是合法 JSON」对诊断零信息，
        # 真机上 T10/T16 反复栽在这一句，看不到原文就只能猜（8 条没通过里 5 条是它）。
        head = " ".join((completion.text or "").split())[:200]
        return Verbalized(ok=False, provider=completion.provider,
                          model=completion.model, error=f"模型输出不是合法 JSON：{head}")
    return _with_restored_aliases(
        _slot_check(data, provider=completion.provider, model=completion.model,
                    finish_reason=completion.finish_reason or ""),
        _alias_map(spec),
    )


def verbalize_batch(
    client: ChatClient,
    specs: Sequence[TaskSpec],
    *,
    temperature: float = 0.6,
    feedback: Mapping[str, Sequence[str]] | None = None,
) -> list[Verbalized]:
    """输入：客户端 + 多个 TaskSpec + 温度 + 按题分组的失败原因。
    输出：与 `specs` **等长且顺序一致**的 Verbalized 列表。
    预期：整批一次请求；结果按 `id` 对齐，缺项 / 位置不合格只让**那一道**失败，不串位。
          整批请求失败（网络或整体 JSON 解析失败）时所有题返回同一个 error——
          批量的失败是整批的，不能假装某一道成功。
    """
    count = len(specs)
    completion = client.complete(build_batch_messages(specs, feedback), temperature=temperature)

    def all_failed(error: str) -> list[Verbalized]:
        """输入：错误描述。输出：整批同错的列表。预期：只用于整批级失败。"""
        return [
            Verbalized(ok=False, provider=completion.provider, model=completion.model, error=error)
            for _ in range(count)
        ]

    if not completion.ok:
        return all_failed(completion.error or "请求失败")
    data = extract_json(completion.text)
    raw = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return all_failed("批量输出不是合法 JSON（缺少 results 数组）")

    by_id: dict[int, Mapping[str, Any]] = {}
    for item in raw:
        if isinstance(item, Mapping) and isinstance(item.get("id"), int):
            by_id[item["id"]] = item

    out: list[Verbalized] = []
    for index in range(1, count + 1):
        item = by_id.get(index)
        if item is None:
            out.append(
                Verbalized(ok=False, provider=completion.provider,
                           model=completion.model, error=f"批量输出缺少 id={index}")
            )
            continue
        out.append(
            _with_restored_aliases(
                _slot_check(item, provider=completion.provider,
                            model=completion.model, label=f"第 {index} 题",
                            finish_reason=completion.finish_reason or ""),
                _alias_map(specs[index - 1]),
            )
        )
    return out
