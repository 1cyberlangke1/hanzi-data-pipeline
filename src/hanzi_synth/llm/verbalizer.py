"""L3 自然语言化：LLM 唯一出场的地方。

契约（synthesis/README.md §7）：
- 输入是**已确定的事实 JSON**，LLM 只填 `stem` / `explanation` 两个槽位；
- 输出必须是**单个 JSON 对象**，不得带 markdown 围栏或解释性前后缀；
- 数值 / 字符 / IDS 必须**原样引用** facts，不得换算、不得改写；
- 题干**不得包含答案**（这条由 L4 校验器兜底，不只靠提示词）。

思考段在 provider 层已被无条件剥离，这里拿到的 text 一定是剥干净的。

指令分两层（冲突时以下面为准）：
1. `SYSTEM_PROMPT`：全局规则 + 多样性要求；
2. `instruction`：**英文**的出题说明，只说这个题型要问什么。

**程序侧文本一律英文，产出侧文本一律中文**：`instruction` / `stem_must_contain` / `hidden_subject`
这些字段名与说明都是给模型看的，题干与解析必须由模型用中文自己写。实测把中文 instruction 交给
2B 时，它会一字不改地把它当题干输出（数据集里全是程序模板，不是自然语言），英文说明从机制上
切断了这条路——它想抄也抄不进中文题干。
另外重试时会把上一轮被拒的原因作为 `feedback` 追加一轮用户消息（reflexion），
把「猜哪里错了」变成「按原因改」。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..tasks.base import TaskSpec
from ..verify.checks import JUDGE_CRITERIA, JUDGE_FEEDBACK_KEY
from .provider import ChatClient

def _judge_checklist() -> str:
    """输入：无。输出：第 7 道防线（LLM 裁判）四条判据的清单。
    预期：与裁判自己的 system prompt 共用 `verify.checks.JUDGE_CRITERIA`——回答的模型
          若不知道裁判按什么判，就会在同一个地方反复被否决（实测 T13_ids_reverse 连续两轮
          被判「不自足」，而当时的措辞提示词里根本没有这几条判据）。
    """
    return "\n".join(f"- {text}" for _name, text in JUDGE_CRITERIA)


SYSTEM_PROMPT = f"""你是汉字知识数据的措辞助手。你**不需要知道**任何汉字知识，所有事实都在用户给的 JSON 里。

工作方式：
1. 只能使用 facts 里出现的信息，不得引入任何 facts 之外的专有信息（字、读音、页码、书名、数据来源）。
2. 所有数值、字符、IDS 字符串必须**原样引用**，不得换算、不得四舍五入、不得改写描述符。
   题干里**绝不能出现答案本身**，也不能出现答案的等价写法（选择题的候选是唯一例外——候选本来
   就要列出来）；这条对每一道题都成立，不需要在题干里写任何提醒，写了就是把要求漏进数据。
3. `instruction` 是**英文写的出题说明**，它只说明这道题**问什么**。它是**给你看的**，不是题干：
   你**必须用中文自己写** `stem`，不得照抄它的字面，也不得逐字直译。产出里绝不能出现
   instruction / facts / payload 这类字段名，也不能出现任何指代内部流程与写作要求的说法
   （产出必须自足、上下文无关：读者手里只有这道题本身）。
   特别注意：**不要把自己收到的任何要求写成题干里的一句话**——「请列出所有候选」「请说明理由」
   「不能出现笔画数」「不要照抄释义」这类句子都是写给你看的，读者看到的只能是题目本身。
4. `stem` 与 `explanation` 都必须是**通顺的中文自然语言**：写成完整的句子，不能只把字段值
   堆在一起，也不能用英文写。题干必须是一道**题**——向读者提出明确的作答要求
   （「…是什么」「哪一个…」「写出…」），不是复述要求、也不是陈述句。
5. `explanation` 只能复述 proof 里的每一步，不得新增断言。
6. 载荷里 `stem_must_contain` 列出的**每一个串**都必须原样写进题干（如反向题的读音、部件链题的位置描述、
   选择题的候选）——少了任何一个读者都答不了。选择题必须把候选**逐个列出**，读者要有得选。
   列的时候要**自然地嵌进句子里**（例如「下面几个字 A、B、C 里，哪个…」），不要写成清单标题
   或一条要求（「请列出所有候选」「要求：」都不行）。
7. 载荷里 `hidden_subject` 非空时：题干**绝对不能写出这个字**——它是本题的隐藏主语，
   读者只能靠释义与读音认出它。载荷里 `facts.meaning_source` 是**素材**：只能用自己的话改写，
   连续照抄超过十个汉字就会被判不合格。
8. 正文只写题目本身：每个句子都要主谓完整，不要描述自己正在做什么，也不写与作答无关的过渡句。
   写给写作者的限制（「不得出现数字」「不得照抄释义」）一个字都不许进正文——读者不需要知道这些。

**多样性（这条和上面同样重要）**：同一批数据里会有很多道题，读者要能感到每道题是**单独写的**，
不是同一句话换几个字。不要用固定的开场套话（「请问」「这道题」反复开头），不要用同一个句式模板；
换了题目就换一种说法：可以从释义切入、从读音切入、从字形切入、也可以直接给条件再问。
载荷里给了 `avoid_openings` 时，那些开头**已经用过了**，你这次的题干开头必须与它们不同。

**输出格式（必须遵守，否则这次产出作废）**：只输出一个 JSON 对象，不要 markdown 围栏，
不要任何解释性前后缀，形如 {{"stem": "题干", "explanation": "解析"}}；两个槽位都必须非空。

**读者会按下面四条判这道题能不能用**（同一份判据也给了裁判，你写的时候就要满足）：
{_judge_checklist()}
"""

# 重试反馈的开头。明说这些话不能进产出——反馈本身是中文散文，模型抄进题干就会变成
# 「校验器拒绝了上一版产出…」这种内部流程话术，读者看不懂（由 L4.6 的 `is_question` 与
# 可选裁判兜住，本层不靠规则表）。
FEEDBACK_HEADER = (
    "校验器拒绝了上一版产出，原因如下。这只是**修改提示**："
    "绝不能把这些话、以及「校验器」「上一版」「自检」这些词写进 stem 或 explanation。"
)
FEEDBACK_FOOTER = "请针对每一条原因重写；facts、answer、options 都不变。"

# 批量模式的补充系统提示：载荷从「一道题」变成「一个 items 数组」。
# 只有这一段的差别，每道题的写作要求（instruction / stem_must_contain / 自检清单）完全不变——
# 批量不该成为放松要求的借口，质量对比才有意义。
BATCH_SUFFIX = """

批量模式：这一次的载荷是 `items` 数组，里面有 N 道**互不相干**的题（每道带 `id`）。
你必须为每一道题各写一份 `stem` / `explanation`，按 `id` 原样返回，不许合并、不许漏项、不许改 id。
每道题的写作要求与单题模式完全一致（各自的 instruction、stem_must_contain、facts 都只作用于那一道题）。
输出格式：{"results": [{"id": 1, "stem": "...", "explanation": "..."}, ...]}
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


def payload_for(spec: TaskSpec, avoid: Sequence[str] = ()) -> dict[str, Any]:
    """输入：TaskSpec + 语体。输出：给模型的事实载荷（单题与批量共用同一份字段）。
    预期：instruction / facts / answer / proof / distractors 以 JSON 原样给模型；
          `visible_subject` / `must_contain` 合并成独立的 `stem_must_contain` 列表槽位——
          实测把它们埋在散文说明里，小模型会漏（首轮 6 条被拒里有 5 条是必现串），
          单列成字段后模型才当成硬约束看。
          `explanation_must_contain` 同理：解析里必须原样写出答案字符串，这是最高频的拒收原因
          （实测 T12/T1 反复被解析拦下），埋在散文里模型照样漏。
    """
    return {
        # **不给** `task_id`：实测 2B 会把「T9」原样抄进题干（16y 的 T9 写「请辨识T9编码链
        # 的最内层部件」）——那是内部编号，对读者毫无意义。模型不需要它：题型差异已经由
        # 模型不需要它：题型差异已经由 instruction 与载荷字段说清楚了。
        "instruction": spec.instruction,
        "facts": dict(spec.facts),
        "answer": spec.answer,
        "accepted_answers": list(spec.accepted),
        # 题干里必须**原样出现**的全部串（主语 + 题型声明的其它必现串，如 T13 的读音、
        # T9 的位置描述 + **选择题的候选**）。单列成字段模型才当成硬约束看（埋在散文里会漏）。
        # 候选必须进这个槽位：实测 16b 的 T11 产出只写「下面是一些有趣的汉字。请问总笔画数最少的
        # 是哪一个？」，候选一个都没列，读者根本没得选——`options_ok` 会拒收，可模型看不到这条
        # 硬约束，于是每一轮都在同一个地方被拒。
        "stem_must_contain": [
            item
            for item in (spec.visible_subject, *spec.must_contain, *spec.options)
            if item
        ],
        # 反向题的隐藏主语：题干必须靠释义与读音让读者认出它，绝不能直接写出来
        "hidden_subject": spec.hidden_subject,
        # 问字形结构的题（T13）：解析里只许出现这里列出的位置词，题干里一个都不许出现
        "structure_words": list(spec.structure_words),
        # 「题干问的必须是答案那件事」的硬闸门：题干至少要命中这里的一个说法
        "stem_must_ask": list(spec.stem_asks_any),
        "explanation_must_contain": spec.answer,
        "options": list(spec.options),
        "proof": list(spec.proof),
        "distractors": [{"text": d.text, "why_wrong": d.why_wrong} for d in spec.distractors],
        "allowed_slots": list(spec.slots),
        # 已经用过的题干开头：这是**负例**——限制重复，不是限制表达。
        "avoid_openings": list(avoid),
    }


def build_messages(
    spec: TaskSpec, feedback: Sequence[str] = (), avoid: Sequence[str] = ()
) -> list[dict[str, str]]:
    """输入：TaskSpec + 语体 + 上一轮被拒的原因（可空）。输出：OpenAI 格式 messages（单题）。
    预期：载荷走 `payload_for`，与批量模式同一份字段；
          有 feedback 时追加一轮用户消息把失败原因回灌（reflexion），
          并声明这些话绝不能进产出——「校验器 / 上一版 / 自检」同时在 L4 禁用词表里，
          模型一旦照抄就会被 self_contained 抓住。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload_for(spec, avoid), ensure_ascii=False, indent=2)},
    ]
    if feedback:
        reasons = _feedback_block(feedback)
        messages.append({"role": "user", "content": f"{FEEDBACK_HEADER}\n{reasons}\n{FEEDBACK_FOOTER}"})
    return messages


def build_batch_messages(
    specs: Sequence[TaskSpec], feedback: Mapping[str, Sequence[str]] | None = None
) -> list[dict[str, str]]:
    """输入：多个 TaskSpec + 语体 + 按题分组的失败原因（可空）。输出：OpenAI 格式 messages（批量）。
    预期：所有题的事实放进一个 `items` 数组，`id` 从 1 开始（顺序 = 传入顺序）；
          批量只改请求打包方式，不改每道题的写法。
    """
    items: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        payload = payload_for(spec)
        items.append({"id": index, **payload})
    body: dict[str, Any] = {"items": items, "allowed_slots": ["stem", "explanation"]}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + BATCH_SUFFIX},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, indent=2)},
    ]
    if feedback:
        reasons = _batch_feedback_block(feedback)
        messages.append({"role": "user", "content": f"{FEEDBACK_HEADER}\n{reasons}\n{FEEDBACK_FOOTER}"})
    return messages


def extract_json(text: str) -> dict | None:
    """输入：模型输出文本。输出：解析出的 dict，失败时 None。
    预期：先剥 markdown 围栏再解析；仍失败则退化为「取第一个 { 到最后一个 }」。
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def payload_of(messages: Sequence[Mapping[str, Any]]) -> dict:
    """输入：`build_messages` 产出的 messages。输出：其中的事实载荷（含 facts / answer 的那一条）。
    预期：从后往前找第一条能解析成 JSON 对象的 user 消息——重试时会追加反馈消息，
          所以「载荷永远是 messages[-1]」**不再成立**，回放 / 审计 / 假客户端都要用这个函数取载荷。
          取不到时返回空字典，不抛异常。
    """
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        data = extract_json(str(message.get("content", "")))
        if isinstance(data, dict) and "answer" in data:
            return data
    return {}


def batch_items_of(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    """输入：`build_batch_messages` 产出的 messages。输出：`items` 数组（取不到时返回空列表）。
    预期：与 `payload_of` 同源的取法——从后往前找第一条带 `items` 的 user 消息；
          批量模式没有单题载荷，回放 / 假客户端要用这个函数。
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
    预期：`stem` / `explanation` 缺一不可、必须是非空字符串。单题与批量共用这一份校验——
          批量路径**不允许**悄悄放宽槽位要求，否则 A/B 就变成「两套标准比质量」。
    """
    prefix = f"{label}：" if label else ""
    stem = data.get("stem")
    explanation = data.get("explanation")
    if not isinstance(stem, str) or not stem.strip():
        return Verbalized(ok=False, provider=provider, model=model,
                          error=f"{prefix}缺少 stem 槽位", finish_reason=finish_reason)
    if not isinstance(explanation, str):
        return Verbalized(ok=False, provider=provider, model=model,
                          error=f"{prefix}缺少 explanation 槽位", finish_reason=finish_reason)
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
    预期：JSON 解析失败、缺槽位、槽位非字符串一律判失败，绝不半信半疑地放行；
          feedback 非空时把它作为额外一轮用户消息传给模型（reflexion）。
    """
    completion = client.complete(build_messages(spec, feedback, avoid), temperature=temperature)
    if not completion.ok:
        return Verbalized(ok=False, provider=completion.provider,
                          model=completion.model, error=completion.error)

    data = extract_json(completion.text)
    if data is None:
        return Verbalized(ok=False, provider=completion.provider,
                          model=completion.model, error="模型输出不是合法 JSON")
    return _slot_check(data, provider=completion.provider, model=completion.model,
                        finish_reason=completion.finish_reason or "")


def verbalize_batch(
    client: ChatClient,
    specs: Sequence[TaskSpec],
    *,
    temperature: float = 0.6,
    feedback: Mapping[str, Sequence[str]] | None = None,
) -> list[Verbalized]:
    """输入：客户端 + 多个 TaskSpec + 温度 + 按题分组的失败原因。
    输出：与 `specs` **等长且顺序一致**的 Verbalized 列表。
    预期：整批一次请求；结果按 `id` 对齐，缺项 / 槽位不合格只让**那一道**失败，不串位。
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
            _slot_check(item, provider=completion.provider,
                        model=completion.model, label=f"第 {index} 题",
                        finish_reason=completion.finish_reason or "")
        )
    return out
