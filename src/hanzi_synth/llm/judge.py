"""L4.5 LLM 裁判：**只在 L4 的程序校验规则全过后**才出场。

契约：
- 裁判是**最后一道**：前六道能过的题里，它只判「是否自足、是否有歧义、解析是否说清」；
- 只判**二元**（过 / 不过）+ 一句理由，不用 1–10 分；
- 裁判看到的只有**读者能看到的东西**（题干 / 解析 / 选项 / 答案），看不到 instruction /
  facts 这些内部字段——给了内部字段，它就会拿那份「出题说明」当尺子去判，
  而真实读者手上并没有那份说明，等于把「自足性」判丢了；
- 思考段在 provider 层已被无条件剥离，这里拿到的 text 一定是剥干净的；
- **默认关闭**：README §8.2 要求先用 300–500 条金标标定、κ ≥ 0.6 才采用，所以它是个开关，
  不是默认行为。

裁判自己跑挂了（网络错 / 输出不是合法 JSON）**不算否决**：那只是「裁判缺席」，
由调用方按缺席处理——不能因为裁判崩了就把一道好题判死。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..tasks.base import TaskSpec
from ..verify.checks import JUDGE_CRITERIA
from .provider import ChatClient
from .verbalizer import extract_json

def _criteria_block() -> str:
    """输入：无。输出：五条判据的编号清单。
    预期：与措辞提示词共用 `verify.checks.JUDGE_CRITERIA`——两边各自写一份就会漂移，
          变成「裁判按 A 判、回答的模型按 B 写」这种没人能发现的错位。
    """
    return "\n".join(f"{i}. {text}" for i, (_name, text) in enumerate(JUDGE_CRITERIA, start=1))


# **中文**：合成的是中文数据，判据与理由一律中文——英文要多一道翻译，小模型会译错，
# 译错的词还会跟着理由一起回灌给措辞模型。
JUDGE_SYSTEM_PROMPT = f"""你是这道题的一个读者。你手上只有这道题本身。

你会收到一个 JSON 对象：`stem`（题干）、`explanation`（解析）、`options`（候选，可能为空）、
`answer`（定死的正确答案）。判下面五条：

{_criteria_block()}

只回一个 JSON 对象，不要 markdown 围栏，不要前后任何说明：
{{"ok": true 或 false, "reason": "一句很短的否决理由；ok 为 true 时留空"}}
"""


@dataclass(frozen=True)
class Judgement:
    """输入：一次裁判调用的结果。输出：不可变结果对象。
    预期：`error` 非空 = 这次裁判**没跑成**（网络错 / 输出不合法），调用方按「裁判缺席」处理：
          重问（`pipeline/run.py` 的 `_judge_with_retries`）；重问完仍缺席就不发布。
          `error` 为空且 `ok=False` 才是真正的「裁判否决」。
    """

    ok: bool
    reason: str = ""
    error: str = ""
    provider: str = ""
    model: str = ""

    @property
    def absent(self) -> bool:
        """输入：无。输出：裁判这次是不是缺席（没跑成）。预期：缺席不是通过——重问，仍缺席则不发布。"""
        return bool(self.error)

    def as_dict(self) -> dict:
        """输入：无。输出：可 JSON 序列化的字典。预期：只留判定本身与**判定它的模型**，不留凭据。
        留 provider / model 是为了事后标定：同一批数据换了裁判模型，κ 是另一个数，
        不记下来就分不清「裁判改了标准」还是「数据真的变好了」。
        """
        return {
            "ok": self.ok, "reason": self.reason, "error": self.error,
            "provider": self.provider, "model": self.model,
        }


def judge_payload(spec: TaskSpec, verbalized: Any) -> dict:
    """输入：TaskSpec + L3 产出。输出：给裁判的提示内容。
    预期：只放**读者能看到的东西**——题干、解析、选项、答案。绝不放 instruction /
          facts / proof。
    """
    return {
        "stem": verbalized.stem,
        "explanation": verbalized.explanation,
        "options": list(spec.options),
        "answer": spec.answer,
    }


def build_judge_messages(spec: TaskSpec, verbalized: Any) -> list[dict]:
    """输入：TaskSpec + L3 产出。输出：OpenAI 格式 messages。
    预期：与提示内容同源——裁判看到的内容与真实读者一致，不多一个内部字段。
    """
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(judge_payload(spec, verbalized), ensure_ascii=False)},
    ]


def judge_record(
    client: ChatClient, spec: TaskSpec, verbalized: Any, *, temperature: float = 0.0
) -> Judgement:
    """输入：客户端 + TaskSpec + L3 产出 + 温度。输出：Judgement。
    预期：温度默认 0——裁判要的是稳定判定，不是发挥；
          请求失败 / JSON 不合法 / `ok` 不是布尔，一律记 `error`（缺席），不判否决。
    """
    completion = client.complete(build_judge_messages(spec, verbalized), temperature=temperature)
    if not completion.ok:
        return Judgement(
            ok=False, error=completion.error or "裁判请求失败",
            provider=completion.provider, model=completion.model,
        )
    data = extract_json(completion.text)
    if not isinstance(data, dict) or not isinstance(data.get("ok"), bool):
        return Judgement(
            ok=False, error="裁判输出不是合法 JSON（缺少布尔字段 ok）",
            provider=completion.provider, model=completion.model,
        )
    return Judgement(
        ok=bool(data["ok"]),
        reason=str(data.get("reason") or ""),
        provider=completion.provider,
        model=completion.model,
    )
