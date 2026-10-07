"""L4.6 往返检验（round-trip）：把产出交回模型，让它**只凭题面与解析**说出答案。

为什么要这一层：L4 的程序防线只能判**形状**（答案出现了吗、数字来自事实吗、候选列全了吗），
判不了「这段话读起来通不通」。而「读者读不读得懂」用**枚举措辞形状**去追是追不完的——
规则表会无限增长，永远收不了口。换成一个**闭集检验**：答案由代码在出题阶段就定死，
所以只要让模型照着这段文字答题，**答对了**就说明这段文字自足、可读、没把读者带偏。

为什么这条判据不会变成「模型说了算」：判据比对的是**载荷里定死的答案**，不是模型的偏好；
模型不需要会判好坏，它只需要**照着题面答题**——这对 2B 也是做得到的任务。

除答案之外，这条还让读者判两件**只有读自然语言才能判**的事：题干是不是在向读者提问、
题干里有没有夹着写给写作者的要求（「请列出所有候选」「不要出现数字」这类句子）——
机械闸门只会做子串 / 集合比对，判不了这两件事。

与 L4.5 裁判（`judge.py`）的分工：那条问「你觉得这题好不好」（开放判断，小模型标定不过），
这条问「你读出了什么 / 题干是不是只在对读者说话」。两者都只给**读者能看到的东西**。

裁判缺席（请求失败 / 输出不是合法 JSON）**不算否决**：那是工具问题，不是数据问题。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..tasks.base import TaskSpec
from .provider import ChatClient
from .verbalizer import extract_json

ROUNDTRIP_SYSTEM_PROMPT = """你是一个读者，手里只有下面这道汉字题本身，没有任何别的资料。

请**只根据题干**回答这道题，不要使用外部知识，不要猜测，不要评价这道题好不好。
题干信息不足以定出答案时，就照实说「题干没有说明是哪一个」——那正是这道题不合格的样子。
但**只在真的缺信息时才这么说**：题干若已经列出了几个候选、并问「哪一个…」，你就必须从中挑一个答出来，
不许再说「没有说明是哪一个」。

另外判两件事。

第一件：题干**是不是在向读者提问**。
- 题干提出了明确的作答要求（「…是什么」「哪一个…」「写出…」「…是多少」），算 true；
- 题干只是在陈述，或者把一条**写作要求 / 出题说明**复述了一遍
  （「问「候选字…里哪一个…」」「题干需按给定动作替换…」「要求：…」），算 false。

第二件：题干里有没有夹着**写给写作者的要求**。
- 题干通篇只是在向读者提问、给出作答要求，算 true；
- 题干里夹着「请列出所有候选」「请说明理由」「不要出现数字」「必须原样引用」「请用自己的话解释」
  这类**对写题人的要求**（哪怕它同时也在提问），算 false——读者手里没有这些要求，
  那些句子不是题目的一部分。

只输出一个 JSON 对象，不要 markdown 围栏，不要任何前后缀说明：
{"answer": "你从题干里读出的答案", "is_question": true 或 false, "asks_reader_only": true 或 false, "note": "上面任一项为 false 时，一句话说明题干缺了什么 / 多写了什么"}

答案要写成题目要求的那种形式：问字就写那个字，问读音就写拼音，问码位就写码位，问 IDS 就写 IDS 串。
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
    """
    got = normalize_answer(answer)
    if not got:
        return False
    for expected in (spec.answer, *spec.accepted):
        needle = normalize_answer(str(expected))
        if needle and needle in got:
            return True
    return False


@dataclass(frozen=True)
class RoundTrip:
    """输入：一次往返检验的结果。输出：不可变结果对象。
    预期：`error` 非空 = 这次没跑成（请求失败 / 输出不合法），调用方按**缺席**处理；
          `error` 为空且 `ok=False` 才是真正的「这段文字不合格」——三条判据任一条不过都算：
          答不出答案、不是在提问、题干里夹着写给写作者的要求。
    """

    ok: bool
    answer: str = ""
    is_question: bool = True
    asks_reader_only: bool = True
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
            "ok": self.ok, "answer": self.answer, "is_question": self.is_question,
            "asks_reader_only": self.asks_reader_only,
            "note": self.note, "error": self.error,
            "provider": self.provider, "model": self.model,
        }


def roundtrip_payload(verbalized: Any, spec: TaskSpec) -> dict:
    """输入：L3 产出 + TaskSpec。输出：给检验模型的载荷。

    预期：**只放题干与选项**，绝不放解析。解析里写着答案，放了它读者照抄就算"答对"，
          这道检验立刻退化成恒真——实测：带解析跑 6 条，4 条题干不自足
          （「请问这个汉字对应的 Unicode 编码是什么？」——读者不知道是哪个汉字）全被判过。
          去掉解析后，读者只能靠题干本身的信息作答，「题干自不自足」才真的被判到。
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
    client: ChatClient, spec: TaskSpec, verbalized: Any, *, temperature: float = 0.0
) -> RoundTrip:
    """输入：客户端 + TaskSpec + L3 产出 + 温度。输出：RoundTrip。

    预期：温度默认 0——这条检验要的是稳定复现，不是发挥；
          请求失败 / JSON 不合法 / 缺 answer 字段，一律记 `error`（缺席），不判否决。
    """
    completion = client.complete(build_roundtrip_messages(verbalized, spec), temperature=temperature)
    if not completion.ok:
        return RoundTrip(
            ok=False, error=completion.error or "往返检验请求失败",
            provider=completion.provider, model=completion.model,
        )
    data = extract_json(completion.text)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return RoundTrip(
            ok=False, error="往返检验输出不是合法 JSON（缺少字符串字段 answer）",
            provider=completion.provider, model=completion.model,
        )
    answer = data["answer"]
    # 两个布尔字段缺席时按 true 处理——「缺席不否决」是这一层的总规矩（工具问题不判数据死刑）。
    is_question = data.get("is_question")
    is_question = True if not isinstance(is_question, bool) else is_question
    asks_reader_only = data.get("asks_reader_only")
    asks_reader_only = True if not isinstance(asks_reader_only, bool) else asks_reader_only
    note = str(data.get("note") or "")
    return RoundTrip(
        ok=answer_matches(spec, answer) and is_question and asks_reader_only,
        answer=answer,
        is_question=is_question,
        asks_reader_only=asks_reader_only,
        note=note,
        provider=completion.provider,
        model=completion.model,
    )
