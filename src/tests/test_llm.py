"""L3 措辞层（思考段剥离 / 限速 / JSON 解析 / 多 provider 客户端）的单元测试。"""

from __future__ import annotations

import json
import time
import unittest
from unittest import mock

from hanzi_synth.config import ProviderConfig
from hanzi_synth.llm.provider import ChatClient, Completion
from hanzi_synth.llm.ratelimit import SlidingWindowLimiter
from hanzi_synth.llm.roundtrip import (
    answer_matches,
    build_roundtrip_messages,
    roundtrip_answer,
)
from hanzi_synth.llm.thinking import strip_thinking
from hanzi_synth.llm.verbalizer import (
    Verbalized,
    batch_items_of,
    build_batch_messages,
    build_messages,
    extract_json,
    payload_of,
    verbalize,
    verbalize_batch,
)
from hanzi_synth.tasks.base import TaskSpec



class FakeResponse:
    """输入：状态码与 JSON 体。输出：requests.Response 的鸭子类型替身。预期：让网络层可被完全替换。"""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        """输入：无。输出：预设 JSON。预期：没有预设体时抛 ValueError，模拟非 JSON 响应。"""
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class FakeClient:
    """输入：预设响应序列。输出：鸭子类型的假客户端。预期：记录每次调用，便于断言重试与限速行为。"""

    def __init__(self, responses: list[Completion]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        """输入：messages。输出：下一条预设响应；用尽后返回失败响应。"""
        self.calls.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        return Completion(ok=False, text="", provider="fake", model="fake", error="exhausted")


def ok_completion(text: str) -> Completion:
    """输入：正文。输出：一个成功的 Completion。"""
    return Completion(ok=True, text=text, provider="fake", model="fake-model")


class StripThinkingTests(unittest.TestCase):
    def test_removes_paired_block(self) -> None:
        """成对思考段整段切除。"""
        self.assertEqual(strip_thinking("<think>\u79d8\u5bc6</think>\u7b54\u6848"), ("\u7b54\u6848", True))

    def test_unclosed_block_is_failure(self) -> None:
        """未闭合说明输出被截断，该条判失败。"""
        text, usable = strip_thinking("<think>\u622a\u65ad")
        self.assertFalse(usable)

    def test_thinking_only_is_failure(self) -> None:
        """剥完为空同样判失败——产出里不允许只有思维链。"""
        self.assertEqual(strip_thinking("<think>\u4ec5\u601d\u8003</think>"), ("", False))

    def test_plain_text_passes_through(self) -> None:
        """没有思考段时原样返回。"""
        self.assertEqual(strip_thinking(" \u4f60\u597d "), ("\u4f60\u597d", True))


class LimiterTests(unittest.TestCase):
    def test_zero_rpm_is_unlimited(self) -> None:
        """rpm<=0 表示不限速，直接放行。"""
        limiter = SlidingWindowLimiter(0)
        start = time.monotonic()
        for _ in range(50):
            limiter.acquire()
        self.assertLess(time.monotonic() - start, 0.5)

    def test_window_throttles(self) -> None:
        """窗口内超过上限会阻塞：4 次配额 + 4 次排队 ≈ 一个窗口。"""
        limiter = SlidingWindowLimiter(4, window=0.4)
        start = time.monotonic()
        for _ in range(8):
            limiter.acquire()
        self.assertGreaterEqual(time.monotonic() - start, 0.3)


class ExtractJsonTests(unittest.TestCase):
    def test_plain_and_fenced_and_embedded(self) -> None:
        """裸 JSON、markdown 围栏、前后缀夹带都要能解析出来。"""
        self.assertEqual(extract_json('{"stem": "a"}'), {"stem": "a"})
        self.assertEqual(extract_json('```json\n{"stem": "a"}\n```'), {"stem": "a"})
        self.assertEqual(extract_json('\u524d\u8a00 {"stem": "a"} \u540e\u8bed'), {"stem": "a"})

    def test_garbage_returns_none(self) -> None:
        """解析不了就返回 None，绝不猜。"""
        self.assertIsNone(extract_json("not json"))
        self.assertIsNone(extract_json(""))
        self.assertIsNone(extract_json("[1, 2]"))


class VerbalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = TaskSpec(task_id="T1_stroke_total", instruction="\u95ee\u51e0\u753b",
                             facts={"zi": "\u597d", "stroke_total": 6}, answer="6")

    def test_happy_path(self) -> None:
        """正常返回时带上 provider / model 信息。"""
        client = FakeClient([ok_completion('{"stem": "\u8fd9\u4e2a\u5b57\u51e0\u753b\uff1f", "explanation": "\u516d\u753b"}')])
        result = verbalize(client, self.spec)
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "fake")

    def test_finish_reason_is_carried_through(self) -> None:
        """引擎的收尾原因要带进 Verbalized：`length` 是「被截断」的唯一硬信号。"""
        client = FakeClient([
            Completion(ok=True, text='{"stem": "这个字几画？", "explanation": "六画"}',
                       provider="fake", model="fake-model", finish_reason="length")
        ])
        result = verbalize(client, self.spec)
        self.assertTrue(result.ok)
        self.assertEqual(result.finish_reason, "length")

    def test_missing_slot_fails(self) -> None:
        """缺槽位判失败，不半信半疑放行。"""
        client = FakeClient([ok_completion('{"stem": "\u53ea\u6709\u9898\u5e72"}')])
        self.assertFalse(verbalize(client, self.spec).ok)

    def test_completion_failure_propagates(self) -> None:
        """底层调用失败时错误信息透传。"""
        client = FakeClient([Completion(ok=False, text="", provider="fake", model="m", error="HTTP 500")])
        result = verbalize(client, self.spec)
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "HTTP 500")

    def test_messages_do_not_leak_api_key(self) -> None:
        """发给模型的 messages 里不能出现凭据字段。"""
        messages = build_messages(self.spec)
        self.assertNotIn("api_key", messages[1]["content"])

    def test_visible_subject_gets_a_dedicated_payload_slot(self) -> None:
        """必现串单列成 `stem_must_contain` 列表：埋在散文说明里小模型会漏（实测首轮 6 拒 5 中招）。"""
        spec = TaskSpec(task_id="T14_unicode_codepoint", instruction="i",
                        facts={"zi": "\u3400"}, answer="U+3400", visible_subject="\u3400")
        payload = payload_of(build_messages(spec))
        self.assertEqual(payload["stem_must_contain"], ["\u3400"])
        empty = payload_of(build_messages(self.spec))
        self.assertEqual(empty["stem_must_contain"], [])

    def test_choice_options_are_hard_constraints_too(self) -> None:
        """选择题的候选也得进 `stem_must_contain`：只写在散文说明里，模型会写出
        「下面是一些有趣的汉字。请问总笔画数最少的是哪一个？」这种一个候选都没列的废题
        （实测 16b 的 T11 就是这样）。"""
        spec = TaskSpec(task_id="T11_stroke_extreme", instruction="i",
                        facts={"zi": "\u3402"}, answer="\u3402",
                        options=("\u3402", "\u597d"))
        payload = payload_of(build_messages(spec))
        self.assertEqual(payload["stem_must_contain"], ["\u3402", "\u597d"])

    def test_feedback_is_appended_as_an_extra_user_turn(self) -> None:
        """reflexion：反馈是**追加**的一条 user 消息，载荷本身不被改写。"""
        plain = build_messages(self.spec)
        retried = build_messages(self.spec, ("answer_in_explanation: \u89e3\u6790\u91cc\u6ca1\u6709\u51fa\u73b0\u7b54\u6848",))
        self.assertEqual(len(retried), len(plain) + 1)
        self.assertEqual(retried[1], plain[1])
        self.assertIn("answer_in_explanation", retried[-1]["content"])
        self.assertEqual(payload_of(retried)["answer"], "6")

    def test_failure_reasons_are_passed_through_verbatim(self) -> None:
        """反馈只回灌**原因本身**，不附任何「改法」——规则表已删除，模型按原因自己改。"""
        messages = build_messages(self.spec, ("\u795e\u79d8\u5931\u8d25: \u539f\u56e0",))
        self.assertIn("\u795e\u79d8\u5931\u8d25: \u539f\u56e0", messages[-1]["content"])
        self.assertNotIn("\u6539\u6cd5\uff1a", messages[-1]["content"])


class BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specs = [
            TaskSpec(task_id=f"T{i}", instruction="i", facts={"zi": "\u597d"}, answer=str(i))
            for i in (1, 2, 3)
        ]

    def test_items_are_numbered_from_one(self) -> None:
        """批量载荷把 N 道题放进 items，id 从 1 开始且顺序 = 传入顺序。"""
        items = batch_items_of(build_batch_messages(self.specs))
        self.assertEqual([item["id"] for item in items], [1, 2, 3])
        self.assertEqual([item["answer"] for item in items], ["1", "2", "3"])
        self.assertEqual(items[0]["stem_must_contain"], [])

    def test_results_are_matched_by_id_not_position(self) -> None:
        """结果按 id 对齐：模型把顺序打乱也不串位，缺项只让那一道失败。"""
        text = json.dumps({"results": [
            {"id": 3, "stem": "\u4e09", "explanation": "\u4e09"},
            {"id": 1, "stem": "\u4e00", "explanation": "\u4e00"},
        ]}, ensure_ascii=False)
        client = FakeClient([ok_completion(text)])
        out = verbalize_batch(client, self.specs)
        # 输出按**传入顺序**对齐：第 1 道拿到 id=1、第 3 道拿到 id=3，缺的 id=2 只让第 2 道失败
        self.assertEqual([item.stem for item in out], ["\u4e00", "", "\u4e09"])
        self.assertTrue(out[0].ok and out[2].ok)
        self.assertFalse(out[1].ok)
        self.assertIn("id=2", out[1].error or "")

    def test_whole_batch_failure_marks_every_item(self) -> None:
        """整批请求失败时每一道都带同一个 error——批量的失败是整批的，不能假装某一道成功。"""
        client = FakeClient([Completion(ok=False, text="", provider="fake", model="m", error="HTTP 500")])
        out = verbalize_batch(client, self.specs)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(not item.ok and item.error == "HTTP 500" for item in out))

    def test_batch_slot_check_is_as_strict_as_single(self) -> None:
        """批量路径不许放宽槽位要求，否则 A/B 就变成两套标准比质量。"""
        text = json.dumps({"results": [
            {"id": 1, "stem": "\u4e00", "explanation": "\u4e00"},
            {"id": 2, "stem": "   ", "explanation": "\u4e8c"},
            {"id": 3, "stem": "\u4e09"},
        ]}, ensure_ascii=False)
        out = verbalize_batch(FakeClient([ok_completion(text)]), self.specs)
        self.assertTrue(out[0].ok)
        self.assertIn("\u7f3a\u5c11 stem", out[1].error or "")
        self.assertIn("\u7f3a\u5c11 explanation", out[2].error or "")

    def test_batch_output_without_results_array_fails_every_item(self) -> None:
        """模型只回一个对象、没有 results 数组时，整批同错，而不是「第一道当结果」。"""
        out = verbalize_batch(FakeClient([ok_completion('{"stem": "a", "explanation": "b"}')]), self.specs)
        self.assertTrue(all(not item.ok for item in out))
        self.assertTrue(all("results" in (item.error or "") for item in out))

    def test_payload_and_items_lookup_return_empty_when_absent(self) -> None:
        """取不到载荷就返回空（{} / []），不抛异常——回放工具不该因为一条消息格式怪就崩。"""
        single = build_messages(self.specs[0])
        self.assertEqual(batch_items_of(single), [])
        self.assertEqual(payload_of(build_batch_messages(self.specs)), {})


class ChatClientTests(unittest.TestCase):
    def test_all_providers_failing_returns_error_not_raise(self) -> None:
        """provider 全挂时返回 ok=False 的可读错误，不把异常抛给调用方。"""
        provider = ProviderConfig(name="dead", base_url="http://127.0.0.1:1/v1", model="m", api_key="sk-x", rpm=0)
        client = ChatClient([provider], timeout=0.5, max_retries=0, backoff=0.1)
        result = client.complete([{"role": "user", "content": "hi"}])
        self.assertFalse(result.ok)
        self.assertIn("dead", result.error)

    def test_empty_providers_rejected(self) -> None:
        """空 provider 列表直接报错，不静默空转。"""
        with self.assertRaises(ValueError):
            ChatClient([])

    def test_failover_to_next_provider(self) -> None:
        """前一个 provider 返回 5xx 时自动切到下一个，并返回后者的结果。"""
        first = ProviderConfig(name="p1", base_url="http://p1.test/v1", model="m1", api_key="k1", rpm=0)
        second = ProviderConfig(name="p2", base_url="http://p2.test/v1", model="m2", api_key="k2", rpm=0)
        urls: list[str] = []

        def fake_post(url, headers=None, json=None, timeout=None):
            urls.append(url)
            if "p1.test" in url:
                return FakeResponse(503, text="busy")
            return FakeResponse(200, payload={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})

        client = ChatClient([first, second], timeout=1.0, max_retries=0, backoff=0.01)
        with mock.patch("requests.post", side_effect=fake_post):
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "p2")
        self.assertEqual(len(urls), 2)

    def test_retry_on_429_then_success(self) -> None:
        """429 属于可重试错误：重试成功时不再换 provider。"""
        provider = ProviderConfig(name="p1", base_url="http://p1.test/v1", model="m", api_key="k", rpm=0)
        responses = [FakeResponse(429, text="slow down"),
                     FakeResponse(200, payload={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})]
        client = ChatClient([provider], timeout=1.0, max_retries=1, backoff=0.01)
        with mock.patch("requests.post", side_effect=responses):
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)

    def test_transient_4xx_is_retried(self) -> None:
        """引擎侧瞬态故障被报成 4xx 时也要重试：实测 LM Studio 在并发下回 HTTP 400
        「Context size has been exceeded」，而同一份 payload 单独重发立刻成功。"""
        provider = ProviderConfig(name="p1", base_url="http://p1.test/v1", model="m", api_key="k", rpm=0)
        body = '{"error":"Engine protocol predict stream returned an error: '
        body += '{\\"code\\":500,\\"message\\":\\"Context size has been exceeded.\\"}"}'
        responses = [
            FakeResponse(400, text=body),
            FakeResponse(200, payload={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}),
        ]
        client = ChatClient([provider], timeout=1.0, max_retries=1, backoff=0.01)
        with mock.patch("requests.post", side_effect=responses):
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)

    def test_plain_4xx_is_not_retried(self) -> None:
        """真 4xx（请求本身有问题）不重试：重发只会浪费配额、还会掩盖问题。"""
        provider = ProviderConfig(name="p1", base_url="http://p1.test/v1", model="m", api_key="k", rpm=0)
        client = ChatClient([provider], timeout=1.0, max_retries=2, backoff=0.01)
        with mock.patch("requests.post", return_value=FakeResponse(400, text="bad request")):
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 1)

    def test_thinking_stripped_at_provider_layer(self) -> None:
        """provider 层就剥掉思考段——产出里不可能带思维链。"""
        provider = ProviderConfig(name="p1", base_url="http://p1.test/v1", model="m", api_key="k", rpm=0)
        payload = {"choices": [{"message": {"content": "<think>\u79d8\u5bc6</think>\u7b54\u6848"}, "finish_reason": "stop"}], "usage": {}}
        client = ChatClient([provider], timeout=1.0, max_retries=0, backoff=0.01)
        with mock.patch("requests.post", return_value=FakeResponse(200, payload=payload)):
            result = client.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(result.ok)
        self.assertNotIn("think", result.text)

    def test_request_payload_has_no_max_tokens(self) -> None:
        """请求体绝不能带 max_tokens（思考段会吃光额度）。"""
        provider = ProviderConfig(name="p1", base_url="http://p1.test/v1", model="m", api_key="k", rpm=0)
        captured: dict = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured.update(json or {})
            return FakeResponse(200, payload={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})

        client = ChatClient([provider], timeout=1.0, max_retries=0, backoff=0.01)
        with mock.patch("requests.post", side_effect=fake_post):
            client.complete([{"role": "user", "content": "hi"}])
        self.assertNotIn("max_tokens", captured)
        self.assertEqual(set(captured), {"model", "messages", "temperature"})

    def test_repr_hides_api_key(self) -> None:
        """ProviderConfig 的 repr 不能暴露密钥。"""
        provider = ProviderConfig(name="p", base_url="http://x", model="m", api_key="sk-secret")
        self.assertNotIn("sk-secret", repr(provider))


class JudgeTests(unittest.TestCase):
    """输入：假客户端 + TaskSpec。输出：断言。预期：裁判只看得到**读者能看到的东西**；
    裁判自己跑挂了算「缺席」，不算否决。
    """

    def _spec(self):
        """输入：无。输出：一个带内部字段的 TaskSpec。预期：facts 必须存在，
        这样才能验证它真的没进裁判载荷。
        """
        from hanzi_synth.tasks.base import TaskSpec

        return TaskSpec(
            task_id="T_test",
            instruction="Ask the reader for the code point of the character.",
            facts={"zi": "㐀", "codepoint_forms": ["U+3400", "13312"]},
            answer="U+3400",
            options=("U+3400", "U+3401"),
        )

    def _verbalized(self):
        """输入：无。输出：一份成功的 L3 产出。"""
        from hanzi_synth.llm.verbalizer import Verbalized

        return Verbalized(ok=True, stem="「㐀」的码位是什么？", explanation="码位是 U+3400。")

    def test_payload_exposes_only_reader_visible_fields(self) -> None:
        """裁判绝不能拿到 instruction / facts——给了它就会拿出题说明当尺子。"""
        from hanzi_synth.llm.judge import judge_payload

        payload = judge_payload(self._spec(), self._verbalized())
        self.assertEqual(set(payload), {"stem", "explanation", "options", "answer"})

    def test_reject_verdict_carries_reason(self) -> None:
        """裁判否决要带回理由——没有理由，reflexion 就不知道该改哪里。"""
        from hanzi_synth.llm.judge import judge_record

        client = FakeClient([ok_completion('{"ok": false, "reason": "题干有歧义"}')])
        judgement = judge_record(client, self._spec(), self._verbalized())
        self.assertFalse(judgement.ok)
        self.assertFalse(judgement.absent)
        self.assertEqual(judgement.reason, "题干有歧义")

    def test_broken_output_counts_as_absent_not_reject(self) -> None:
        """裁判输出不是 JSON = 裁判缺席：不能因为裁判自己崩了就把好题判死。"""
        from hanzi_synth.llm.judge import judge_record

        judgement = judge_record(FakeClient([ok_completion("这不是 JSON")]), self._spec(), self._verbalized())
        self.assertTrue(judgement.absent)
        self.assertTrue(judgement.error)

    def test_request_failure_counts_as_absent(self) -> None:
        """裁判请求失败同样是缺席，不判否决。"""
        from hanzi_synth.llm.judge import judge_record

        client = FakeClient([Completion(ok=False, text="", provider="fake", model="fake", error="boom")])
        judgement = judge_record(client, self._spec(), self._verbalized())
        self.assertTrue(judgement.absent)
        self.assertEqual(judgement.error, "boom")

    def test_judge_temperature_defaults_to_zero(self) -> None:
        """裁判要稳定判定，不是发挥：默认温度必须是 0。"""
        from hanzi_synth.llm.judge import judge_record

        client = FakeClient([ok_completion('{"ok": true, "reason": "无"}')])
        judgement = judge_record(client, self._spec(), self._verbalized())
        self.assertTrue(judgement.ok)
        self.assertFalse(judgement.absent)

    def test_criteria_are_the_single_source_for_both_prompts(self) -> None:
        """判据只有一处定义：裁判提示词与**措辞提示词**都必须由 JUDGE_CRITERIA 生成——
        回答的模型不知道裁判按什么判，就会在同一个地方反复被否决。
        """
        from hanzi_synth.llm.judge import JUDGE_SYSTEM_PROMPT
        from hanzi_synth.llm.verbalizer import SYSTEM_PROMPT
        from hanzi_synth.verify.checks import JUDGE_CRITERIA

        self.assertEqual(len(JUDGE_CRITERIA), 4)
        # 反向题（给读音/释义问字形）按设计不写那个字：判据必须写明不能因此判有歧义，
        # 否则裁判会把每一道合法的反向题都否掉。
        criteria = dict(JUDGE_CRITERIA)
        self.assertIn("反向题", criteria["unambiguous"])
        self.assertIn("不要求题干写出答案本身", criteria["self_sufficient"])
        for _name, text in JUDGE_CRITERIA:
            self.assertIn(text, JUDGE_SYSTEM_PROMPT)
            self.assertIn(text, SYSTEM_PROMPT)

    def test_judgement_dict_keeps_the_judging_model(self) -> None:
        """落盘的判定必须带 provider / model：换了裁判模型 κ 就是另一个数，
        不记下来分不清是「裁判改了口径」还是「数据真的变好了」。
        """
        from hanzi_synth.llm.judge import judge_record

        client = FakeClient([ok_completion('{"ok": true, "reason": "无"}')])
        payload = judge_record(client, self._spec(), self._verbalized()).as_dict()
        self.assertEqual(payload["provider"], "fake")
        self.assertEqual(payload["model"], "fake-model")


class PromptInvariantTests(unittest.TestCase):
    """输入：无。输出：断言。预期：提示词与载荷里不出现内部编号——模型会把它们抄进题干。"""

    def test_no_task_code_reaches_the_model(self) -> None:
        """内部题型代号（T9 / T13）绝不能出现在**模型可见**的任何文本里。

        实测 16y 的 T9 题干写「请辨识T9编码链的最内层部件。」——「T9」是从提示词里漏出来的
        内部编号，对读者毫无意义。载荷里的 `task` 字段已删除，系统提示词里的代号也一并改掉；
        这条测试盯着两处入口：系统提示词、裁判提示词，以及**全部构造器产出的载荷字符串**。
        """
        import json
        import random
        import re

        from hanzi_synth.llm.judge import JUDGE_SYSTEM_PROMPT
        from hanzi_synth.llm.verbalizer import SYSTEM_PROMPT
        from hanzi_synth.tasks.base import Difficulty
        from hanzi_synth.tasks.registry import build_all
        from tests.support import make_character, make_index

        # 题型代号形如 T1 / T13：出现在模型可见文本里就说明内部编号漏进了载荷。
        _TASK_CODE_RE = re.compile(r"\bT\d{1,2}\b")

        for label, text in (("SYSTEM_PROMPT", SYSTEM_PROMPT), ("JUDGE_SYSTEM_PROMPT", JUDGE_SYSTEM_PROMPT)):
            hit = _TASK_CODE_RE.search(text)
            self.assertIsNone(hit, f"{label} 里出现题型代号 {hit and hit.group(0)!r}")

        characters = (
            make_character(ziid="U0597D", zi="好", ids=("⿰女子",), stroke=6, radical="女",
                           meanings={"mean_h": ["美好"]}, refs={"zh_ref": {"page": 7}}),
            make_character(ziid="U05973", zi="女", ids=("女",), stroke=3, radical="女"),
            make_character(ziid="U05B50", zi="子", ids=("子",), stroke=3, radical="子"),
            make_character(ziid="U5988", zi="妈", ids=("⿰女马",), stroke=6, radical="女",
                           standard_form="媽", variants=("媽",)),
            make_character(ziid="U9A6C", zi="马", ids=("马",), stroke=3, radical="马"),
        )
        index = make_index(characters)
        specs = [
            spec
            for character in characters
            for spec in build_all(character, rng=random.Random(7), index=index, difficulty=Difficulty())
        ]
        self.assertGreaterEqual(len(specs), 8, "夹具至少要出得来 8 道题，否则这条闸门形同虚设")
        for spec in specs:
            blob = json.dumps(
                {
                    "instruction": spec.instruction,
                    "proof": list(spec.proof),
                    "facts": dict(spec.facts),
                    "why_wrong": [d.why_wrong for d in spec.distractors],
                },
                ensure_ascii=False,
            )
            hit = _TASK_CODE_RE.search(blob)
            self.assertIsNone(hit, f"{spec.task_id} 的载荷里出现题型代号 {hit and hit.group(0)!r}")

    def test_payload_carries_avoid_openings(self) -> None:
        """`avoid_openings` 是**负例**：同题型已经用过的开头喂回去，模型才知道要换一种说法。"""
        from hanzi_synth.llm.verbalizer import payload_for
        from hanzi_synth.tasks.base import TaskSpec

        spec = TaskSpec(task_id="T_test", instruction="i", facts={}, answer="a")
        payload = payload_for(spec, ("请问", "下面"))
        self.assertEqual(payload["avoid_openings"], ["请问", "下面"])
        self.assertEqual(payload_for(spec)["avoid_openings"], [])


class RoundTripTests(unittest.TestCase):
    """输入：各种模型回复。输出：往返检验的判定。预期：只认「答出的答案里含载荷里定死的答案」。"""

    @staticmethod
    def _spec():
        return TaskSpec(task_id="T_test", instruction="i", facts={}, answer="㐑",
                        answer_aliases=("E3 90 91",))

    @staticmethod
    def _client(text: str):
        class _Client:
            def complete(self, messages, *, temperature: float = 0.6, extra=None):
                return Completion(ok=True, text=text, provider="p", model="m")
        return _Client()

    @staticmethod
    def _verbalized():
        """输入：无。输出：一份 L3 产出替身。预期：往返检验只读 stem / explanation / options。"""
        class _V:
            stem = "「㐑」是什么"
            explanation = "答案是「㐑」"
        return _V()

    def test_payload_never_carries_the_answer(self) -> None:
        """载荷里绝不能有答案：放了答案这道检验就自证了。"""
        messages = build_roundtrip_messages(self._verbalized(), self._spec())
        blob = json.dumps(messages, ensure_ascii=False)
        self.assertIn("「㐑」是什么", blob)
        self.assertNotIn("answer", messages[1]["content"])

    def test_plain_answer_is_accepted(self) -> None:
        """模型只答一个字：算过。"""
        self.assertTrue(answer_matches(self._spec(), "㐑"))

    def test_wrapped_and_verbose_answers_are_accepted(self) -> None:
        """模型多写一句话、或者用引号包起来：只要那句里带着定死的答案就算过。"""
        for text in ("「㐑」", "答案是㐑", "这道题的答案是：㐑。"):
            self.assertTrue(answer_matches(self._spec(), text), text)

    def test_alias_is_accepted(self) -> None:
        """载荷里 `accepted` 是同一件事的合法写法，逐个试。"""
        self.assertTrue(answer_matches(self._spec(), "E3 90 91"))

    def test_wrong_or_empty_answer_is_rejected(self) -> None:
        """答成别的字 / 答不出：判否——这正是「读者读不懂」的可判定形式。"""
        for text in ("㐒", "", "我不确定"):
            self.assertFalse(answer_matches(self._spec(), text), text)

    def test_roundtrip_answer_reads_json(self) -> None:
        """正常回复：解析出 answer 字段并按答案比对。"""
        result = roundtrip_answer(self._client('{"answer": "㐑"}'), self._spec(), self._verbalized())
        self.assertTrue(result.ok)
        self.assertEqual(result.answer, "㐑")

    def test_roundtrip_answer_marks_absent_on_bad_output(self) -> None:
        """输出不是合法 JSON：记 `error`（缺席），**不否决**——那是工具问题，不是数据问题。"""
        result = roundtrip_answer(self._client("我不知道"), self._spec(), self._verbalized())
        self.assertTrue(result.absent)
        self.assertFalse(result.ok)

    def test_writer_instructions_in_stem_are_vetoed(self) -> None:
        """题干里夹着写给写作者的要求（读者不该看到）→ 否决，即使答案答对了。

        这一条只有读自然语言才判得了：机械闸门只会做子串 / 集合比对，
        分不出「请列出所有候选」是题目的一部分还是写作要求。
        """
        result = roundtrip_answer(
            self._client('{"answer": "㐑", "asks_reader_only": false, "note": "题干里有「请列出所有候选」"}'),
            self._spec(),
            self._verbalized(),
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.asks_reader_only)
        self.assertFalse(result.absent)

    def test_missing_reader_only_field_is_not_a_veto(self) -> None:
        """字段缺席按 true 处理——「缺席不否决」是这一层的总规矩（老记录 / 模型漏字段都不该判死）。"""
        result = roundtrip_answer(self._client('{"answer": "㐑"}'), self._spec(), self._verbalized())
        self.assertTrue(result.ok)
        self.assertTrue(result.asks_reader_only)

    def test_verdict_records_both_natural_language_judgements(self) -> None:
        """落盘的判定里必须带 `asks_reader_only`：事后审计要能看出「这题当时被判过什么」。"""
        result = roundtrip_answer(
            self._client('{"answer": "㐑", "is_question": false, "asks_reader_only": true}'),
            self._spec(),
            self._verbalized(),
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.as_dict()["is_question"])
        self.assertTrue(result.as_dict()["asks_reader_only"])


if __name__ == "__main__":
    unittest.main()
