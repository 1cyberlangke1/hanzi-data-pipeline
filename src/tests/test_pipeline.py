"""管线端到端测试（用假客户端，不联网）。"""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from hanzi_synth.facts.index import CharacterIndex
from hanzi_synth.facts.loader import DataError, load_characters
from hanzi_synth.llm.provider import Completion
from hanzi_synth.llm.verbalizer import Verbalized, batch_items_of, extract_json, payload_of
from hanzi_synth.config import ConfigError, ProviderConfig
from hanzi_synth.pipeline.run import (
    LOCAL_WORKER_CAP,
    PipelineBusy,
    PipelineConfig,
    auto_replay,
    _plan,
    _verified_with_semantic_answer,
    is_local_endpoint,
    replay_rejects,
    run_pipeline,
    spec_from_record,
    with_local_worker_cap,
)
from hanzi_synth.tasks.base import TaskSpec
from hanzi_synth.tasks.registry import TASK_STEM_ASKS, build_all
from hanzi_synth.verify.checks import verify


class ScriptedClient:
    """输入：一串预设正文。输出：鸭子类型客户端。预期：每次调用吐一条，用尽后返回失败。"""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        """输入：messages。输出：预设正文包装的成功响应。"""
        self.calls += 1
        if not self._texts:
            return Completion(ok=False, text="", provider="scripted", model="m", error="exhausted")
        return Completion(ok=True, text=self._texts.pop(0), provider="scripted", model="m")


def compliant_body(payload: dict) -> str:
    """输入：L3 的 payload。输出：一份能过 L4 的规范正文。
    预期：题干直接用 `instruction`——它就是这道题要问的那句话，也天然带上主语与「问什么」的说法
          （假客户端要像真模型一样看这份提示内容，否则它造出的「规范」正文其实是残废题）；
          候选只在**说明里没写全**时补一次——说明现在就把候选写在句首，再补一遍就是同一个字
          出现两次，`no_answer_leak` 判的正是「答案出现几次」，补第二遍等于自己造泄题；
          解析里说出答案。
          这几条都是 L4 的硬要求，缺一条整批都会被判 C。
    """
    answer = payload["answer"]
    # 假客户端要像真模型一样**改写**说明，不能原样回显：原样回显正是 `no_instruction_copy`
    # 拦下的那种产出（真机实测 23/1,321 条；离线模板池 0 条）。加一句问法前缀即可，
    # 主语与「问什么」都还在，别的校验照样过。
    stem = f"请问{payload.get('instruction') or '这个字是什么'}？"
    options = [option for option in (payload.get("options") or []) if option]
    if options and not all(option in stem for option in options):
        stem += "，" + "、".join(options) + " 里选哪一个"
    return json.dumps({"stem": stem, "explanation": f"答案是「{answer}」"}, ensure_ascii=False)


class PayloadEchoClient:
    """输入：无。输出：按请求里的 payload 造出规范正文的假客户端。
    预期：解析里一定出现答案，题干里一定出现主语——L4 的硬要求。
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        """输入：messages。输出：规范响应。"""
        self.calls += 1
        payload = payload_of(messages)
        return Completion(ok=True, text=compliant_body(payload), provider="scripted", model="m")


class FlakyClient:
    """输入：先坏几轮。输出：坏轮给不符合要求正文，之后给规范正文的假客户端。
    预期：用来验证「L4 不过 -> 带原因重试」这条链路真的接上了。
    """

    def __init__(self, bad_rounds: int) -> None:
        self.bad_rounds = bad_rounds
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        """输入：messages。输出：前 bad_rounds 次不符合要求，之后规范。"""
        self.calls += 1
        if self.calls <= self.bad_rounds:
            body = json.dumps({"stem": "\u9898\u5e72", "explanation": "\u89e3\u6790"}, ensure_ascii=False)
        else:
            payload = payload_of(messages)
            body = compliant_body(payload)
        return Completion(ok=True, text=body, provider="scripted", model="m")


class RecordingFlakyClient(FlakyClient):
    """输入：先坏几轮。输出：与 FlakyClient 同，但记录每次收到的 messages。
    预期：用来验证 reflexion——第二轮请求里必须带上第一轮被 L4 判死的**具体原因**，
          而不是什么原因都不带地重新掷一次。
    """

    def __init__(self, bad_rounds: int) -> None:
        super().__init__(bad_rounds)
        self.requests: list[list[dict]] = []

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        """输入：messages。输出：同父类结果，并把 messages 存进 self.requests（深拷贝一层）。"""
        self.requests.append([dict(message) for message in messages])
        return super().complete(messages, temperature=temperature, extra=extra)


class BatchEchoClient:
    """输入：让第几次请求的第几道题出错（可空）。输出：按 items 逐题造规范正文的批量假客户端。
    预期：results 与 items 等长、id 对齐；记录每次请求的题数，用来断言「重试只带没过的那几道」。
    """

    def __init__(self, bad: dict[int, int] | None = None) -> None:
        # bad: {第几次请求(从 1 起): 该次让哪一道 id 出错}
        self._bad = dict(bad or {})
        self.calls = 0
        self.item_counts: list[int] = []

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        """输入：messages。输出：批量规范响应。"""
        self.calls += 1
        items = batch_items_of(messages)
        self.item_counts.append(len(items))
        bad_id = self._bad.get(self.calls, 0)
        results = []
        for item in items:
            if item["id"] == bad_id:
                results.append({"id": item["id"], "stem": "\u9898\u5e72", "explanation": "\u89e3\u6790"})
                continue
            body = json.loads(compliant_body(item))
            results.append({"id": item["id"], **body})
        return Completion(ok=True, text=json.dumps({"results": results}, ensure_ascii=False),
                          provider="scripted", model="m")


def write_chars(path: Path) -> None:
    """输入：目标路径。输出：无。预期：写 4 条有 IDS 的记录，其中 2 条带释义。"""
    records = []
    for index, (zi, ids, meaning) in enumerate([
        ("\u597d", "\u2ff0\u5973\u5b50", "\u62fc\u97f3hao3<br>\u7f8e\u597d\u7684\u610f\u601d\u3002"),
        ("\u4e00", "\u2ff0\u4e00\u4e8c", None),
        ("\u4e8c", "\u2ff1\u4e00\u4e00", "\u62fc\u97f3er4<br>\u6570\u5b57\u4e8c\u3002"),
        ("\u4e09", "\u2ff1\u4e00\u4e8c", None),
    ]):
        prop = {"ziid": f"U{index:05X}", "zi": zi, "ids": ids, "stroke_total": 6, "radical": "\u5973", "structure": 1}
        meanings = {"mean_h": [meaning]} if meaning else {}
        records.append({"ziid": prop["ziid"], "property": prop, "meanings": meanings, "variants": []})
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class PipelineOfflineTests(unittest.TestCase):
    def test_offline_writes_specs_without_llm(self) -> None:
        """offline 模式只跑 L0/L2，产出里没有 stem，也不会去加载凭据。"""
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "chars.jsonl"
            out = Path(tmp) / "out.jsonl"
            write_chars(data)
            stats = run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=data, offline=True))
            self.assertEqual(stats["ok"], 4)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("stem" not in row for row in rows))
            self.assertTrue(all(row["answer"] for row in rows))
            self.assertTrue(all("options" in row for row in rows))

    def test_offline_difficulty_is_configurable(self) -> None:
        """难度旋钮经 PipelineConfig 生效：hops=2 时 T9 的 facts 里带 depth=2。"""
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "chars.jsonl"
            out = Path(tmp) / "out.jsonl"
            write_chars(data)
            stats = run_pipeline(
                PipelineConfig(limit=4, out_path=out, data_path=data, offline=True, hops=2, distractor_depth="far")
            )
            self.assertEqual(stats["ok"], 4)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("answer" in row for row in rows))

    def test_offline_records_are_replay_ready(self) -> None:
        """dry-run 记录带上重跑所需的 proof / visible_subject / difficulty，否则重跑无从重建。"""
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "chars.jsonl"
            out = Path(tmp) / "out.jsonl"
            write_chars(data)
            run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=data, offline=True))
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            for row in rows:
                self.assertIn("proof", row)
                self.assertIn("visible_subject", row)
                self.assertIn("must_contain", row)
                # 校验判据（题干必须问什么、解析里只许出现哪些结构词）也要写入文件，
                # 否则重跑路径比实时路径松（replay 会把实时拦下的泄题放出去）
                self.assertIn("stem_asks_any", row)
                self.assertIn("structure_words", row)
                # 请求模型时只发这几个事实键；它不写入文件，重跑就会退回「整份 facts 全发」——
                # 实测把 `source`（康熙字典）又发了出去，重跑那条写成「根据《康熙字典》记载」。
                self.assertIn("payload_facts", row)
                self.assertEqual(row["difficulty"]["kind"], "analytic-prior")

    def test_offline_record_round_trips_gate_criteria(self) -> None:
        """`spec_from_record` 必须把两条校验判据原样读回来（互逆是重跑路径的正确性前提）。"""
        spec = TaskSpec(
            task_id="T13_ids_reverse", instruction="i", facts={}, answer="⿸𠂆⿱𫜹𠃌",
            stem_asks_any=("IDS", "拆解"), structure_words=("左上包围", "上下"),
            payload_facts=("zi", "gloss_items"),
        )
        row = {"task_id": spec.task_id, "answer": spec.answer,
               "stem_asks_any": list(spec.stem_asks_any), "structure_words": list(spec.structure_words),
               "payload_facts": list(spec.payload_facts)}
        restored = spec_from_record(row)
        self.assertEqual(restored.stem_asks_any, spec.stem_asks_any)
        self.assertEqual(restored.structure_words, spec.structure_words)
        self.assertEqual(restored.payload_facts, spec.payload_facts)
        # 缺字段时退回空元组（= 全发，只给手工 spec 用）：旧记录不能崩，
        # 也不能被凭空补上键名。
        self.assertEqual(spec_from_record({"task_id": spec.task_id, "answer": spec.answer}).payload_facts, ())


class PipelineLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def test_round_robin_covers_all_tasks(self) -> None:
        """轮转取样让一次小批量覆盖多个题型，而不是被 T1 刷屏。"""
        client = PayloadEchoClient()
        out = self.dir / "out.jsonl"
        stats = run_pipeline(PipelineConfig(limit=8, out_path=out, data_path=self.data), client=client)
        self.assertGreater(len(stats["by_task"]), 1)
        self.assertEqual(stats["ok"] + stats["rejected"], 8)
        # 假客户端一次就规范：首次措辞通过数应等于成功数
        self.assertEqual(stats["first_pass"], stats["ok"])
        self.assertEqual(stats["requests"], 8)  # 单题模式：一道题一个请求

    def test_concurrent_workers_produce_same_result(self) -> None:
        """并发只影响速度：workers>1 的产出与串行逐条一致（顺序与内容都不变）。"""
        serial_out = self.dir / "serial.jsonl"
        parallel_out = self.dir / "parallel.jsonl"
        run_pipeline(PipelineConfig(limit=8, out_path=serial_out, data_path=self.data), client=PayloadEchoClient())
        stats = run_pipeline(
            PipelineConfig(limit=8, out_path=parallel_out, data_path=self.data, workers=4),
            client=PayloadEchoClient(),
        )
        self.assertEqual(stats["ok"], 8)
        self.assertEqual(
            serial_out.read_text(encoding="utf-8"),
            parallel_out.read_text(encoding="utf-8"),
        )

    def test_batch_mode_keeps_question_content_identical(self) -> None:
        """批量只改请求打包方式，不改题目内容：同数据同 limit 下题干/解析与单题模式逐条一致。
        预期：语体是唯一差别——一次请求只能有一种语体，所以同批共享，这也是批量模式的语义代价。
        """
        single = self.dir / "single.jsonl"
        batched = self.dir / "batched.jsonl"
        run_pipeline(PipelineConfig(limit=8, out_path=single, data_path=self.data), client=PayloadEchoClient())
        stats = run_pipeline(
            PipelineConfig(limit=8, out_path=batched, data_path=self.data, batch=4), client=BatchEchoClient()
        )
        self.assertEqual(stats["batch"], 4)
        self.assertEqual(stats["ok"], 8)
        self.assertEqual(stats["requests"], 2)  # 8 道题 / 每请求 4 道 = 2 个请求

        def content(path: Path) -> list[tuple[str, str, str, str]]:
            """输入：产出文件。输出：与打包方式无关的题目内容四元组列表。"""
            return [
                (row["task_id"], row["ziid"], row["stem"], row["explanation"])
                for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            ]

        self.assertEqual(content(single), content(batched))

    def test_batch_retry_only_resends_failed_items(self) -> None:
        """批量重试只把没过的那几道再发一次：过了的不重复花配额。"""
        client = BatchEchoClient({1: 1})  # 第一次请求里 id=1 那题坏
        out = self.dir / "batch_retry.jsonl"
        stats = run_pipeline(
            PipelineConfig(limit=4, out_path=out, data_path=self.data, batch=4, retries=1), client=client
        )
        self.assertEqual(client.item_counts, [4, 1])
        self.assertEqual(stats["requests"], 2)  # 首轮 1 个请求 + 重试 1 个请求
        self.assertEqual(stats["requests"], client.calls)
        self.assertEqual(stats["ok"], 4)
        self.assertEqual(stats["retried"], 1)
        self.assertEqual(stats["first_pass"], 3)

    def test_l4_failure_is_retried(self) -> None:
        """L4 不过时带原因重试：第一轮坏、第二轮好 -> 记录仍能进主产出，并计入 retried。"""
        out = self.dir / "retry.jsonl"
        stats = run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=1), client=FlakyClient(bad_rounds=1)
        )
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(stats["retried"], 1)
        self.assertEqual(stats["first_pass"], 0)  # 首轮被拒过，不算首轮通过

    def test_retries_zero_means_no_retry(self) -> None:
        """retries=0 时不过就是不过，直接进 rejects（重试次数是可配置的，不是写死的）。"""
        out = self.dir / "noretry.jsonl"
        stats = run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0), client=FlakyClient(bad_rounds=1)
        )
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(stats["retried"], 0)

    def test_retry_feeds_back_previous_failures(self) -> None:
        """reflexion：重试请求比首轮多一条反馈消息，且反馈里带着上一轮 L4 判死的具体原因。"""
        client = RecordingFlakyClient(bad_rounds=1)
        out = self.dir / "feedback.jsonl"
        stats = run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=1), client=client
        )
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(len(client.requests), 2)
        first, second = client.requests
        self.assertEqual(len(second), len(first) + 1)  # 第二轮多一条反馈消息
        self.assertEqual(second[-1]["role"], "user")
        # 坏正文的题干没写出主语 -> L4 一定报 subject_in_stem；这条原因必须出现在反馈里。
        # 回灌的是模型侧说法（`notes`，不是给人看的 `failures`），但必须指名道姓：
        # 反馈里要带着缺的那个串，模型才知道改哪里。
        self.assertIn("必现串", second[-1]["content"])
        self.assertIn("二", second[-1]["content"])
        # 中文失败原因仍然进 rejects / attempt-stats，两条路各留各的
        attempts = [
            json.loads(line)
            for line in out.with_suffix(".attempts.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(any("subject_in_stem" in item for item in attempts[0]["failures"]))

    def test_resume_does_not_silently_produce_nothing(self) -> None:
        """续跑时已被覆盖满的前缀字不占扫描预算：max_scan 只数真正构造过题的字。
        预期：第一次跑出前两个字的 T1 后，第二次即使 max_scan=1 也能产出新记录，而不是静默空跑。
        """
        out = self.dir / "plan.jsonl"
        first = run_pipeline(
            PipelineConfig(
                limit=2, out_path=out, data_path=self.data, offline=True,
                task_ids=("T1_stroke_total",), resume=False,
            )
        )
        self.assertEqual(first["ok"], 2)
        second = run_pipeline(
            PipelineConfig(
                limit=2, out_path=out, data_path=self.data, offline=True,
                task_ids=("T1_stroke_total",), max_scan=1,
            )
        )
        self.assertGreater(second["skipped_covered"], 0)
        self.assertEqual(second["scanned"], 1)  # 真扫了 1 个字，不是把 max_scan 当结果抄回来
        self.assertGreater(second["planned"], 0)
        self.assertGreater(second["ok"], 0)

    def test_rejects_are_written_separately(self) -> None:
        """验证不过的记录进 rejects 文件，不污染主产出。"""
        bad = '{"stem": "\u6839\u636e\u8bc1\u660e\uff0c\u7b54\u6848\u662f 6", "explanation": "\u6839\u636e\u8bc1\u660e"}'
        client = ScriptedClient([bad] * 4)
        out = self.dir / "out.jsonl"
        stats = run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=client)
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 4)
        rejects = out.with_suffix(".rejects.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(rejects), 4)
        self.assertTrue(any("answer_in_explanation" in line for line in rejects))

    def test_provenance_is_recorded(self) -> None:
        """每条产出都要能回查到种子字与构造器版本。"""
        client = PayloadEchoClient()
        out = self.dir / "out.jsonl"
        run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=client)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            self.assertTrue(row["provenance"]["seed_ziid"])
            self.assertTrue(row["provenance"]["constructor_version"])

    def test_live_records_carry_facts_for_reverification(self) -> None:
        """产出必须带 facts：否则无法独立复验（重跑 L4、审计题干有没有写出那个字）。"""
        out = self.dir / "facts.jsonl"
        run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(row["facts"])


class PipelineDurabilityTests(unittest.TestCase):
    """数据不丢：续跑追加、半行修复、覆盖前备份、拒绝写到输入上、独占锁、覆盖率附带。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def test_resume_appends_and_keeps_old_records(self) -> None:
        """续跑：旧记录一字不改，新记录追加在后面，主键不重复。"""
        out = self.dir / "resume.jsonl"
        first = run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        before = out.read_text(encoding="utf-8")
        second = run_pipeline(PipelineConfig(limit=8, out_path=out, data_path=self.data), client=PayloadEchoClient())
        after = out.read_text(encoding="utf-8")
        self.assertEqual(first["resumed"], 0)
        self.assertEqual(second["resumed"], first["ok"])
        self.assertGreater(second["ok"], 0)
        self.assertTrue(after.startswith(before))
        self.assertGreater(len(after), len(before))
        keys = [(row["task_id"], row["ziid"]) for row in map(json.loads, after.splitlines())]
        self.assertEqual(len(keys), len(set(keys)))

    def test_torn_tail_is_separated_before_append(self) -> None:
        """上轮掐断留下的半行：先补换行，残行之后每一行仍是完整 JSON。"""
        out = self.dir / "torn.jsonl"
        torn = '{"task_id": "T1_stroke_total", "ziid": "U0000'
        out.write_text(torn, encoding="utf-8")
        stats = run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], torn)
        self.assertEqual(len(lines), stats["ok"] + 1)
        for line in lines[1:]:
            json.loads(line)

    def test_overwrite_backs_up_instead_of_destroying(self) -> None:
        """--overwrite 语义：旧产出先备份，绝不静默销毁。"""
        out = self.dir / "ow.jsonl"
        run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        before = out.read_text(encoding="utf-8")
        stats = run_pipeline(
            PipelineConfig(limit=4, out_path=out, data_path=self.data, resume=False), client=PayloadEchoClient()
        )
        backups = list(self.dir.glob("ow.jsonl.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), before)
        self.assertEqual(stats["resumed"], 0)
        self.assertTrue(out.read_text(encoding="utf-8").strip())

    def test_refuses_to_write_output_onto_input_data(self) -> None:
        """产出路径 == 输入数据路径时直接拒绝，原始数据一个字节都不动。"""
        before = self.data.read_text(encoding="utf-8")
        with self.assertRaises(DataError):
            run_pipeline(PipelineConfig(limit=1, out_path=self.data, data_path=self.data), client=PayloadEchoClient())
        self.assertEqual(self.data.read_text(encoding="utf-8"), before)

    def test_lock_rejects_live_writer_and_reclaims_dead_one(self) -> None:
        """同一产出文件的独占锁：活进程持有 -> 拒绝；持有者已死 -> 接管并跑完释放。"""
        out = self.dir / "locked.jsonl"
        lock = out.with_name(out.name + ".lock")
        lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
        with self.assertRaises(PipelineBusy):
            run_pipeline(PipelineConfig(limit=1, out_path=out, data_path=self.data), client=PayloadEchoClient())
        lock.write_text("999999\n", encoding="utf-8")
        stats = run_pipeline(PipelineConfig(limit=1, out_path=out, data_path=self.data), client=PayloadEchoClient())
        self.assertEqual(stats["ok"], 1)
        self.assertFalse(lock.exists())

    def test_coverage_sidecar_reports_progress(self) -> None:
        """覆盖率附带文件：报字库大小、已覆盖、剩余，以及本轮实测速度。"""
        out = self.dir / "cov.jsonl"
        stats = run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        self.assertEqual(stats["coverage"]["pool"], 4)
        self.assertGreaterEqual(stats["coverage"]["covered"], 1)
        self.assertLessEqual(stats["coverage"]["covered"], stats["coverage"]["pool"])
        sidecar = json.loads((self.dir / "cov.jsonl.coverage.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["pool"], 4)
        self.assertIn("measured_seconds_per_record", sidecar)
        self.assertIn("eta_hours_with_workers", sidecar)

    def test_coverage_sidecar_counts_whole_file_not_just_this_run(self) -> None:
        """附带报告的 by_task / records 是累计进度，本轮增量单独放 this_run，两者不混。"""
        out = self.dir / "cum.jsonl"
        run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        first = json.loads((self.dir / "cum.jsonl.coverage.json").read_text(encoding="utf-8"))
        run_pipeline(PipelineConfig(limit=2, out_path=out, data_path=self.data), client=PayloadEchoClient())
        second = json.loads((self.dir / "cum.jsonl.coverage.json").read_text(encoding="utf-8"))
        self.assertEqual(second["records"], sum(second["by_task"].values()))
        self.assertGreater(second["records"], first["records"])
        self.assertEqual(second["this_run"]["resumed"], first["records"])
        self.assertLessEqual(second["this_run"]["ok"], 2)

    def test_records_carry_difficulty_note(self) -> None:
        """每条产出都带难度注记（解析式先验），并明确标注 kind，避免被当成实测标定。"""
        out = self.dir / "note.jsonl"
        run_pipeline(PipelineConfig(limit=4, out_path=out, data_path=self.data), client=PayloadEchoClient())
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["difficulty"]["kind"], "analytic-prior")
            self.assertGreater(row["difficulty"]["target_facility"], 0.0)

    def test_rejects_carry_replay_payload(self) -> None:
        """rejects 里必须存完整 spec 提示内容，否则重跑时无从重建题目。"""
        out = self.dir / "rej.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0), client=FlakyClient(bad_rounds=1)
        )
        rejects = [
            json.loads(line)
            for line in (self.dir / "rej.rejects.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rejects), 1)
        payload = rejects[0]["spec"]
        self.assertEqual(payload["task_id"], rejects[0]["task_id"])
        self.assertTrue(payload["answer"])
        self.assertIn("facts", payload)
        self.assertIn("difficulty", payload)
        # 校验判据也要写入文件：丢了它们，重跑路径会比实时路径松（有单测盯着这条）
        self.assertIn("stem_asks_any", payload)
        self.assertIn("structure_words", payload)
        self.assertIn("hidden_subject", payload)

    def test_replay_recovers_rejects_into_main_output(self) -> None:
        """L7 飞轮：rejects 里的失败样本单条重试，成功就追加进主产出。"""
        out = self.dir / "fly.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0), client=FlakyClient(bad_rounds=1)
        )
        rejects = self.dir / "fly.rejects.jsonl"
        stats = replay_rejects(
            PipelineConfig(out_path=out, data_path=self.data, retries=1),
            rejects,
            client=PayloadEchoClient(),
        )
        self.assertEqual(stats["recovered"], 1)
        self.assertEqual(stats["still_failed"], 0)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["stem"])

    def test_replay_is_idempotent(self) -> None:
        """重跑可重复跑：已进主产出的 (题型, 字) 直接跳过，不重复花配额。"""
        out = self.dir / "idem.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0), client=FlakyClient(bad_rounds=1)
        )
        rejects = self.dir / "idem.rejects.jsonl"
        config = PipelineConfig(out_path=out, data_path=self.data, retries=1)
        self.assertEqual(replay_rejects(config, rejects, client=PayloadEchoClient())["recovered"], 1)
        second = replay_rejects(config, rejects, client=PayloadEchoClient())
        self.assertEqual(second["skipped_existing"], 1)
        self.assertEqual(second["planned"], 0)
        self.assertEqual(second["recovered"], 0)

    def test_replay_reports_unreplayable_rows(self) -> None:
        """老格式 rejects（没有 spec 提示内容）如实计入 unreplayable，并另存新文件，原文件不动。"""
        out = self.dir / "old.jsonl"
        rejects = self.dir / "old.rejects.jsonl"
        rejects.write_text(
            json.dumps({"task_id": "T1_stroke_total", "ziid": "U00000"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        before = rejects.read_text(encoding="utf-8")
        stats = replay_rejects(
            PipelineConfig(out_path=out, data_path=self.data), rejects, client=PayloadEchoClient()
        )
        self.assertEqual(stats["unreplayable"], 1)
        self.assertEqual(stats["planned"], 0)
        self.assertEqual(stats["left"], 1)
        self.assertTrue(Path(stats["left_path"]).exists())
        self.assertEqual(rejects.read_text(encoding="utf-8"), before)

    def test_replay_of_replayed_file_does_not_nest(self) -> None:
        """拿重跑产物再重跑：产出仍写回同一个 <x>.replayed.jsonl，不生成更深一层。"""
        out = self.dir / "nest.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0), client=FlakyClient(bad_rounds=1)
        )
        config = PipelineConfig(out_path=out, data_path=self.data, retries=1)
        first = replay_rejects(config, self.dir / "nest.rejects.jsonl", client=PayloadEchoClient())
        left = Path(first["left_path"])
        self.assertEqual(left.name, "nest.rejects.jsonl.replayed.jsonl")
        second = replay_rejects(config, left, client=PayloadEchoClient())
        self.assertEqual(Path(second["left_path"]), left)
        self.assertFalse((self.dir / "nest.rejects.jsonl.replayed.jsonl.replayed.jsonl").exists())

    def test_replay_left_path_strips_every_suffix(self) -> None:
        """路径规范化：无论输入套了几层 .replayed.jsonl，输出都只有一层。"""
        from hanzi_synth.pipeline.run import _replay_left_path

        base = self.dir / "x.rejects.jsonl"
        self.assertEqual(_replay_left_path(base), self.dir / "x.rejects.jsonl.replayed.jsonl")
        deep = self.dir / "x.rejects.jsonl.replayed.jsonl.replayed.jsonl.replayed.jsonl"
        self.assertEqual(_replay_left_path(deep), self.dir / "x.rejects.jsonl.replayed.jsonl")

    def test_replay_refuses_offline(self) -> None:
        """重跑必须联网措辞：offline 直接报错，不静默空跑。"""
        with self.assertRaises(ConfigError):
            replay_rejects(
                PipelineConfig(offline=True, out_path=self.dir / "x.jsonl", data_path=self.data),
                self.dir / "x.rejects.jsonl",
                client=PayloadEchoClient(),
            )


class _FakeReplay:
    """输入：每一轮要返回的统计。输出：鸭子类型的重跑函数。
    预期：记下每一轮收到的路径（断言「下一轮吃的是上一轮的残留文件」），脚本用尽还被打
          就报错——那正是「一轮没救回还继续喂」的空转。
    """

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> dict:
        self.paths.append(Path(path))
        if not self._script:
            raise AssertionError("自动重跑多跑了一轮：脚本里没有这一轮的预设")
        return self._script.pop(0)


def _round_stats(planned: int, recovered: int, left: str = "x.rejects.jsonl.replayed.jsonl",
                 still_failed: int = 0) -> dict:
    """输入：一轮的关键计数。输出：重跑统计 dict。预期：字段名与 `replay_rejects` 一致。"""
    return {
        "planned": planned,
        "recovered": recovered,
        "still_failed": still_failed,
        "left_path": left,
        "started": "2026-10-08 00:00:00",
        "finished": "2026-10-08 00:00:01",
    }


class AutoReplayTests(unittest.TestCase):
    """输入：假的单轮重跑函数。输出：断言。预期：收敛判据只有「这一轮有没有新救回」——
    一条都没救回就立刻停（同一批输入 + 同一温度 + 没有状态变化，再喂只会复现同一批失败）。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.rejects = self.dir / "x.rejects.jsonl"
        self.rejects.write_text("{}\n", encoding="utf-8")
        # 残留文件：`auto_replay` 只吃真实存在的文件，假重跑也得把它写出来。
        self.leftover = self.dir / "x.rejects.jsonl.replayed.jsonl"
        self.leftover.write_text("{}\n", encoding="utf-8")

    def test_stops_when_a_round_recovers_nothing(self) -> None:
        """零救回的那一轮之后必须刹车：这是主人点名的「一轮没救回还继续喂」死循环。"""
        replay = _FakeReplay([_round_stats(planned=3, recovered=0, still_failed=3)])
        rounds = auto_replay(replay, self.rejects, max_rounds=5)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(replay.paths, [self.rejects])

    def test_keeps_going_while_rounds_recover(self) -> None:
        """有净救回就继续，且下一轮吃的是上一轮的残留文件（不是原 rejects）。"""
        replay = _FakeReplay([
            _round_stats(planned=3, recovered=2, left=str(self.leftover)),
            _round_stats(planned=1, recovered=0, left=str(self.leftover), still_failed=1),
        ])
        rounds = auto_replay(replay, self.rejects, max_rounds=5)
        self.assertEqual(len(rounds), 2)
        self.assertEqual(replay.paths, [self.rejects, self.leftover])

    def test_stops_when_there_is_nothing_to_replay(self) -> None:
        """残留文件里全是老格式（无 spec 提示内容）时 planned=0，一轮就停。"""
        replay = _FakeReplay([_round_stats(planned=0, recovered=0)])
        self.assertEqual(len(auto_replay(replay, self.rejects, max_rounds=5)), 1)

    def test_respects_max_rounds(self) -> None:
        """一直有救回时最多跑 max_rounds 轮，不会无限重跑。"""
        replay = _FakeReplay(
            [_round_stats(planned=9, recovered=1, left=str(self.leftover)) for _ in range(5)]
        )
        self.assertEqual(len(auto_replay(replay, self.rejects, max_rounds=2)), 2)
        self.assertEqual(len(replay.paths), 2)

    def test_missing_rejects_file_runs_no_round(self) -> None:
        """起始文件不存在（主流程一条都没判失败）时一轮都不跑，也不报错。"""
        replay = _FakeReplay([])
        self.assertEqual(auto_replay(replay, self.dir / "nope.rejects.jsonl", max_rounds=3), [])
        self.assertEqual(replay.paths, [])

    def test_round_stats_drop_bookkeeping_timestamps(self) -> None:
        """逐轮统计去掉起止时刻：它们是内部记账，不该混进 CLI 打印的 JSON。"""
        replay = _FakeReplay([_round_stats(planned=1, recovered=0)])
        rounds = auto_replay(replay, self.rejects, max_rounds=1)
        self.assertNotIn("started", rounds[0])
        self.assertNotIn("finished", rounds[0])
        self.assertEqual(rounds[0]["planned"], 1)

    def test_say_reports_every_stop_reason(self) -> None:
        """进度回调必须说清「为什么停」：零救回与无可重跑是两种不同的收尾原因。"""
        lines: list[str] = []
        replay = _FakeReplay([_round_stats(planned=2, recovered=0, still_failed=2)])
        auto_replay(replay, self.rejects, max_rounds=3, say=lines.append)
        self.assertTrue(any("一条都没救回" in line for line in lines), lines)


class AttemptSummaryTests(unittest.TestCase):
    """输入：手写的 attempts 文件。输出：断言。预期：首轮通过 / 被救回 / 仍失败三类分清，
    并指出首轮是哪条校验规则、哪种语体把题拦下来的。
    """

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "x.attempts.jsonl"

    def _write(self, rows: list[dict]) -> None:
        """输入：行列表。输出：无。预期：写成 jsonl，中文不转义。"""
        self.path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def test_splits_first_pass_recovered_and_still_failed(self) -> None:
        """三条题分别走三条路：首轮就过 / 带原因救回 / 重试也没救回。"""
        from hanzi_synth.pipeline.run import summarize_attempts

        self._write([
            {"task_id": "T1", "ziid": "A", "round": 1, "style": "字典条目", "tier": "A", "failures": []},
            {"task_id": "T2", "ziid": "B", "round": 1, "style": "教材讲解", "tier": "C",
             "failures": ["options_ok: 抄了选项"]},
            {"task_id": "T2", "ziid": "B", "round": 2, "style": "问答对话", "tier": "A", "failures": []},
            {"task_id": "T3", "ziid": "C", "round": 1, "style": "教材讲解", "tier": "C",
             "failures": ["options_ok: 抄了选项"]},
            {"task_id": "T3", "ziid": "C", "round": 2, "style": "考试题干", "tier": "C",
             "failures": ["subject_visible: 没写主语"]},
        ])
        report = summarize_attempts(self.path)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["first_pass"], 1)
        self.assertEqual(report["recovered"], 1)
        self.assertEqual(report["still_failed"], 1)
        self.assertEqual(report["by_first_defense"], {"options_ok": 2})

    def test_judge_veto_is_not_counted_as_first_pass(self) -> None:
        """裁判否决那轮的 tier 是 A（程序校验规则的结论）却没发布：attempt-stats 必须记成首轮没过，
        并把「裁判否决」记进 by_first_defense；裁判**缺席**（没跑成）同样不算通过，
        记进「裁判缺席」桶——没被检验过的题不能算过。
        """
        from hanzi_synth.pipeline.run import summarize_attempts

        self._write([
            {"task_id": "T1", "ziid": "A", "round": 1, "style": "字典条目", "tier": "A",
             "failures": [], "judge": {"ok": False, "reason": "题干有歧义", "error": ""}},
            {"task_id": "T1", "ziid": "A", "round": 2, "style": "教材讲解", "tier": "A",
             "failures": [], "judge": {"ok": True, "reason": "无", "error": ""}},
            {"task_id": "T2", "ziid": "B", "round": 1, "style": "字典条目", "tier": "A",
             "failures": [], "judge": {"ok": False, "error": "boom"}},
        ])
        report = summarize_attempts(self.path)
        self.assertEqual(report["first_pass"], 0)   # T2：裁判缺席也不是通过
        self.assertEqual(report["recovered"], 1)    # T1：第二轮裁判放行
        self.assertEqual(report["still_failed"], 1) # T2：只有一轮，且那轮缺席
        self.assertEqual(report["by_first_defense"], {"裁判否决": 1, "裁判缺席": 1})

    def test_missing_file_raises(self) -> None:
        """文件不存在时报人话错误，不静默返回空汇总。"""
        from hanzi_synth.pipeline.run import summarize_attempts

        with self.assertRaises(DataError):
            summarize_attempts(self.dir / "nope.attempts.jsonl")


class AttemptLogTests(unittest.TestCase):
    """输入：假客户端。输出：断言。预期：跑一次真管线就把每轮判定落进 <out>.attempts.jsonl。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def test_every_round_is_persisted(self) -> None:
        """首轮被拒、次轮救回：日志里必须两条都在，首轮那条带失败原因。"""
        out = self.dir / "att.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=1),
            client=FlakyClient(bad_rounds=1),
        )
        rows = [
            json.loads(line)
            for line in out.with_suffix(".attempts.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["round"], 1)
        self.assertNotEqual(rows[0]["tier"], "A")
        self.assertTrue(rows[0]["failures"])
        self.assertEqual(rows[1]["round"], 2)
        self.assertEqual(rows[1]["tier"], "A")


class JudgeWiringTests(unittest.TestCase):
    """输入：假措辞客户端 + 脚本化裁判客户端。输出：断言。预期：裁判**只在程序校验规则全过后**出场；
    裁判否决的题进 rejects 并留下理由；裁判缺席（跑挂了）先重问、重问完仍缺席就不发布；
    批量模式与裁判互斥。
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def _run(self, name: str, judge: ScriptedClient, **kwargs):
        """输入：产出文件名 + 裁判客户端 + 额外配置。输出：统计 dict。预期：措辞用规范的假客户端。"""
        out = self.dir / name
        return out, run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0, judge=True, **kwargs),
            client=PayloadEchoClient(),
            judge_client=judge,
        )

    def test_judge_reject_sends_record_to_rejects_with_reason(self) -> None:
        """裁判否决 = 不进产出，改走 rejects，且理由必须写入文件。"""
        out, stats = self._run("rej.jsonl", ScriptedClient(['{"ok": false, "reason": "题干里写着写作要求"}']))
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(out.read_text(encoding="utf-8").strip(), "")
        rows = [
            json.loads(line)
            for line in (self.dir / "rej.rejects.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(rows[0]["judge"]["ok"], False)
        self.assertEqual(rows[0]["judge"]["reason"], "题干里写着写作要求")

    def test_judge_pass_publishes_and_records_verdict(self) -> None:
        """裁判放行 = 进产出，且判定如实写入文件（事后要靠它标定裁判）。"""
        out, stats = self._run("pass.jsonl", ScriptedClient(['{"ok": true, "reason": "无"}']))
        self.assertEqual(stats["ok"], 1)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows[0]["judge"]["ok"])

    def test_absent_judge_is_reasked_until_it_answers(self) -> None:
        """裁判第一次吐坏输出 = 缺席（工具没干成活），重问；第二次真答了就按它走。"""
        judge = ScriptedClient(["这不是 JSON", '{"ok": true, "reason": "无"}'])
        out, stats = self._run("judge_retry.jsonl", judge)
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(judge.calls, 2)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows[0]["judge"]["ok"])

    def test_absent_judge_keeps_the_record_out_of_the_output(self) -> None:
        """重问完仍缺席 = 这条题没被检验过 → 不发布，进 rejects（可重跑，题不丢）。"""
        out, stats = self._run("absent.jsonl", ScriptedClient(["这不是 JSON"]))
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 1)
        rows = [
            json.loads(line)
            for line in out.with_suffix(".rejects.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(any("裁判缺席" in failure for failure in rows[0]["failures"]), rows[0]["failures"])

    def test_judge_is_never_called_when_rules_fail(self) -> None:
        """程序校验规则没过时一次都不该叫裁判——规则优先，裁判是最后一道。"""
        judge = ScriptedClient([])
        out = self.dir / "never.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0, judge=True),
            client=FlakyClient(bad_rounds=1),
            judge_client=judge,
        )
        self.assertEqual(judge.calls, 0)

    def test_judge_off_by_default(self) -> None:
        """默认不开裁判：不传 judge_client 时一次请求都不该多花。"""
        judge = ScriptedClient([])
        out = self.dir / "off.jsonl"
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=0),
            client=PayloadEchoClient(),
        )
        self.assertEqual(judge.calls, 0)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertIsNone(rows[0]["judge"])

    def test_judge_requests_are_counted(self) -> None:
        """裁判也是 provider 请求：requests 必须把它算进去，否则配额账目会漏。"""
        _out, stats = self._run("count.jsonl", ScriptedClient(['{"ok": true, "reason": "无"}']))
        self.assertEqual(stats["requests"], 2)  # 1 次措辞 + 1 次裁判

    def test_veto_reason_lands_in_reject_failures(self) -> None:
        """裁判否决的理由必须进 rejects 的 failures——否则 reject-stats 里这条题看不出为什么被拒
        （首版就是 failures 为空、理由只躺在 judge 字段里）。
        """
        from hanzi_synth.pipeline.run import summarize_rejects

        _out, stats = self._run("rejfail.jsonl", ScriptedClient(['{"ok": false, "reason": "题干有歧义"}']))
        self.assertEqual(stats["rejected"], 1)
        reject_path = self.dir / "rejfail.rejects.jsonl"
        rows = [json.loads(line) for line in reject_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["failures"], ["裁判否决: 题干有歧义"])
        self.assertEqual(summarize_rejects(reject_path)["by_defense"], {"裁判否决": 1})

    def test_veto_reason_reaches_next_round_with_fix_hint(self) -> None:
        """裁判否决的理由必须回灌给下一轮：不告诉模型错在哪，它会在同一处再摔。
        首轮程序校验规则没过 -> 不叫裁判；次轮程序过了被裁判否决 -> 第三轮必须带着理由重写。
        """
        from hanzi_synth.verify.checks import JUDGE_FEEDBACK_KEY

        judge = ScriptedClient([
            '{"ok": false, "reason": "题干有歧义"}',
            '{"ok": true, "reason": "无"}',
        ])
        client = RecordingFlakyClient(bad_rounds=1)
        out = self.dir / "hint.jsonl"
        stats = run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, retries=2, judge=True),
            client=client,
            judge_client=judge,
        )
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(len(client.requests), 3)
        last = json.dumps(client.requests[-1], ensure_ascii=False)
        self.assertIn("题干有歧义", last)
        # 校验名是给 rejects 看的行话，不进模型上下文——模型只需要「哪里不对」。
        self.assertNotIn(JUDGE_FEEDBACK_KEY, last)

    def test_batch_and_judge_are_mutually_exclusive(self) -> None:
        """批量模式与裁判互斥：静默忽略开关比报错更糟，所以直接抛。"""
        with self.assertRaises(ConfigError):
            run_pipeline(
                PipelineConfig(
                    limit=1, out_path=self.dir / "x.jsonl", data_path=self.data, batch=2, judge=True
                ),
                client=PayloadEchoClient(),
            )


class BetterRoundTests(unittest.TestCase):
    """输入：逐轮判定记录。输出：选中的那一轮。

    预期：重试用尽时返回**最好的一轮**，不是最后一轮——实测第二轮退化（T10 整段输出散文、
    T5_variant 把提示内容原样回显）会把第一轮的合格产出顶掉，rejects 里记的也变成第二轮的原因。
    """

    def test_prefers_tier_a_over_tier_c(self) -> None:
        from hanzi_synth.pipeline.run import _better_round

        history = [
            {"tier": "A", "failures": [], "judge": None, "roundtrip": None},
            {"tier": "C", "failures": ["schema: 缺少 stem 位置"], "judge": None, "roundtrip": None},
        ]
        self.assertEqual(_better_round(history, 1, 0), 0)

    def test_prefers_the_round_without_a_roundtrip_veto(self) -> None:
        from hanzi_synth.pipeline.run import _better_round

        history = [
            {"tier": "A", "failures": [], "judge": None, "roundtrip": {"ok": True, "error": ""}},
            {"tier": "A", "failures": [], "judge": None, "roundtrip": {"ok": False, "error": ""}},
        ]
        self.assertEqual(_better_round(history, 1, 0), 0)

    def test_absent_roundtrip_is_not_preferred(self) -> None:
        """缺席的一轮不再被当成最好的一轮：缺席与否决同权（都没通过），没通过的轮次不该被选中。"""
        from hanzi_synth.pipeline.run import _better_round

        history = [
            {"tier": "A", "failures": [], "judge": None, "roundtrip": {"ok": False, "error": "boom"}},
            {"tier": "A", "failures": ["x"], "judge": None, "roundtrip": None},
        ]
        self.assertEqual(_better_round(history, 1, 0), 1)

    def test_fewer_failures_wins_and_ties_keep_the_earlier_round(self) -> None:
        from hanzi_synth.pipeline.run import _better_round

        history = [
            {"tier": "C", "failures": ["a", "b"], "judge": None, "roundtrip": None},
            {"tier": "C", "failures": ["a"], "judge": None, "roundtrip": None},
        ]
        self.assertEqual(_better_round(history, 0, 1), 1)
        same = [dict(history[0]), dict(history[0])]
        self.assertEqual(_better_round(same, 0, 1), 0)

    def test_chosen_row_falls_back_to_the_last_row(self) -> None:
        from hanzi_synth.pipeline.run import _chosen_row

        self.assertEqual(_chosen_row([{"round": 1}, {"round": 2, "chosen": True}])["round"], 2)
        self.assertEqual(_chosen_row([{"round": 1}])["round"], 1)
        self.assertEqual(_chosen_row([]), {})


class RoundTripRetryTests(unittest.TestCase):
    """输入：第一轮程序全过但读者答错、第二轮整段退化的假客户端。输出：rejects 行。

    预期：rejects 里记的是**被选中那一轮**的原因（读者答错），不是第二轮退化的原因。
    """

    class _DegradingClient:
        """输入：无。输出：第一轮规范、第二轮吐散文的假客户端；读者轮固定答错。"""

        def __init__(self) -> None:
            self.calls = 0
            self.answer = ""

        def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
            """输入：messages。输出：按角色与轮次给出的响应。"""
            self.calls += 1
            payload = payload_of(messages)
            if "answer" not in payload:  # 读者轮：只有 stem / options
                # 错答案要**在题干里找得到出处**，否则按「读者在猜」处理、不否决（见 roundtrip 模块）。
                wrong = read_wrong_answer_from_stem(self.answer, _last_json_object(messages))
                return Completion(ok=True, text=json.dumps({"answer": wrong}, ensure_ascii=False),
                                  provider="scripted", model="m")
            self.answer = str(payload["answer"])
            if self.calls == 1:
                return Completion(ok=True, text=compliant_body(payload), provider="scripted", model="m")
            return Completion(ok=True, text="题目要求从下列汉字中选出答案。",
                              provider="scripted", model="m")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def test_reject_keeps_the_best_round_reason(self) -> None:
        from hanzi_synth.verify.checks import ROUNDTRIP_FEEDBACK_KEY

        out = self.dir / "degrade.jsonl"
        stats = run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, seed=7, retries=1, roundtrip=True),
            client=self._DegradingClient(),
        )
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 1)
        rows = [
            json.loads(line)
            for line in out.with_suffix(".rejects.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(
            any(ROUNDTRIP_FEEDBACK_KEY in failure for failure in rows[0]["failures"]),
            rows[0]["failures"],
        )
        attempts = [
            json.loads(line)
            for line in out.with_suffix(".attempts.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[0]["chosen"])
        self.assertFalse(attempts[1].get("chosen", False))


class RoundPassedTests(unittest.TestCase):
    """输入：tier / 裁判 / 读者三件判定。输出：这一轮过没过。

    预期：缺席（裁判 / 读者没跑成）**既不是否决也不是通过**——没被检验过的题不能发布；
          没要求做这条检验（None）不算缺席，不阻塞发布。
    """

    def test_absence_is_not_a_pass(self) -> None:
        from hanzi_synth.pipeline.run import _round_passed

        self.assertFalse(_round_passed("A", None, {"ok": False, "error": "boom"}))
        self.assertFalse(_round_passed("A", {"ok": False, "error": "boom"}, None))
        self.assertTrue(_round_passed("A", None, {"ok": True, "error": ""}))
        self.assertTrue(_round_passed("A", None, None))
        self.assertFalse(_round_passed("C", None, None))


class AbsentCheckRetryTests(unittest.TestCase):
    """输入：读者先缺席几次的假客户端。输出：断言。

    预期：缺席（请求挂了 / 只吐思考段 / 输出不是 JSON）是**工具没干成活**，不是「这段文字不合格」——
          同一条题重问（`CHECK_ATTEMPTS` 次）：重问出判定就按判定走；重问完仍缺席**不发布**，
          题进 rejects 重跑（题不丢，也不会带着「读者没读过的题」进数据集）。
    """

    class _ReaderClient:
        """输入：让第几次读者请求缺席（0 = 每次都缺席）。输出：假客户端；读者答对。"""

        def __init__(self, absent_at: int) -> None:
            self.absent_at = absent_at
            self.reader_calls = 0
            self.answer = ""

        def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
            """输入：messages。输出：读者轮按 absent_at 缺席或答对；措辞轮给规范正文。"""
            payload = payload_of(messages)
            body = _last_json_object(messages)
            if "explanation" in body and "answer" not in body:
                # 措辞检验（L4.7）的提示内容：只有 stem / explanation，没有答案也没有候选。
                # 夹具一律回「没意见」——本类测试钉的不是那条校验。
                return Completion(
                    ok=True, text=json.dumps({"phrases": [], "note": ""}), provider="scripted", model="m"
                )
            if "answer" not in payload:  # 读者轮：提示内容里只有题干与候选
                self.reader_calls += 1
                if self.absent_at == 0 or self.reader_calls == self.absent_at:
                    return Completion(ok=False, text="", provider="scripted", model="m", error="boom")
                return Completion(
                    ok=True, text=json.dumps({"answer": self.answer}, ensure_ascii=False),
                    provider="scripted", model="m",
                )
            self.answer = str(payload["answer"])
            return Completion(ok=True, text=compliant_body(payload), provider="scripted", model="m")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def _run(self, name: str, client) -> tuple[Path, dict]:
        """输入：产出文件名 + 假客户端。输出：(产出路径, 统计)。预期：开往返检验、单轮。"""
        out = self.dir / name
        return out, run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, seed=7, retries=0, roundtrip=True),
            client=client,
        )

    def test_reader_is_reasked_after_an_absence(self) -> None:
        """第一次读者请求缺席 → 重问同一道题（不重新措辞），第二次读出判定 → 发布。"""
        client = self._ReaderClient(absent_at=1)
        out, stats = self._run("retry.jsonl", client)
        self.assertEqual(stats["ok"], 1)
        self.assertEqual(client.reader_calls, 2)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows[0]["roundtrip"]["ok"])

    def test_persistent_absence_keeps_the_record_in_rejects(self) -> None:
        """重问完仍读不成 → 不发布，进 rejects，理由单列「读者缺席」桶。"""
        client = self._ReaderClient(absent_at=0)
        out, stats = self._run("absent.jsonl", client)
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(client.reader_calls, 3)
        rows = [
            json.loads(line)
            for line in out.with_suffix(".rejects.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(any("读者缺席" in failure for failure in rows[0]["failures"]), rows[0]["failures"])

    def test_a_reader_reply_without_an_answer_is_absent_not_a_pass(self) -> None:
        """读者回了**合法 JSON 但没有答案**（只回一句说明）→ 同样按缺席处理，不发布。

        预期：这条堵的是「字段级缺席被当成通过」——记录里不能写着「读者检验 ok」，而读者其实
              一个字都没答。走的是与「请求挂了」同一条路：重问；仍缺席进 rejects 的「读者缺席」桶。
        """

        class _NoteOnlyClient(AbsentCheckRetryTests._ReaderClient):
            """输入：messages。输出：读者轮只回一句说明；措辞轮照旧。预期：模拟漏 answer 的读者。"""

            def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
                """输入：messages。输出：读者轮回 note、措辞轮交回父类。"""
                payload = payload_of(messages)
                if "answer" not in payload:
                    self.reader_calls += 1
                    return Completion(
                        ok=True, text=json.dumps({"note": "我只读了一句"}, ensure_ascii=False),
                        provider="scripted", model="m",
                    )
                return super().complete(messages, temperature=temperature, extra=extra)

        client = _NoteOnlyClient(absent_at=0)
        out, stats = self._run("note_only.jsonl", client)
        self.assertEqual(stats["ok"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(client.reader_calls, 3)
        rows = [
            json.loads(line)
            for line in out.with_suffix(".rejects.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(any("读者缺席" in failure for failure in rows[0]["failures"]), rows[0]["failures"])


class RejectSummaryTests(unittest.TestCase):
    """输入：手写的 rejects / replayed 文件。输出：断言。预期：汇总按校验规则名与题型计数，
    且**原文件一字不动**（rejects 是只增日志，汇总工具只能读）。
    """

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "x.rejects.jsonl"

    def _write(self, path: Path, rows: list[dict]) -> None:
        """输入：目标路径 + 行列表。输出：无。预期：写成 jsonl，中文不转义。"""
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def test_counts_by_defense_and_task(self) -> None:
        """同一校验规则在一行里出现两次要计两次；汇总后原文件字节不变。"""
        from hanzi_synth.pipeline.run import summarize_rejects

        self._write(self.path, [
            {"task_id": "T4_reading", "failures": ["no_answer_leak: x", "no_answer_leak: y"], "spec": {"a": 1}},
            {"task_id": "T4_reading", "failures": ["self_contained: z"], "spec": {"a": 1}},
            {"task_id": "T1_stroke_total", "failures": ["subject_visible: w"], "error": "boom", "spec": {"a": 1}},
        ])
        before = self.path.read_bytes()
        report = summarize_rejects(self.path)
        self.assertEqual(report["lines"], 3)
        self.assertEqual(report["by_task"], {"T1_stroke_total": 1, "T4_reading": 2})
        self.assertEqual(report["by_defense"], {"no_answer_leak": 2, "self_contained": 1, "subject_visible": 1})
        self.assertEqual(report["by_error"], {"boom": 1})
        self.assertEqual(report["unreplayable"], 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_reports_replayed_sibling_and_unreplayable(self) -> None:
        """没有 spec 提示内容的旧格式记 unreplayable；同级 .replayed.jsonl 一并汇总。"""
        from hanzi_synth.pipeline.run import summarize_rejects

        self._write(self.path, [{"task_id": "T4_reading", "failures": [], "spec": None}])
        self._write(
            self.path.with_name(self.path.name + ".replayed.jsonl"),
            [{"task_id": "T4_reading", "failures": ["no_answer_leak: x"], "spec": {"a": 1}}],
        )
        report = summarize_rejects(self.path)
        self.assertEqual(report["unreplayable"], 1)
        self.assertEqual(report["replayed"]["lines"], 1)
        self.assertEqual(report["replayed"]["by_defense"], {"no_answer_leak": 1})

    def test_missing_file_raises(self) -> None:
        """文件不存在时报人话错误，不静默返回空汇总。"""
        from hanzi_synth.pipeline.run import summarize_rejects

        with self.assertRaises(DataError):
            summarize_rejects(self.dir / "nope.rejects.jsonl")

class StemGateRecoveryTests(unittest.TestCase):
    """复验要按**当前**题型校验判定：旧记录的 `stem_asks_any` 是空的（生产时还没这道校验）。"""

    def test_spec_from_record_fills_the_gate_from_the_task_table(self) -> None:
        """输入：一份 `stem_asks_any` 为空的旧提示内容。输出：补上该题型当前声明的校验。
        预期：不补的话 `stem_asks` 那一关整条跳过，「题干问读音、答案是笔画数」的废题会被判 A。
        """
        spec = spec_from_record(
            {"task_id": "T1_stroke_total", "instruction": "i", "facts": {}, "answer": "4", "stem_asks_any": []}
        )
        self.assertEqual(spec.stem_asks_any, TASK_STEM_ASKS["T1_stroke_total"])

    def test_spec_from_record_keeps_a_recorded_gate(self) -> None:
        """反例：提示内容自己带了校验就原样用，不被题型表覆盖。"""
        spec = spec_from_record(
            {
                "task_id": "T1_stroke_total",
                "instruction": "i",
                "facts": {},
                "answer": "4",
                "stem_asks_any": ["\u81ea\u5b9a\u4e49"],
            }
        )
        self.assertEqual(spec.stem_asks_any, ("\u81ea\u5b9a\u4e49",))

    def test_spec_from_record_unknown_task_gets_no_gate(self) -> None:
        """未知题型（如已下线的 T17）补不出校验，退回空元组——不猜。"""
        spec = spec_from_record(
            {"task_id": "T17_stroke_rank", "instruction": "i", "facts": {}, "answer": "x", "stem_asks_any": []}
        )
        self.assertEqual(spec.stem_asks_any, ())


class ExcludePathsTests(unittest.TestCase):
    """`--exclude`：开新一轮时把已有产出的主键也跳过，覆盖才真的往前走。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def _ziids(self, path: Path) -> set[str]:
        """输入：产出文件。输出：里面的 ziid 集合。"""
        return {json.loads(line)["ziid"] for line in path.read_text(encoding="utf-8").splitlines()}

    def test_excluded_round_samples_new_characters(self) -> None:
        """预期：第二轮把第一轮的主键排除掉后，采到的是**另一批**字，两轮 ziid 不重合。"""
        first = self.dir / "round1.jsonl"
        second = self.dir / "round2.jsonl"
        one = run_pipeline(
            PipelineConfig(
                limit=2, out_path=first, data_path=self.data, offline=True,
                task_ids=("T1_stroke_total",), resume=False,
            )
        )
        self.assertEqual(one["ok"], 2)
        two = run_pipeline(
            PipelineConfig(
                limit=2, out_path=second, data_path=self.data, offline=True,
                task_ids=("T1_stroke_total",), resume=False, exclude_paths=(first,),
            )
        )
        self.assertEqual(two["excluded"], 2)
        self.assertEqual(two["ok"], 2)
        self.assertFalse(self._ziids(first) & self._ziids(second), "两轮采到了同一批字，覆盖没往前走")

    def test_without_exclude_the_next_round_repeats(self) -> None:
        """反例（记录当前行为）：不给 `--exclude` 时第二轮会重复采同一批字——这正是要修的病。"""
        first = self.dir / "r1.jsonl"
        second = self.dir / "r2.jsonl"
        for path in (first, second):
            run_pipeline(
                PipelineConfig(
                    limit=2, out_path=path, data_path=self.data, offline=True,
                    task_ids=("T1_stroke_total",), resume=False,
                )
            )
        self.assertEqual(self._ziids(first), self._ziids(second))


class PlanCompletenessTests(unittest.TestCase):
    """`_plan` 不许静默丢 (题型, 字) 组合：桶里每一条都是真实可出题的组合。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def test_every_buildable_combination_reaches_the_plan(self) -> None:
        """预期：planned 的 (题型, 字) 集合 == 逐字 `build_all` 的吐量——轮转取样只负责排序。

        反例（修复前）：同一个字能出两个题型时，第二型会让位给别的字并被**永久丢掉**，
        4 个字只排出 4 条（应为 8 条）。让位的候选现在进 `deferred`，第二轮补齐。
        """
        config = PipelineConfig(
            limit=1000, out_path=self.dir / "plan.jsonl", data_path=self.data, offline=True,
            task_ids=("T1_stroke_total", "T3_ids_forward"), resume=False,
        )
        characters = load_characters(self.data)
        index = CharacterIndex(characters)

        planned, _scanned, _skipped = _plan(config, index, random.Random(config.seed), exclude=())

        expected: list[tuple[str, str]] = []
        rng = random.Random(config.seed)
        for character in characters:
            for spec in build_all(
                character, rng=rng, index=index, difficulty=config.difficulty(), only=config.task_ids
            ):
                expected.append((spec.task_id, character.ziid))
        self.assertGreater(
            len(expected), len({ziid for _task, ziid in expected}), "字库太小，测不出丢组合"
        )
        self.assertEqual(Counter((spec.task_id, ch.ziid) for ch, spec in planned), Counter(expected))

    def test_small_limit_still_spreads_across_characters(self) -> None:
        """小批量仍然要铺开字：`--limit` 小于字库时，第一轮就把字分散开，不重复用字。"""
        config = PipelineConfig(
            limit=4, out_path=self.dir / "small.jsonl", data_path=self.data, offline=True,
            task_ids=("T1_stroke_total", "T3_ids_forward"), resume=False,
        )
        characters = load_characters(self.data)
        planned, _scanned, _skipped = _plan(
            config, CharacterIndex(characters), random.Random(config.seed), exclude=()
        )
        self.assertEqual(len(planned), 4)
        ziids = [character.ziid for character, _spec in planned]
        self.assertEqual(len(set(ziids)), 4, "同一个字被排了多次，字没铺开")


class LocalWorkerCapTests(unittest.TestCase):
    """输入：各种 provider 组合 + 并发数。输出：收口后的并发与说明。预期：只收本机端点。"""

    @staticmethod
    def _provider(name: str, base_url: str):
        return ProviderConfig(name=name, base_url=base_url, model="m", api_key="k")

    def test_local_endpoint_is_capped(self) -> None:
        """本机端点 + 并发 8 → 收到上限，且说明里带上原值，避免以为参数没生效。"""
        cfg = PipelineConfig(workers=8)
        capped, note = with_local_worker_cap(cfg, [self._provider("local", "http://127.0.0.1:1234/v1")])
        self.assertEqual(capped.workers, LOCAL_WORKER_CAP)
        self.assertIn("8", note)

    def test_remote_endpoint_keeps_workers(self) -> None:
        """远端网关不受上限约束——它的瓶颈是网络与网关限速，并发才值得开。"""
        cfg = PipelineConfig(workers=8)
        capped, note = with_local_worker_cap(cfg, [self._provider("edge", "https://api.example.com/v1")])
        self.assertEqual(capped.workers, 8)
        self.assertEqual(note, "")

    def test_under_cap_is_untouched(self) -> None:
        """并发本来就没超上限时一个字都不改（主人要开 1 路就是 1 路）。"""
        cfg = PipelineConfig(workers=1)
        capped, note = with_local_worker_cap(cfg, [self._provider("local", "http://localhost:1234/v1")])
        self.assertEqual(capped.workers, 1)
        self.assertEqual(note, "")

    def test_any_local_provider_triggers_cap(self) -> None:
        """混合列表里只要有一个本机端点就收口——本轮的措辞请求可能落到它身上。"""
        cfg = PipelineConfig(workers=8)
        capped, _ = with_local_worker_cap(
            cfg,
            [self._provider("edge", "https://api.example.com/v1"),
             self._provider("local", "http://[::1]:1234/v1")],
        )
        self.assertEqual(capped.workers, LOCAL_WORKER_CAP)

    def test_local_host_detection(self) -> None:
        """主机名判据：loopback 三种写法算本机，远端域名与局域网 IP 不算。"""
        self.assertTrue(is_local_endpoint("http://127.0.0.1:1234/v1"))
        self.assertTrue(is_local_endpoint("http://localhost:1234/v1"))
        self.assertTrue(is_local_endpoint("http://[::1]:1234/v1"))
        self.assertFalse(is_local_endpoint("https://api.edgefn.net/v1"))
        self.assertFalse(is_local_endpoint("http://192.168.1.9:1234/v1"))


def _last_json_object(messages) -> dict:
    """输入：messages。输出：最后一条能解析成 JSON 对象的 user 消息（取不到返回空字典）。
    预期：往返检验的提示内容没有 `answer` 键，`payload_of` 按定义认不出来，所以这里按形状单独取。
    """
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        data = extract_json(str(message.get("content", "")))
        if isinstance(data, dict):
            return data
    return {}


def read_answer_from_stem(expected: str, body: dict) -> str:
    """输入：这道题定死的答案 + 往返检验提示内容。输出：一个称职读者会答出的内容。

    预期：**往返检验的提示内容里没有解析**（有解析这道检验就自证了），所以假读者只能靠题干；
          而 `compliant_body` 的题干是「「好」是什么？」这种最朴素的问法，题干本身推不出答案。
          本夹具因此直接给出标准答案，用来钉**接线**：答对就发布、答错就不发布。
          它不负责证明「小模型读者能读懂」——那是真机探针的事。
    """
    return expected


def read_wrong_answer_from_stem(expected: str, body: dict) -> str:
    """输入：定死的答案 + 往返检验提示内容。输出：一个**在题干里找得到出处**的错答案。

    预期：钉住「题干把读者带偏了」那一半接线——读者照着题干里的料答错了（选了题干列出的
          另一个候选，或抄了题干里的值）。与「读者凭空猜一个值」分开：后者不否决，
          见 `llm.roundtrip._traceable_to_stem`（汉字题一大类是知识题，小模型编值是常态）。
    """
    options = [str(item) for item in (body.get("options") or []) if item and str(item) != expected]
    if options:
        return options[0]
    for char in reversed(str(body.get("stem") or "")):
        if char.strip() and char != expected:
            return char
    return "读不出答案"


def read_guessed_answer_from_stem(_expected: str, _body: dict) -> str:
    """输入：定死的答案 + 提示内容。输出：一个题干里根本没有的值（读者自己编的）。
    预期：与 `read_wrong_answer_from_stem` 成对，用来钉「猜出来的值不否决」那一半接线。
    """
    return "E1A0"


class RoundTripGateTests(unittest.TestCase):
    """输入：往返检验答对 / 答错两种假客户端。输出：记录发不发布。预期：答错就不发布。"""

    class _ReaderClient:
        """输入：读者函数 `(定死的答案, 往返提示内容) -> 读者答出的文本`。输出：假客户端。
        预期：先答 L3（规范正文），再当读者答一次。

        两步靠提示内容形状区分：L3 提示内容带 `answer` 键，往返检验提示内容只有 stem / options。
        往返提示内容**没有** `payload_of` 认得的 `answer` 键，所以读者那一步直接解析最后一条 user 消息；
        定死的答案由上一步的 L3 提示内容带过来（读者手上没有它，夹具只是拿它替读者「答对」）。
        """

        def __init__(self, reader) -> None:
            self._reader = reader
            self.calls = 0
            self.answer = ""

        def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
            self.calls += 1
            payload = payload_of(messages)
            body = _last_json_object(messages)
            if "explanation" in body and "answer" not in body:
                # 措辞检验（L4.7）的提示内容：只有 stem / explanation，没有答案也没有候选。
                # 夹具一律回「没意见」——本类测试钉的不是那条校验。
                return Completion(
                    ok=True, text=json.dumps({"phrases": [], "note": ""}), provider="scripted", model="m"
                )
            if "answer" not in payload:  # 往返检验的提示内容：只有 stem / options
                return Completion(
                    ok=True,
                    text=json.dumps({"answer": self._reader(self.answer, body)}, ensure_ascii=False),
                    provider="scripted", model="m",
                )
            self.answer = str(payload["answer"])
            return Completion(ok=True, text=compliant_body(payload), provider="scripted", model="m")

    def setUp(self) -> None:
        # 用与 `PipelineLiveTests` 同一份 4 字固定字库：`compliant_body` 只在那份字库上
        # 保证能过 L4（默认全池里有带 `must_contain` 的题型，它满足不了）。测试要钉的是
        # 往返校验的接线，不是 `compliant_body` 的覆盖面。
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def _run(self, reader, name: str) -> list[dict]:
        """输入：读者函数 + 产出文件名。输出：发布出来的记录。"""
        out = self.dir / name
        client = self._ReaderClient(reader)
        config = PipelineConfig(
            limit=1, out_path=out, data_path=self.data, seed=7, retries=0, roundtrip=True,
        )
        run_pipeline(config, client=client)
        if not out.exists():
            return []
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]

    def test_wrong_reader_answer_blocks_publication(self) -> None:
        """读者照着题干里的料答错（选了另一个候选 / 抄了题干里的值）→ 这道题没被发布。"""
        self.assertEqual(self._run(read_wrong_answer_from_stem, "wrong.jsonl"), [])

    def test_a_guessed_answer_does_not_block_publication(self) -> None:
        """读者凭空编一个题干里没有的值 → 不否决：那是读者的知识缺口，不是题目的毛病。"""
        self.assertEqual(len(self._run(read_guessed_answer_from_stem, "guessed.jsonl")), 1)

    def test_correct_reader_answer_is_published(self) -> None:
        """读者答出了定死的答案 → 照常发布。判据比对的是提示内容里定死的答案，不是模型的偏好。"""
        self.assertEqual(len(self._run(read_answer_from_stem, "right.jsonl")), 1)


class AnswerArbitrationWiringTests(unittest.TestCase):
    """输入：同时扮措辞 / 读者 / 仲裁三个角色的假客户端。输出：发不发布。

    预期：**答案仲裁不受 `--judge` 开关控制**——读者答出的是一句自然语言，与程序定死的答案
          对不上时要用 LLM 判「是不是同一个答案」（只做子串比对等于把自然语言当字符串）。
          判据仍是先规则后仲裁：子串比对过了就不发仲裁请求，一次配额都不多花。
          读者自报「缺条件」还要过旁证：机械校验确认题面写全了识别条件时按误报处理。
    """

    class _TripleClient:
        """输入：读者函数 + 仲裁结论。输出：假客户端。

        预期：一个客户端扮三个角色，靠提示内容形状分流——L3 提示内容带 `answer`、答案仲裁提示内容带
              `expected`、往返检验提示内容两者都没有（只有 stem / options）。
        """

        def __init__(self, reader, equivalent: bool = True) -> None:
            self._reader = reader
            self._equivalent = equivalent
            self.answer = ""
            self.arbitrations = 0

        def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
            """输入：messages。输出：按提示内容形状分流后的响应。"""
            # `payload_of` 只认带 `answer` 的 L3 提示内容，所以另外两种提示内容先按形状认出来。
            body = _last_json_object(messages)
            if "explanation" in body and "answer" not in body:
                # 措辞检验（L4.7）的提示内容：只有 stem / explanation，没有答案也没有候选。
                # 夹具一律回「没意见」——本类测试钉的不是那条校验。
                return Completion(
                    ok=True, text=json.dumps({"phrases": [], "note": ""}), provider="scripted", model="m"
                )
            if "expected" in body:  # 答案仲裁：提示内容里带标准答案
                self.arbitrations += 1
                return Completion(
                    ok=True,
                    text=json.dumps({"equivalent": self._equivalent, "reason": "夹具结论"}),
                    provider="scripted", model="m",
                )
            payload = payload_of(messages)
            if not payload:  # 往返检验：只有 stem / options，没有 answer
                return Completion(
                    ok=True,
                    text=json.dumps({"answer": self._reader(self.answer, body)}, ensure_ascii=False),
                    provider="scripted", model="m",
                )
            self.answer = str(payload["answer"])
            return Completion(ok=True, text=compliant_body(payload), provider="scripted", model="m")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def _run(self, client, name: str) -> list[dict]:
        """输入：假客户端 + 产出文件名。输出：发布出来的记录。预期：`--judge` 一律不传。"""
        out = self.dir / name
        config = PipelineConfig(
            limit=1, out_path=out, data_path=self.data, seed=7, retries=0, roundtrip=True,
        )
        run_pipeline(config, client=client)
        if not out.exists():
            return []
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]

    def test_equivalent_rewrite_is_published_without_the_judge_switch(self) -> None:
        """没开 `--judge`：读者答的是题干里的料（只是写法不同），仲裁判等价 → 照样发布。"""
        client = self._TripleClient(read_wrong_answer_from_stem)
        rows = self._run(client, "arb_pass.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.arbitrations, 1)
        self.assertIn("LLM 仲裁", rows[0]["roundtrip"]["answer_checked"])

    def test_arbiter_rejection_keeps_the_record_out(self) -> None:
        """读者答的是题干里的料、仲裁判「与标准答案不是一回事」→ 不进产出（进 rejects，不丢）。"""
        client = self._TripleClient(read_wrong_answer_from_stem, equivalent=False)
        self.assertEqual(self._run(client, "arb_reject.jsonl"), [])

    def test_rule_pass_never_calls_the_arbiter(self) -> None:
        """子串比对过了就过：一次仲裁请求都不该多花（优先规则，然后 LLM judge）。"""
        client = self._TripleClient(read_answer_from_stem)
        rows = self._run(client, "arb_rule.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.arbitrations, 0)

    def test_missing_condition_self_report_is_corroborated(self) -> None:
        """读者自报「缺条件」，但机械校验确认题干写全了识别条件 → 按误报处理，照样发布。"""
        client = self._TripleClient(lambda _expected, _body: "缺条件")
        rows = self._run(client, "arb_missing.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertIn("不予否决", rows[0]["roundtrip"]["answer_checked"])


class SemanticAnswerWiringTests(unittest.TestCase):
    """输入：机械校验结论 + 假仲裁客户端。输出：修正过的结论。

    预期：`answer_in_explanation` 只做**快速通过**——子串没过、而且**只有这一条没过**时才问 LLM；
          别的校验也挂了就不花这次配额（救回来照样发布不了）；仲裁缺席维持机械结论（工具问题
          不判数据死刑，也不放行）。
    """

    class _Arbiter:
        """输入：预设正文。输出：假仲裁客户端。预期：记录调用次数，用来钉「什么时候不该问」。"""

        def __init__(self, text: str) -> None:
            self._text = text
            self.calls = 0

        def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
            self.calls += 1
            return Completion(ok=True, text=self._text, provider="scripted", model="m")

    @staticmethod
    def _spec():
        return TaskSpec(task_id="T_test", instruction="i", facts={}, answer="xié")

    @staticmethod
    def _verdict(explanation: str = "「㐑」读作别的音。"):
        """输入：解析。输出：只差 `answer_in_explanation` 一条的机械结论。

        预期：用真的 `verify` 造夹具——手搓 `Verification` 会把 `failures` / `notes` 的配对写错，
              而修正那一支正是按这两份同源文本撤的。
        """
        return verify(
            SemanticAnswerWiringTests._spec(),
            Verbalized(ok=True, stem="「㐑」的读音是什么？", explanation=explanation),
        )

    def test_semantic_pass_lifts_the_substring_failure(self) -> None:
        """解析里没有答案串（子串校验不过），语义核验判「解析说的就是标准答案」→ 撤掉那条失败。"""
        arbiter = self._Arbiter('{"states_answer": true, "stated": "xié", "reason": "换了个说法"}')
        result = _verified_with_semantic_answer(self._verdict(), self._spec(), None, arbiter)
        self.assertTrue(result.answer_in_explanation)
        self.assertEqual(result.tier, "A")
        self.assertEqual(result.failures, ())
        self.assertEqual(result.notes, ())
        self.assertIn("LLM 仲裁", result.semantic_answer)

    def test_semantic_veto_keeps_the_failure(self) -> None:
        """语义核验判「解析没说答案」→ 失败照旧留着，理由落进 `semantic_answer`。"""
        arbiter = self._Arbiter('{"states_answer": false, "stated": "", "reason": "解析没说答案"}')
        result = _verified_with_semantic_answer(self._verdict(), self._spec(), None, arbiter)
        self.assertFalse(result.answer_in_explanation)
        self.assertEqual(result.tier, "C")
        self.assertTrue(result.failures)
        self.assertIn("解析没说答案", result.semantic_answer)

    def test_absent_arbiter_keeps_the_mechanical_verdict(self) -> None:
        """仲裁跑挂了（输出不是 JSON）→ 维持机械结论，理由里留下缺席痕迹。"""
        arbiter = self._Arbiter("这不是 JSON")
        before = self._verdict()
        result = _verified_with_semantic_answer(before, self._spec(), None, arbiter)
        self.assertEqual(result.tier, "C")
        self.assertEqual(result.failures, before.failures)
        self.assertIn("仲裁缺席", result.semantic_answer)

    def test_fast_pass_never_calls_the_arbiter(self) -> None:
        """子串比对过了就不问 LLM：一次配额都不该多花。"""
        arbiter = self._Arbiter('{"states_answer": false, "reason": "夹具乱判"}')
        before = self._verdict("「㐑」读作 xié。")
        self.assertTrue(before.answer_in_explanation)
        result = _verified_with_semantic_answer(before, self._spec(), None, arbiter)
        self.assertEqual(result.semantic_answer, "")
        self.assertEqual(arbiter.calls, 0)

    def test_other_failures_skip_the_semantic_check(self) -> None:
        """别的校验也挂了 → 这一条救回来照样发布不了，不花配额。"""
        arbiter = self._Arbiter('{"states_answer": true, "reason": "夹具乱判"}')
        verdict = verify(
            self._spec(),
            Verbalized(ok=True, stem="What is the reading?", explanation="「㐑」读作别的音。"),
        )
        self.assertTrue(len(verdict.failures) > 1, verdict.failures)
        result = _verified_with_semantic_answer(verdict, self._spec(), None, arbiter)
        self.assertEqual(result.failures, verdict.failures)
        self.assertEqual(arbiter.calls, 0)

    def test_no_client_keeps_the_mechanical_verdict(self) -> None:
        """没接客户端（offline）→ 只有子串比对，结论原样。"""
        before = self._verdict()
        result = _verified_with_semantic_answer(before, self._spec(), None, None)
        self.assertEqual(result.failures, before.failures)
        self.assertEqual(result.semantic_answer, "")


class _SemanticAnswerClient:
    """输入：语义核验的结论。输出：同时扮 L3 与语义核验两个角色的假客户端。

    预期：L3 那一轮返回一份**解析里不写答案串**的规范正文（子串校验必然不过），语义核验那一轮
          返回给定结论。靠提示内容形状分流：语义核验的提示内容带 `explanation`，L3 的带 `facts`。
    """

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict
        self.semantic_calls = 0

    def complete(self, messages, *, temperature: float = 0.6, extra=None) -> Completion:
        body = _last_json_object(messages)
        if "explanation" in body:  # 语义核验：提示内容里只有解析与答案
            self.semantic_calls += 1
            return Completion(ok=True, text=self._verdict, provider="scripted", model="m")
        payload = payload_of(messages)
        data = json.loads(compliant_body(payload))
        data["explanation"] = "这个字读作某个音。"  # 故意不写答案串
        return Completion(ok=True, text=json.dumps(data, ensure_ascii=False), provider="scripted", model="m")


class SemanticAnswerEndToEndTests(unittest.TestCase):
    """输入：解析里不写答案串的假客户端。输出：发不发布。预期：语义核验说了算。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def _run(self, client, name: str) -> list[dict]:
        """输入：假客户端 + 产出文件名。输出：发布出来的记录。预期：不开往返检验，只钉这一条接线。"""
        out = self.dir / name
        run_pipeline(
            PipelineConfig(limit=1, out_path=out, data_path=self.data, seed=7, retries=0),
            client=client,
        )
        if not out.exists():
            return []
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]

    def test_semantic_pass_publishes_with_a_trace(self) -> None:
        """子串校验不过、语义核验判过 → 照样发布，并把仲裁痕迹落进 verification。"""
        client = _SemanticAnswerClient('{"states_answer": true, "stated": "xié", "reason": "换了个说法"}')
        rows = self._run(client, "sem_pass.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.semantic_calls, 1)
        self.assertTrue(rows[0]["verification"]["answer_in_explanation"])
        self.assertIn("LLM 仲裁", rows[0]["verification"]["semantic_answer"])

    def test_semantic_veto_keeps_the_record_out(self) -> None:
        """语义核验判「解析没说答案」→ 不进产出（进 rejects，不丢）。"""
        client = _SemanticAnswerClient('{"states_answer": false, "reason": "解析没说答案"}')
        self.assertEqual(self._run(client, "sem_reject.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
