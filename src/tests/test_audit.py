"""L6 发布门禁（`pipeline/audit.py`）的单元测试：只读产出、零网络、零 LLM。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hanzi_synth.pipeline.audit import (
    audit_record,
    audit_sources,
    collect_sources,
    is_published_record,
    is_published_source,
    write_jsonl,
)


def make_record(**overrides) -> dict:
    """输入：要覆盖的字段。输出：一条「长得像已发布产出」的最小记录。
    预期：字段集与 `_record` 落盘的一致（不含 proof/distractors/requirements），
          足以让 `spec_from_record` 重建 TaskSpec 并跑完 L4。
    """
    row = {
        "task_id": "T_test",
        "ziid": "U0597D",
        "instruction": "问这个字一共有几画。",
        "facts": {"zi": "好"},
        "answer": "6",
        "accepted": ["6"],
        "options": [],
        # 闸门字段：旧 schema 的产出记录没有这几个键，复验会静默降级，
        # 所以夹具按**当前** schema 给全，另有用例专门覆盖「缺字段」那条路。
        "visible_subject": "",
        "must_contain": [],
        "hidden_subject": "",
        "stem_asks_any": [],
        "structure_words": [],
        # 自然语言那一层的判定（往返检验）：机械闸门只比子串，判不了可读性，
        # 所以记录必须自己带上这一份，发布门禁也据此判。
        "roundtrip": {"ok": True, "answer": "6", "is_question": True, "error": ""},
        "stem": "这个字「好」一共有几画？",
        "explanation": "这个字「好」一共有 6 画。",
    }
    row.update(overrides)
    return row


def write_rows(path: Path, rows: list[dict]) -> Path:
    """输入：目标路径 + 记录列表。输出：该路径。预期：每条一行 JSON，测试辅助专用。"""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class AuditRecordTests(unittest.TestCase):
    """`audit_record`：一条记录的判定口径必须与主流程完全一致。"""

    def test_clean_record_is_kept(self) -> None:
        """干净记录：判 A，且一条失败原因都没有。"""
        outcome = audit_record(make_record())
        self.assertTrue(outcome["keep"])
        self.assertEqual(outcome["tier"], "A")
        self.assertEqual(outcome["failures"], [])

    def test_choice_record_without_candidates_is_rejected(self) -> None:
        """选择题题干没列出候选 → 判 C，失败原因里点名缺了哪个候选。"""
        row = make_record(
            answer="㐀",
            accepted=["㐀"],
            options=["㐀", "㐁"],
            facts={"zi": "㐀"},
            stem="请从下列选项中选出笔画最少的字。",
            explanation="「㐀」是笔画最少的。",
        )
        outcome = audit_record(row)
        self.assertFalse(outcome["keep"])
        self.assertEqual(outcome["tier"], "C")
        self.assertTrue(any("options_listed:" in f for f in outcome["failures"]))
        self.assertTrue(any("㐀" in f for f in outcome["failures"]))

    def test_judge_veto_blocks_publication(self) -> None:
        """程序全过但裁判否决 → 不许发布（与 `_publishable` 同口径）。"""
        row = make_record(judge={"ok": False, "reason": "不自足"})
        outcome = audit_record(row)
        self.assertEqual(outcome["tier"], "A")
        self.assertFalse(outcome["keep"])

    def test_roundtrip_veto_blocks_publication(self) -> None:
        """程序全过、裁判没开，但往返检验否决（读者答不出定死的答案）→ 不许发布。

        机械闸门只会做子串 / 集合比对，「这段话读者读得懂吗」不在它的能力范围里；
        那一层只有往返检验能判，所以它的否决必须与裁判否决同等对待。
        """
        row = make_record(
            roundtrip={"ok": False, "answer": "读不出答案", "is_question": True, "error": ""}
        )
        outcome = audit_record(row)
        self.assertEqual(outcome["tier"], "A")
        self.assertFalse(outcome["keep"])

    def test_roundtrip_absence_is_not_a_veto(self) -> None:
        """往返检验**缺席**（请求挂了 / 输出坏了）不算否决：工具问题不判数据死刑。"""
        row = make_record(roundtrip={"ok": False, "error": "往返检验请求失败"})
        outcome = audit_record(row)
        self.assertTrue(outcome["keep"])


class SourceCollectionTests(unittest.TestCase):
    """`is_published_source` / `collect_sources`：输入集合怎么挑出来。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_sidecars_are_not_sources(self) -> None:
        """旁路日志 / 回炉副本 / 去重副本都不是产出源。"""
        for name in (
            "batch.jsonl",
            "batch.attempts.jsonl",
            "batch.rejects.jsonl",
            "batch.rejects.jsonl.replayed.jsonl",
            "batch.dedup.jsonl",
            "batch.dropped.jsonl",
            "batch.jsonl.audit.json",
            "batch.audit.json",
        ):
            expected = name == "batch.jsonl"
            self.assertEqual(is_published_source(Path(name)), expected, name)

    def test_directory_expansion_skips_sidecars(self) -> None:
        """给目录：展开一层 *.jsonl，旁路文件自动剔掉。"""
        write_rows(self.dir / "a.jsonl", [make_record()])
        write_rows(self.dir / "a.rejects.jsonl", [make_record()])
        write_rows(self.dir / "a.attempts.jsonl", [make_record()])
        sources = collect_sources([self.dir], order="path")
        self.assertEqual([p.name for p in sources], ["a.jsonl"])

    def test_output_file_is_excluded(self) -> None:
        """本次的输出文件必须从输入里剔掉，否则第二次跑会吃回上一次的数据集。"""
        out = write_rows(self.dir / "dataset.jsonl", [make_record()])
        write_rows(self.dir / "a.jsonl", [make_record()])
        sources = collect_sources([self.dir], order="path", exclude=[out])
        self.assertEqual([p.name for p in sources], ["a.jsonl"])

    def test_repeated_paths_collapse(self) -> None:
        """同一个文件既给目录又给文件时只留一份。"""
        path = write_rows(self.dir / "a.jsonl", [make_record()])
        sources = collect_sources([self.dir, path], order="path")
        self.assertEqual(len(sources), 1)

    def test_order_path_is_sorted(self) -> None:
        """`order="path"` 按路径字典序，跨机器可复现。"""
        for name in ("b.jsonl", "a.jsonl", "c.jsonl"):
            write_rows(self.dir / name, [make_record()])
        sources = collect_sources([self.dir], order="path")
        self.assertEqual([p.name for p in sources], ["a.jsonl", "b.jsonl", "c.jsonl"])


class AuditSourcesTests(unittest.TestCase):
    """`audit_sources`：整批筛选、收唯一、自噬防护与计数。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_keeps_passers_and_reports_reasons(self) -> None:
        """能过的留下、过不去的进旁路，失败原因按防线名汇总。"""
        good = make_record()
        bad = make_record(
            ziid="U03400",
            answer="㐀",
            accepted=["㐀"],
            options=["㐀", "㐁"],
            facts={"zi": "㐀"},
            stem="请从下列选项中选出笔画最少的字。",
            explanation="「㐀」是笔画最少的。",
        )
        path = write_rows(self.dir / "a.jsonl", [good, bad])
        kept, dropped, report = audit_sources([path])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(report["kept"], 1)
        self.assertEqual(report["dropped"], 1)
        self.assertIn("options_listed", report["dropped_by_defense"])
        self.assertEqual(dropped[0]["source"], "a.jsonl")

    def test_unique_by_task_zi_keeps_first(self) -> None:
        """`unique_by="task-zi"`：同一道题只留先出现的那条（文件顺序已由调用方定好）。"""
        first = make_record(stem="这个字「好」一共有几画？", explanation="这个字「好」一共有 6 画。")
        second = make_record(stem="汉字「好」共有几画？", explanation="汉字「好」共有 6 画。")
        path_a = write_rows(self.dir / "a.jsonl", [first])
        path_b = write_rows(self.dir / "b.jsonl", [second])
        kept, _, report = audit_sources([path_a, path_b], unique_by="task-zi")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["stem"], first["stem"])
        self.assertEqual(report["fact_duplicates"], 1)
        self.assertEqual(report["unique_facts"], 1)

    def test_duplicates_are_kept_by_default(self) -> None:
        """默认不丢数据：同一道题重复出现也全收，只把重复量报出来。"""
        path_a = write_rows(self.dir / "a.jsonl", [make_record()])
        path_b = write_rows(self.dir / "b.jsonl", [make_record()])
        kept, _, report = audit_sources([path_a, path_b])
        self.assertEqual(len(kept), 2)
        self.assertEqual(report["fact_duplicates"], 1)

    def test_already_assembled_rows_are_not_rejudged(self) -> None:
        """带 `audit` 块的行是上一轮组装出的数据集，不再当产出源（防自我增殖）。"""
        assembled = make_record(audit={"source": "a.jsonl", "recheck": "full"})
        path = write_rows(self.dir / "dataset.jsonl", [assembled])
        kept, dropped, report = audit_sources([path])
        self.assertEqual(kept, [])
        self.assertEqual(dropped, [])
        self.assertEqual(report["already_assembled"], 1)
        self.assertEqual(report["already_assembled_by_source"], {"dataset.jsonl": 1})

    def test_offline_payloads_are_not_records(self) -> None:
        """`dry-run` 的离线载荷（没有 stem/explanation）单独计数，不算失败记录。"""
        payload = {"task_id": "T_test", "ziid": "U0597D", "instruction": "i", "facts": {"zi": "好"}, "answer": "6"}
        self.assertFalse(is_published_record(payload))
        path = write_rows(self.dir / "dry.jsonl", [payload, make_record()])
        kept, dropped, report = audit_sources([path])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, [])
        self.assertEqual(report["non_records"], 1)
        self.assertEqual(report["total_records"], 1)

    def test_unknown_unique_mode_raises(self) -> None:
        """收唯一口径写错当场报错，不静默按默认跑。"""
        with self.assertRaises(ValueError):
            audit_sources([], unique_by="whatever")

    def test_missing_file_is_not_fatal(self) -> None:
        """文件不存在：如实记 0 条，不抛异常。"""
        kept, dropped, report = audit_sources([self.dir / "nope.jsonl"])
        self.assertEqual((kept, dropped), ([], []))
        self.assertEqual(report["total_records"], 0)


class DatasetRowTests(unittest.TestCase):
    """数据集行的字段：`verification` 必须与它自己的文本对得上。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_verification_is_recomputed(self) -> None:
        """源记录里那份 `verification` 是旧代码的结论，落盘时按本行文本重算。"""
        row = make_record(verification={"tier": "C", "failures": ["旧的结论"]})
        path = write_rows(self.dir / "a.jsonl", [row])
        kept, _, _ = audit_sources([path])
        self.assertEqual(kept[0]["verification"]["tier"], "A")
        self.assertEqual(kept[0]["verification"]["failures"], [])
        self.assertEqual(kept[0]["audit"], {"source": "a.jsonl", "recheck": "full"})

    def test_old_schema_record_is_not_published(self) -> None:
        """旧 schema（缺闸门字段）**不发布**：降级复验对「这条合不合格」没有证明力。

        实测旧格式记录里躺着「这个字是什么？」这种读者无从作答的废题——缺 `visible_subject`
        时「题干有没有写出那个字」这条判据整条空转，降级复验照样判 A。查得没那么严 ≠ 查过了，
        所以这类记录进旁路文件、原因记 `unverifiable`（当前管线重新合成即可）。
        """
        row = make_record()
        for field in ("facts", "visible_subject", "must_contain", "hidden_subject", "stem_asks_any", "structure_words"):
            row.pop(field)
        path = write_rows(self.dir / "old.jsonl", [row])
        kept, dropped, report = audit_sources([path])
        self.assertEqual(kept, [])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["tier"], "unverifiable")
        self.assertIn("缺闸门字段", dropped[0]["failures"][0])
        self.assertEqual(report["recheck_partial"], 1)
        self.assertEqual(report["recheck_partial_dropped"], 1)
        self.assertEqual(report["recheck_partial_by_source"], {"old.jsonl": 1})

    def test_record_without_llm_verdict_is_not_published(self) -> None:
        """记录里没有往返检验的判定（跑的时候关了 / 旧 schema）→ 不发布。

        只留机械防线的结论，数据集对「这段话读得通」就没有证据；而「没查」不能当「查过了」。
        """
        row = make_record(roundtrip=None)
        path = write_rows(self.dir / "no_llm.jsonl", [row])
        kept, dropped, report = audit_sources([path])
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["tier"], "unverifiable")
        self.assertIn("roundtrip", dropped[0]["failures"][0])
        self.assertEqual(report["recheck_partial"], 1)

    def test_other_fields_are_carried_over(self) -> None:
        """除文本与判定外，源记录字段原样带出（数据集要能逐字段追回源记录）。"""
        row = make_record(provider="local-2b", model="m", provenance={"seed_ziid": "U0597D"})
        path = write_rows(self.dir / "a.jsonl", [row])
        kept, _, _ = audit_sources([path])
        self.assertEqual(kept[0]["provider"], "local-2b")
        self.assertEqual(kept[0]["provenance"], {"seed_ziid": "U0597D"})
        self.assertEqual(kept[0]["ziid"], "U0597D")


class WriteJsonlTests(unittest.TestCase):
    """`write_jsonl`：数据集是纯文本交付物，行尾与编码不能带惊喜。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_lf_no_bom_and_roundtrip(self) -> None:
        """LF 行尾、UTF-8 无 BOM、汉字不转义。"""
        target = self.dir / "sub" / "dataset.jsonl"
        count = write_jsonl(target, [make_record()])
        self.assertEqual(count, 1)
        raw = target.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r\n", raw)
        self.assertIn("好".encode("utf-8"), raw)
        self.assertEqual(json.loads(raw.decode("utf-8").strip())["task_id"], "T_test")


if __name__ == "__main__":
    unittest.main()