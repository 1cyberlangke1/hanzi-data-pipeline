"""CLI 入口的单元测试（只跑 list / dry-run / 错误路径，不发真实请求）。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from hanzi_synth.cli import _exclude_paths, main


def write_chars(path: Path) -> None:
    """输入：目标路径。输出：无。预期：写两条有 IDS 的记录。"""
    records = [
        {"ziid": "U00001", "property": {"ziid": "U00001", "zi": "\u597d", "ids": "\u2ff0\u5973\u5b50", "stroke_total": 6}, "meanings": {}, "variants": []},
        {"ziid": "U00002", "property": {"ziid": "U00002", "zi": "\u4e00", "ids": "\u2ff1\u4e00\u4e8c", "stroke_total": 3}, "meanings": {}, "variants": []},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.data = self.dir / "chars.jsonl"
        write_chars(self.data)

    def test_list_prints_task_ids(self) -> None:
        """list 子命令打印已注册题型并返回 0。"""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["list"])
        self.assertEqual(code, 0)
        self.assertIn("T13_ids_reverse", buffer.getvalue())

    def test_dry_run_writes_specs(self) -> None:
        """dry-run 不联网，只把 TaskSpec 写盘。"""
        out = self.dir / "dry.jsonl"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["dry-run", "--limit", "2", "--data", str(self.data), "--out", str(out)])
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn("answer", rows[0])

    def test_unknown_task_id_exits(self) -> None:
        """未知题型立刻报错退出，不静默忽略。"""
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["dry-run", "--limit", "1", "--tasks", "T999", "--data", str(self.data)])

    def test_missing_data_returns_exit_code_2(self) -> None:
        """数据文件缺失时返回退出码 2 并打印人话。"""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main(["dry-run", "--limit", "1", "--data", str(self.dir / "nope.jsonl")])
        self.assertEqual(code, 2)

    def test_task_filter_is_applied(self) -> None:
        """题型白名单生效。"""
        out = self.dir / "one.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["dry-run", "--limit", "2", "--tasks", "T1_stroke_total", "--data", str(self.data), "--out", str(out)])
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(row["task_id"] == "T1_stroke_total" for row in rows))

    def test_difficulty_flags_reach_the_specs(self) -> None:
        """难度旋钮必须能从命令行调——否则又是「配置里有、实际调不动」的死开关。"""
        out = self.dir / "hard.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(
                [
                    "dry-run", "--limit", "1", "--tasks", "T1_stroke_total",
                    "--hops", "2", "--constraints", "3", "--distractor-depth", "far",
                    "--pool", "10", "--data", str(self.data), "--out", str(out),
                ]
            )
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)

    def test_invalid_depth_is_rejected_by_argparse(self) -> None:
        """非法档位由 argparse 挡下，不进管线。"""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["dry-run", "--limit", "1", "--distractor-depth", "ultra", "--data", str(self.data)])

    def test_overwrite_flag_backs_up_old_output(self) -> None:
        """--overwrite 先把旧产出备份成 .bak-<时间戳>，再重跑。"""
        out = self.dir / "ow.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["dry-run", "--limit", "2", "--data", str(self.data), "--out", str(out)])
        before = out.read_text(encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["dry-run", "--limit", "2", "--overwrite", "--data", str(self.data), "--out", str(out)])
        self.assertEqual(code, 0)
        backups = list(self.dir.glob("ow.jsonl.bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), before)

    def test_coverage_subcommand_is_read_only(self) -> None:
        """coverage 子命令只读产出文件，打印池子覆盖情况，不改动任何文件。"""
        out = self.dir / "cov.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["dry-run", "--limit", "2", "--data", str(self.data), "--out", str(out)])
        before = out.read_text(encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["coverage", "--out", str(out), "--data", str(self.data)])
        self.assertEqual(code, 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["pool"], 2)
        self.assertGreaterEqual(report["covered"], 1)
        self.assertEqual(out.read_text(encoding="utf-8"), before)

    def test_min_coverage_guard_returns_exit_code_1(self) -> None:
        """覆盖率没到下限量 -> 退出码 1 并报还差多少，不静默放行。"""
        out = self.dir / "mc.jsonl"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main(
                [
                    "dry-run", "--limit", "1", "--min-coverage", "0.9",
                    "--data", str(self.data), "--out", str(out),
                ]
            )
        self.assertEqual(code, 1)
        self.assertIn("覆盖率未达标", buffer.getvalue())

    def test_dedup_subcommand_reports_and_keeps_original(self) -> None:
        """dedup 只读原产出：报告写旁路文件，--write-kept 另存副本，原文件一个字节不动。"""
        out = self.dir / "d.jsonl"
        rows = [
            {"task_id": "T1", "ziid": "U1", "stem": "请回答下面这道关于汉字的问题：这个「好」字的总笔画数到底是多少呢"},
            {"task_id": "T1", "ziid": "U2", "stem": "请回答下面这道关于汉字的问题：这个「好」字的总笔画数到底是多少呀"},
        ]
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        before = out.read_text(encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            code = main(["dedup", "--out", str(out), "--write-kept"])
        self.assertEqual(code, 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["duplicates"], 1)
        self.assertEqual(out.read_text(encoding="utf-8"), before)
        self.assertTrue((self.dir / "d.jsonl.dedup.json").exists())
        self.assertTrue(Path(report["kept_path"]).exists())

    def test_dedup_subcommand_missing_output_returns_2(self) -> None:
        """产出文件不存在时 dedup 报错退出 2，不静默产出空报告。"""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main(["dedup", "--out", str(self.dir / "nope.jsonl")])
        self.assertEqual(code, 2)

    def test_contamination_subcommand_is_honest_without_corpus(self) -> None:
        """不给语料 -> checked=False、退出码 1（如实报「没查」，不假装干净）；给了语料 -> 0。"""
        out = self.dir / "c.jsonl"
        source = "这是一个很长很长的释义原文用来测试污染检测能不能抓到连续照抄"
        with out.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": "T13", "ziid": "U1", "stem": source}, ensure_ascii=False) + "\n")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["contamination", "--out", str(out)]), 1)
        self.assertFalse(json.loads(buffer.getvalue())["checked"])
        corpus = self.dir / "corpus.txt"
        corpus.write_text(source + "\n", encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["contamination", "--out", str(out), "--corpus", str(corpus)]), 0)
        self.assertEqual(json.loads(buffer.getvalue())["contaminated"], 1)

    def test_difficulty_subcommand_without_responses_returns_1(self) -> None:
        """没有作答数据 -> calibrated=False、退出码 1，结果仍写盘（不假装标定过）。"""
        responses = self.dir / "responses.jsonl"
        responses.write_text("", encoding="utf-8")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            code = main(["difficulty", "--responses", str(responses)])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(buffer.getvalue())["calibrated"])
        self.assertTrue((self.dir / "responses.jsonl.difficulty.json").exists())

    def test_difficulty_subcommand_missing_file_returns_2(self) -> None:
        """作答文件不存在时返回 2 并打印人话。"""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main(["difficulty", "--responses", str(self.dir / "nope.jsonl")])
        self.assertEqual(code, 2)

    def test_audit_filters_产出_and_writes_sidecars(self) -> None:
        """audit：只留过门禁的记录，剔除写旁路，报告写 .audit.json。"""
        # 闸门字段必须给全：门禁只发布「可全量复验」的记录（缺字段 = 复验降级 = 不发布）。
        gates = {
            "visible_subject": "", "must_contain": [], "hidden_subject": "",
            "stem_asks_any": [], "structure_words": [],
            # 自然语言那一层的判定：门禁发布时必查（机械闸门只比子串，判不了可读性）
            "roundtrip": {"ok": True, "answer": "6", "is_question": True, "error": ""},
        }
        good = {
            "task_id": "T_test", "ziid": "U0597D", "instruction": "i", "facts": {"zi": "好"},
            "answer": "6", "accepted": ["6"], "options": [], **gates,
            "stem": "这个字「好」一共有几画？", "explanation": "这个字「好」一共有 6 画。",
        }
        bad = dict(good, ziid="U03400", answer="㐀", accepted=["㐀"], options=["㐀", "㐁"],
                   facts={"zi": "㐀"}, stem="请从下列选项中选出笔画最少的字。",
                   explanation="「㐀」是笔画最少的。")
        source = self.dir / "published.jsonl"
        with source.open("w", encoding="utf-8", newline="\n") as handle:
            for row in (good, bad):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        out = self.dir / "dataset.jsonl"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
            code = main(["audit", "--in", str(source), "--out", str(out)])
        self.assertEqual(code, 0)
        report = json.loads(buffer.getvalue())
        self.assertEqual((report["kept"], report["dropped"]), (1, 1))
        self.assertEqual(report["dropped_by_defense"], {"options_listed": 1})
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["stem"], good["stem"])
        self.assertEqual(rows[0]["audit"]["source"], "published.jsonl")
        self.assertEqual(rows[0]["verification"]["tier"], "A")
        dropped = [json.loads(line) for line in out.with_suffix(".dropped.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(dropped[0]["ziid"], "U03400")
        self.assertTrue(out.with_suffix(".audit.json").exists())

    def test_audit_without_sources_returns_2(self) -> None:
        """没有任何产出文件时返回 2，不假装产出了一个空数据集。"""
        empty = self.dir / "empty"
        empty.mkdir()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = main(["audit", "--in", str(empty), "--out", str(self.dir / "ds.jsonl")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

class ExcludePathsCliTests(unittest.TestCase):
    """`--exclude` 的参数解析：目录展开一层 jsonl、旁路日志剔掉、路径不存在直接报错。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_directory_expands_and_drops_sidecar_logs(self) -> None:
        """输入：一个目录。输出：里面的一层 `*.jsonl`，旁路日志（rejects / attempts 等）被剔掉。"""
        (self.dir / "a.jsonl").write_text("{}\n", encoding="utf-8")
        (self.dir / "b.jsonl").write_text("{}\n", encoding="utf-8")
        (self.dir / "b.rejects.jsonl").write_text("{}\n", encoding="utf-8")
        (self.dir / "c.attempts.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertEqual(
            [p.name for p in _exclude_paths([str(self.dir)])], ["a.jsonl", "b.jsonl"]
        )

    def test_missing_path_exits_instead_of_silently_ignoring(self) -> None:
        """路径打错一个字母就报错退出——静默失效会让人以为已经排除了。"""
        with self.assertRaises(SystemExit):
            _exclude_paths([str(self.dir / "nope.jsonl")])

    def test_none_and_empty_give_empty_tuple(self) -> None:
        """没给 `--exclude` 时是空元组，不影响原有行为。"""
        self.assertEqual(_exclude_paths(None), ())
        self.assertEqual(_exclude_paths([]), ())
