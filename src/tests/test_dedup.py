"""L6 去重与污染检测的单元测试（不联网、不读真实数据）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hanzi_synth.pipeline.dedup import (
    CONTAMINATION_NGRAM,
    DedupIndex,
    _jaccard,
    _shingles,
    contamination,
    contamination_file,
    dedup_file,
    load_jsonl,
    read_corpus,
    write_kept,
    write_report,
)

# 这两句只差最后一个字：4-gram 层面 30/32 重合，超过默认 0.9 阈值
LONG_A = "请回答下面这道关于汉字的问题：这个「好」字的总笔画数到底是多少呢"
LONG_B = "请回答下面这道关于汉字的问题：这个「好」字的总笔画数到底是多少呀"
OTHER = "「好」字的读音应该怎么标注才算正确？"


class ShingleTests(unittest.TestCase):
    def test_normalizes_listing_separators(self) -> None:
        """「E3 90 80」与「E3、90、80」归一后是同一串，指纹必须一致。"""
        self.assertEqual(_shingles("E3 90 80", 3), _shingles("E3、90、80", 3))

    def test_empty_and_short_text(self) -> None:
        """空文本给空集；短于 n 的文本给整串单元素集（否则短句全部互相判不同）。"""
        self.assertEqual(_shingles("", 4), frozenset())
        self.assertEqual(_shingles("好", 4), frozenset({"好"}))

    def test_jaccard_edges(self) -> None:
        """Jaccard：任一为空返回 0.0，完全相同返回 1.0。"""
        self.assertEqual(_jaccard(frozenset(), frozenset({"a"})), 0.0)
        self.assertEqual(_jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})), 1.0)
        self.assertAlmostEqual(_jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})), 1 / 3)


class DedupIndexTests(unittest.TestCase):
    def test_detects_near_duplicate(self) -> None:
        """近似重复：第二条与第一条高度重合时判重，并指出撞上的是第 0 条。"""
        index = DedupIndex()
        duplicate, _score, match = index.add(LONG_A)
        self.assertFalse(duplicate)
        self.assertIsNone(match)
        duplicate, score, match = index.add(LONG_B)
        self.assertTrue(duplicate)
        self.assertGreaterEqual(score, index.threshold)
        self.assertEqual(match, 0)

    def test_unrelated_text_is_not_duplicate(self) -> None:
        """换了个问法（问读音）不该被判成重复，否则有效多样性会被误杀。"""
        index = DedupIndex()
        index.add(LONG_A)
        duplicate, _score, _match = index.add(OTHER)
        self.assertFalse(duplicate)
        self.assertEqual(len(index), 2)

    def test_exact_duplicate_goes_through_exact_layer(self) -> None:
        """逐字相同的复制品由精确层兜住，相似度必须是 1.0（高频模板语料里这层最关键）。"""
        index = DedupIndex()
        index.add(LONG_A)
        duplicate, score, match = index.add(LONG_A)
        self.assertTrue(duplicate)
        self.assertEqual(score, 1.0)
        self.assertEqual(match, 0)

    def test_threshold_is_configurable(self) -> None:
        """阈值是参数不是常量：1.0 时只有逐字相同才判重。"""
        index = DedupIndex(threshold=1.0)
        index.add(LONG_A)
        self.assertFalse(index.add(LONG_B)[0])
        self.assertTrue(index.add(LONG_A)[0])

    def test_invalid_arguments_raise(self) -> None:
        """非法参数当场报错，不静默降级。"""
        with self.assertRaises(ValueError):
            DedupIndex(ngram=0)
        with self.assertRaises(ValueError):
            DedupIndex(threshold=0.0)
        with self.assertRaises(ValueError):
            DedupIndex(candidate_cap=0)
        with self.assertRaises(ValueError):
            DedupIndex(gram_budget=0)

    def test_results_are_reproducible(self) -> None:
        """候选按 (公共 gram 数降序, 下标升序) 排序，结果与哈希种子无关。"""
        first = DedupIndex()
        second = DedupIndex()
        for text in (LONG_A, LONG_B, OTHER, LONG_A):
            self.assertEqual(first.add(text)[:2], second.add(text)[:2])


class LoadJsonlTests(unittest.TestCase):
    def test_missing_file_is_empty_not_fatal(self) -> None:
        """文件不存在返回空，旁路工具不该因为缺文件崩掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            rows, skipped = load_jsonl(Path(tmp) / "nope.jsonl")
            self.assertEqual((rows, skipped), ([], 0))

    def test_bad_lines_are_counted(self) -> None:
        """坏行与空行跳过并计数，好行照常解析。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text('{"a": 1}\n\nnot json\n[1, 2]\n{"a": 2}\n', encoding="utf-8")
            rows, skipped = load_jsonl(path)
            self.assertEqual([row["a"] for row in rows], [1, 2])
            self.assertEqual(skipped, 2)


def write_rows(path: Path, rows: list[dict]) -> None:
    """输入：目标路径与记录。输出：无。预期：写 UTF-8 jsonl，供去重测试用。"""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class DedupFileTests(unittest.TestCase):
    def test_keeps_first_and_groups_duplicates(self) -> None:
        """首条保留，其后重复归到同一组，报告里给出相似度与行号。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_rows(
                path,
                [
                    {"task_id": "T1", "ziid": "U1", "stem": LONG_A},
                    {"task_id": "T1", "ziid": "U2", "stem": LONG_B},
                    {"task_id": "T4", "ziid": "U3", "stem": OTHER},
                ],
            )
            report, kept = dedup_file(path)
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["unique"], 2)
            self.assertEqual(report["duplicates"], 1)
            self.assertEqual([row["ziid"] for row in kept], ["U1", "U3"])
            self.assertEqual(len(report["groups"]), 1)
            group = report["groups"][0]
            self.assertEqual(group["kept"]["line"], 1)
            self.assertEqual(group["dropped"][0]["ziid"], "U2")
            self.assertEqual(group["dropped"][0]["line"], 2)

    def test_original_file_is_untouched(self) -> None:
        """只读：去重绝不改写原产出（要替换得由主人显式决定）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_rows(path, [{"task_id": "T1", "ziid": "U1", "stem": LONG_A}])
            before = path.read_text(encoding="utf-8")
            dedup_file(path)
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_chained_duplicates_share_one_group(self) -> None:
        """链式重复（A≈B≈C）必须归到同一组。
        回归点：`DedupIndex` 返回的是索引内部下标，而组号是保留记录的下标，
        有重复行之后两者会错位（曾直接 KeyError 崩掉）。
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chain.jsonl"
            write_rows(
                path,
                [
                    {"task_id": "T1", "ziid": "U1", "stem": LONG_A},
                    {"task_id": "T1", "ziid": "U2", "stem": LONG_B},
                    {"task_id": "T1", "ziid": "U3", "stem": LONG_A[:-1] + "嘛"},
                ],
            )
            report, kept = dedup_file(path)
            self.assertEqual(report["duplicates"], 2)
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(report["groups"]), 1)
            self.assertEqual([row["ziid"] for row in report["groups"][0]["dropped"]], ["U2", "U3"])

    def test_template_heavy_corpus_does_not_over_merge(self) -> None:
        """整批题共用同一模板、只换一个字：不能判成重复。
        回归点：高频 gram（模板套话）如果不封顶，每条都要扫全量倒排表，整体退化成 O(N²)。
        """
        rows = [
            {
                "task_id": "T1",
                "ziid": f"U{index:05d}",
                "stem": f"请回答下面这道关于汉字的问题：这个「{chr(0x4E00 + index)}」字的总笔画数到底是多少呢",
            }
            for index in range(2000)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tpl.jsonl"
            write_rows(path, rows)
            report, kept = dedup_file(path)
            self.assertEqual(report["unique"], 2000)
            self.assertEqual(report["duplicates"], 0)
            self.assertEqual(len(kept), 2000)

    def test_empty_key_and_bad_lines_are_reported(self) -> None:
        """缺 stem / 空 stem 单独计数；坏行计入 skipped。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(
                '{"task_id": "T1", "ziid": "U1", "stem": "  "}\n{"task_id": "T1", "ziid": "U2"}\nbroken\n',
                encoding="utf-8",
            )
            report, kept = dedup_file(path)
            self.assertEqual(report["empty"], 2)
            self.assertEqual(report["skipped"], 1)
            self.assertEqual(kept, [])

    def test_write_report_and_kept(self) -> None:
        """旁路文件写在 <out>.dedup.json / <out>.dedup.jsonl，原文件不动。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_rows(path, [{"task_id": "T1", "ziid": "U1", "stem": LONG_A}])
            before = path.read_text(encoding="utf-8")
            report, kept = dedup_file(path)
            report_path = write_report(path, report)
            kept_path = write_kept(path, kept)
            self.assertEqual(report_path.name, "out.jsonl.dedup.json")
            self.assertEqual(kept_path.name, "out.jsonl.dedup.jsonl")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["unique"], 1)
            self.assertEqual(len(kept_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(path.read_text(encoding="utf-8"), before)


class CorpusTests(unittest.TestCase):
    def test_read_corpus_jsonl_extracts_meaning_not_ziid(self) -> None:
        """chars.jsonl 形态：抽到 meanings.mean_h，绝不把 ziid 当成语料。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chars.jsonl"
            row = {
                "ziid": "U00001",
                "property": {"zi": "好"},
                "meanings": {"mean_h": ["拼音hao3<br>美好的意思。"]},
            }
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            texts = read_corpus(path)
            self.assertEqual(len(texts), 1)
            self.assertIn("美好的意思", texts[0])
            self.assertNotIn("U00001", texts[0])

    def test_read_corpus_text_file(self) -> None:
        """纯文本按行读，空行跳过。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.txt"
            path.write_text("第一行\n\n第二行\n", encoding="utf-8")
            self.assertEqual(read_corpus(path), ["第一行", "第二行"])

    def test_without_corpus_is_honest(self) -> None:
        """没给语料就如实报 checked=False，不假装查过。"""
        report = contamination(["题干一"], [])
        self.assertFalse(report["checked"])
        self.assertIn("reason", report)

    def test_flags_long_copy(self) -> None:
        """题干里连续照抄 13 字以上即判污染，并给出命中样本。"""
        source = "这是一个很长很长的释义原文用来测试污染检测能不能抓到连续照抄"
        stem = f"请根据下面这段内容作答：{source}，对吗？"
        report = contamination([stem, OTHER], [source])
        self.assertTrue(report["checked"])
        self.assertEqual(report["contaminated"], 1)
        self.assertEqual(report["hits"][0]["index"], 0)
        self.assertGreaterEqual(len(report["hits"][0]["sample"]), CONTAMINATION_NGRAM)

    def test_rewrite_is_not_contamination(self) -> None:
        """改写过的不算污染——否则正常措辞会被误判成抄原文。"""
        source = "这是一个很长很长的释义原文用来测试污染检测能不能抓到连续照抄"
        stem = "这个字表示美好的意思，请说出它的读音。"
        report = contamination([stem], [source])
        self.assertEqual(report["contaminated"], 0)

    def test_truncation_is_reported(self) -> None:
        """语料超过扫描上限就截断，并在报告里如实标记，绝不假装查全了。"""
        source = "这是一个很长很长的释义原文用来测试污染检测能不能抓到连续照抄"
        report = contamination([f"请重述：{source}"], [source, "第二段语料"], max_texts=1)
        self.assertTrue(report["truncated"])
        self.assertEqual(report["processed_texts"], 1)
        self.assertEqual(report["contaminated"], 1)

    def test_contamination_file_reads_output_and_corpus(self) -> None:
        """contamination_file 从产出读题干、从语料文件读原文，并回填两个路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            corpus = Path(tmp) / "corpus.txt"
            source = "这是一个很长很长的释义原文用来测试污染检测能不能抓到连续照抄"
            write_rows(out, [{"task_id": "T13", "ziid": "U1", "stem": f"请重述：{source}"}])
            corpus.write_text(source + "\n", encoding="utf-8")
            report = contamination_file(out, corpus)
            self.assertEqual(report["out_path"], str(out))
            self.assertEqual(report["corpus_path"], str(corpus))
            self.assertEqual(report["contaminated"], 1)

    def test_contamination_file_without_corpus(self) -> None:
        """不传语料路径时仍返回报告（checked=False），不抛错。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            write_rows(out, [{"task_id": "T1", "ziid": "U1", "stem": OTHER}])
            report = contamination_file(out, None)
            self.assertFalse(report["checked"])
            self.assertIsNone(report["corpus_path"])


if __name__ == "__main__":
    unittest.main()
