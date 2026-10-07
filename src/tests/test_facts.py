"""L0 事实底座（加载 + 准入 + 记录属性）的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hanzi_synth.facts.loader import DataError, iter_characters, load_characters
from hanzi_synth.facts.records import clean_meaning_block


def write_jsonl(path: Path, records: list[dict]) -> None:
    """输入：目标路径与记录列表。输出：无。预期：UTF-8 逐行写入，中文不转义。"""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_record(ziid: str, zi: str, ids: str | None, meanings: dict | None = None) -> dict:
    """输入：字 ID / 字 / IDS / 释义。输出：一条最小可用的 chars.jsonl 记录。"""
    prop: dict = {"ziid": ziid, "zi": zi, "stroke_total": 6, "radical": "\u5973", "structure": 1}
    if ids is not None:
        prop["ids"] = ids
    return {"ziid": ziid, "property": prop, "meanings": meanings or {}, "variants": []}


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "chars.jsonl"

    def test_keeps_records_with_usable_ids(self) -> None:
        """有合法 IDS 的字进池，多候选原样保留。"""
        write_jsonl(self.path, [make_record("U03406", "\u3406", "\u2ff0\u5973\u5b50(G),\u2ff1\u597d\u4e00(H)")])
        records = load_characters(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].ids_candidates), 2)

    def test_skips_records_without_ids(self) -> None:
        """没有 ids 字段、或只有笔画记法的字，一律不进池。"""
        write_jsonl(self.path, [
            make_record("Z00001", "", None),
            make_record("U04E00", "\u4e00", "#(H)(G),{\u4e00}#(T)(t)"),
            make_record("U0597D", "\u597d", "\u2ff0\u5973\u5b50"),
        ])
        records = load_characters(self.path)
        self.assertEqual([r.ziid for r in records], ["U0597D"])

    def test_limit_counts_after_filtering(self) -> None:
        """limit 作用在过滤之后，不会被被丢弃的记录占掉名额。"""
        write_jsonl(self.path, [
            make_record("Z00001", "", None),
            make_record("U04E00", "\u4e00", "\u2ff0\u5973\u5b50"),
            make_record("U04E01", "\u4e01", "\u2ff1\u4e00\u4e8c"),
        ])
        self.assertEqual(len(load_characters(self.path, limit=1)), 1)

    def test_missing_file_raises_data_error(self) -> None:
        """数据文件缺失时报人话，而不是抛 FileNotFoundError。"""
        with self.assertRaises(DataError):
            list(iter_characters(Path(self._tmp.name) / "nope.jsonl"))

    def test_bad_json_reports_line_number(self) -> None:
        """坏行必须报出行号，方便定位。"""
        self.path.write_text('{"ziid": "x"}\nnot json\n', encoding="utf-8")
        with self.assertRaises(DataError) as ctx:
            list(iter_characters(self.path))
        self.assertIn(":2", str(ctx.exception))


class RecordTests(unittest.TestCase):
    def _load_one(self, ziid: str, meanings: dict | None = None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "chars.jsonl"
        write_jsonl(path, [make_record(ziid, "\u597d", "\u2ff0\u5973\u5b50", meanings)])
        return load_characters(path)[0]

    def test_group_and_codepoint(self) -> None:
        """U 组有码位，Z/P 组没有。"""
        self.assertTrue(self._load_one("U0597D").has_codepoint)
        self.assertFalse(self._load_one("Z00001").has_codepoint)
        self.assertEqual(self._load_one("P00001").group, "P")

    def test_meaning_blocks_skips_empty(self) -> None:
        """空释义块被丢掉，只留非空文本。"""
        record = self._load_one("U0597D", {"mean_h": ["\u597d\u7684\u610f\u601d", ""], "mean_z": [], "mean_k": ["\u53e4\u5b57"]})
        self.assertEqual(record.meaning_blocks(), ("\u597d\u7684\u610f\u601d", "\u53e4\u5b57"))


class CleanMeaningBlockTests(unittest.TestCase):
    def test_drops_layout_and_dictionary_markers(self) -> None:
        """排版残片、读音标注、互见条目都不是释义内容：逐段删掉，剩下的分点保留成换行。"""
        self.assertEqual(clean_meaning_block("ㄨㄩ拼音yi1<br>①归依。<br>②同【依】。"), "①归依。")

    def test_headword_segment_is_dropped(self) -> None:
        """`[～毒]` 这类词头引用讲的是**词**的意思（`～` 就是本字本身），本字单用没有这层意思：
        删掉词头后剩下的「印度的古译名」无所指，模型会自己编个主语接上去，所以整段丢弃。"""
        self.assertEqual(clean_meaning_block("拼音xie2<br>[～毒]印度的古译名。"), "")

    def test_plain_meaning_is_kept(self) -> None:
        """没有标记的释义原样保留。"""
        self.assertEqual(clean_meaning_block("表示美好善良的意思。"), "表示美好善良的意思。")

    def test_leading_reading_marker_is_stripped_not_the_whole_segment(self) -> None:
        """段首读音标注只剥标记、留下同段剩下的释义：「读音gul。」是朝鲜吏读注音，
        而「音译字。」是本字的释义，两者挤在同一段里，整段丢掉会白扔释义。"""
        self.assertEqual(clean_meaning_block("读音gul。音译字。"), "音译字。")


    def test_layout_only_block_yields_empty(self) -> None:
        """只有排版残片的块清洗后是空串——调用方据此判断「这块没有释义」（实测大量字如此）。"""
        self.assertEqual(clean_meaning_block("<br>"), "")


if __name__ == "__main__":
    unittest.main()
