"""L0 事实底座（加载 + 准入 + 记录属性）的单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hanzi_synth.facts.gloss import gloss_items
from hanzi_synth.facts.loader import DataError, iter_characters, load_characters
from hanzi_synth.facts.records import clean_meaning_block
from tests.support import make_character


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

    def test_skips_records_without_a_writable_glyph(self) -> None:
        """没有字形（`zi` 为空）的记录不进池：这种记录一条题都出不了。

        预期：字海内部编号记录（ziid 形如 `Z31607`，无 Unicode 码位）常常有完整 IDS，但**写不出
              这个字本身**，而任何一道题都得先把它写出来才能问。放进来只会渲染出「从出发，取上边
              的部件…」这种缺主语残句（全量 739,005 条里实测 10,917 条）。
        """
        write_jsonl(self.path, [
            make_record("Z31607", "", "\u2ff1\u5927\u516b"),
            make_record("U04ECB", "\u4ecb", "\u2ff1\u5927\u516b"),
        ])
        records = load_characters(self.path)
        self.assertEqual([r.ziid for r in records], ["U04ECB"])

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
        """排版残片、读音标注、互见条目、分点序号都不是释义内容：逐段删掉，正文保留成换行。"""
        self.assertEqual(clean_meaning_block("ㄨㄩ拼音yi1<br>①归依。<br>②同【依】。"), "归依。")

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


    def test_headword_segment_with_trailing_tilde_is_dropped(self) -> None:
        """词头在后的写法（`[三～]姓氏`）与在前的（`[～毒]…`）是同一种东西：整段丢弃。

        只认前者时后者会漏进题干——实测 T13 因此漏出 `释义是「[三～]姓氏」` 这种词典记法。
        """
        self.assertEqual(clean_meaning_block("拼音san1<br>[三～]姓氏。"), "")

    def test_see_also_segment_is_dropped(self) -> None:
        """「见《直音篇》」「见汉语方言大词典」「见「偛」」是**互见条目**：讲的是别处怎么说，
        不是本字单用的释义。留着它，出处就会被写进产出（实测 3,401 条 LLM 产出里 1,174 条带
        「根据康熙字典 / 出自《中华字海》」），读者手上并没有那份资料。"""
        self.assertEqual(clean_meaning_block("拼音fan4<br>轻也，逞也。<br>见《直音篇》。"), "轻也，逞也。")
        self.assertEqual(clean_meaning_block("见汉语方言大词典。"), "")
        self.assertEqual(clean_meaning_block("①见「偛」。"), "")

    def test_ordinary_hanzi_with_jian_is_kept(self) -> None:
        """「罕见」「意见」里的「见」是正常的汉字用法，不许被当成互见条目误删。"""
        self.assertEqual(clean_meaning_block("罕见的姓氏。"), "罕见的姓氏。")
        self.assertEqual(clean_meaning_block("义待考，见于香港。"), "义待考，见于香港。")

    def test_layout_only_block_yields_empty(self) -> None:
        """只有排版残片的块清洗后是空串——调用方据此判断「这块没有释义」（实测大量字如此）。"""
        self.assertEqual(clean_meaning_block("<br>"), "")

class GlossItemsTests(unittest.TestCase):
    """T17 的答案必须是一条**纯中文短语**：残片（缺一半的括号 / 词头占位符 / 注音）当答案就是错题。"""

    def _character(self, *blocks: str):
        """输入：若干释义块原文。输出：带这些块的假字记录。预期：只挂 mean_h（中华字海）。"""
        return make_character(meanings={"mean_h": list(blocks)})

    def test_plain_gloss_is_kept(self) -> None:
        """干净的义项原样保留，来源书名照写，答案取第一条。"""
        items, source = gloss_items(self._character("拼音hao3<br>美好。友爱。"))
        self.assertEqual(items, ("美好", "友爱"))
        self.assertEqual(source, "中华字海")

    def test_fragments_are_not_answers(self) -> None:
        """残片一律不当答案：缺一半的括号、词头占位符、注音与罗马字、岁数页码。

        这些片段**不是**本字的释义——`<日乘` 是原文 `【日乘】` 掉了半个括号，`堪～]传说中的一种鱼`
        讲的是词、`读音たたみ` / `韩国读音nol` 是注音、`77歳` 是注记。当答案就是错题。
        """
        for block in ("<日乘。", "捣】的二简字。", "[堪～]传说中的一种鱼。", "读音たたみ。",
                      "韩国读音nol。", "77歳。"):
            with self.subTest(block=block):
                items, _source = gloss_items(self._character(block))
                self.assertEqual(items, (), f"{block} 的残片被当成了义项")

    def test_see_also_reference_is_not_an_answer(self) -> None:
        """互见条目形状上是一条**全汉字**短语，会直接当上 T17 的答案——必须被清洗掉。"""
        items, _source = gloss_items(self._character("拼音fan4<br>轻也。<br>逞也。<br>见《直音篇》。"))
        self.assertEqual(items, ("轻也", "逞也"))

    def test_dirty_item_does_not_take_the_whole_character(self) -> None:
        """一条脏一条干净时，答案取干净的那条——只丢义项，不整字丢掉（覆盖与精度都要）。"""
        items, _source = gloss_items(self._character("人名也。地名也（俗字攷。音译字。"))
        self.assertEqual(items, ("人名也", "音译字"))

    def test_ascii_comma_is_the_same_separator(self) -> None:
        """半角逗号与顿号一样是义项内部分隔，不因为它是 ASCII 就把整条义项丢掉。"""
        items, _source = gloss_items(self._character("分别,分离,分解。"))
        self.assertEqual(items, ("分别,分离,分解",))


if __name__ == "__main__":
    unittest.main()
