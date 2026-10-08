"""IDS 解析与 surface 变换的单元测试。"""

from __future__ import annotations

import unittest

from hanzi_synth.facts.ids import (
    IdsSyntaxError,
    is_component_char,
    is_wellformed,
    leaves,
    parse_candidates,
    parse_tree,
)
from hanzi_synth.facts.surface import char_surfaces, has_idc, ids_surfaces, reading_surfaces


class ParseCandidatesTests(unittest.TestCase):
    def test_multi_candidate_with_source_tags(self) -> None:
        """多候选串联：剥掉来源标签后每一项都保留，顺序不变。"""
        raw = "\u2ff0\u5973\u5b50(G),\u2ff1\u597d\u4e00(T)"
        self.assertEqual(parse_candidates(raw), ("\u2ff0\u5973\u5b50", "\u2ff1\u597d\u4e00"))

    def test_pure_stroke_notation_is_not_ids(self) -> None:
        """`#(H)(G)` 是叶典笔画记法，`{X}` 是「X 本身」，都不是 IDS，必须丢弃。"""
        self.assertEqual(parse_candidates("#(H)(G),{\u4e00}#(T)(t)"), ())

    def test_katakana_component_is_rejected(self) -> None:
        """片假名只是叶典的记号，不是合法部件——含它的候选整条判非法。

        真实例子：`丑` 的三个候选（`⿼ユ十` / `⿻⿼ユ丨一` / `⿻⿼コ丨一`）全部含片假名，
        所以这个字在严格标准下没有可用 IDS。
        """
        self.assertEqual(parse_candidates("\u2ffc\u30e6\u5341"), ())
        self.assertEqual(parse_candidates("\u2ffc\u30e6\u5341(G),\u2ffb\u2ffc\u30e6\u4e28\u4e00(H)"), ())

    def test_empty_and_none(self) -> None:
        """空串与 None 都返回空元组，不抛异常。"""
        self.assertEqual(parse_candidates(""), ())
        self.assertEqual(parse_candidates(None), ())

    def test_duplicate_candidates_are_deduped(self) -> None:
        """同一候选重复出现只保留一次。"""
        self.assertEqual(parse_candidates("\u2ff0\u5973\u5b50,\u2ff0\u5973\u5b50"), ("\u2ff0\u5973\u5b50",))


class ParseTreeTests(unittest.TestCase):
    def test_binary_tree_and_leaves(self) -> None:
        """二元描述符按左右顺序解析，叶子顺序与书写顺序一致。"""
        tree = parse_tree("\u2ff0\u5973\u5b50")
        self.assertEqual(tree.op, "\u2ff0")
        self.assertEqual(leaves(tree), ("\u5973", "\u5b50"))

    def test_nested_tree(self) -> None:
        """嵌套 IDS 递归展开。"""
        self.assertEqual(leaves(parse_tree("\u2ff1\u4e03\u2ff0\u4e03\u4e03")), ("\u4e03", "\u4e03", "\u4e03"))

    def test_ternary_tree(self) -> None:
        """三元描述符消费三个操作数。"""
        self.assertEqual(leaves(parse_tree("\u2ff2\u4e00\u4e8c\u4e09")), ("\u4e00", "\u4e8c", "\u4e09"))

    def test_incomplete_ids_raises(self) -> None:
        """元数不够（如 `⿱丨`）必须抛错，而不是静默截断。"""
        with self.assertRaises(IdsSyntaxError):
            parse_tree("\u2ff1\u4e28")

    def test_trailing_garbage_raises(self) -> None:
        """描述符消费完还有剩余字符同样抛错。"""
        with self.assertRaises(IdsSyntaxError):
            parse_tree("\u2ff0\u5973\u5b50\u5b50")

    def test_non_component_char_raises(self) -> None:
        """拉丁字母不是部件，解析必须失败。"""
        with self.assertRaises(IdsSyntaxError):
            parse_tree("\u2ff0a\u5b50")


class WellFormedTests(unittest.TestCase):
    def test_requires_idc_leading(self) -> None:
        """首字符不是描述符的一律非法。"""
        self.assertFalse(is_wellformed("\u5973\u5b50"))
        self.assertTrue(is_wellformed("\u2ff0\u5973\u5b50"))

    def test_component_char_ranges(self) -> None:
        """汉字与部首补充区算部件；拉丁字母/片假名不算。"""
        self.assertTrue(is_component_char("\u4e00"))
        self.assertTrue(is_component_char("\u2e80"))
        self.assertFalse(is_component_char("A"))
        self.assertFalse(is_component_char("\u30e6"))


class ReadingSurfaceTests(unittest.TestCase):
    def test_tone_mark_placement(self) -> None:
        """标调位置遵守汉语拼音规则：有 a 标 a、无 a 标 o/e、否则标最后一个元音。"""
        self.assertEqual(reading_surfaces("hao4"), ("hao4", "h\u00e0o"))
        self.assertEqual(reading_surfaces("zhang1"), ("zhang1", "zh\u0101ng"))
        self.assertEqual(reading_surfaces("gui1"), ("gui1", "gu\u012b"))
        self.assertEqual(reading_surfaces("liu2"), ("liu2", "li\u00fa"))

    def test_v_stands_for_umlaut(self) -> None:
        """canonical 用 v 表示 ü，surface 还原成带调符的 ü。"""
        self.assertEqual(reading_surfaces("lv4"), ("lv4", "l\u01dc"))

    def test_neutral_and_toneless_keep_single_form(self) -> None:
        """轻声与无调音节没有第二种写法。"""
        self.assertEqual(reading_surfaces("ma5"), ("ma5",))
        self.assertEqual(reading_surfaces("xu"), ("xu",))
        self.assertEqual(reading_surfaces(""), ())


class OtherSurfaceTests(unittest.TestCase):
    def test_char_surfaces(self) -> None:
        """字符同时给出原字与码点写法。"""
        self.assertEqual(char_surfaces("\u3402"), ("\u3402", "U+3402"))
        self.assertEqual(char_surfaces(""), ())

    def test_ids_surfaces_describes_operator(self) -> None:
        """IDS 的描述式用运算符的中文意译。"""
        forms = ids_surfaces("\u2ff0\u5973\u5b50")
        self.assertEqual(forms[0], "\u2ff0\u5973\u5b50")
        self.assertIn("\u5de6\u53f3", forms[1])

    def test_has_idc(self) -> None:
        """has_idc 用于判断文本里有没有描述符。"""
        self.assertTrue(has_idc("\u2ff0\u5973\u5b50"))
        self.assertFalse(has_idc("\u5973\u5b50"))


if __name__ == "__main__":
    unittest.main()
