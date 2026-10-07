"""合成池索引的单元测试。"""

from __future__ import annotations

import unittest

from hanzi_synth.facts.index import BLANK, CharacterIndex, blanked_key, bucket_key
from tests.support import make_character


class BucketKeyTests(unittest.TestCase):
    def test_uses_ids_top_operator(self) -> None:
        """桶键的第三项取自 IDS 顶层算子，而不是无文档的 structure 整数码。"""
        character = make_character(ids=("\u2ff1\u597d\u4e00",), radical="\u5973", stroke=6)
        self.assertEqual(bucket_key(character), ("\u5973", 6, "\u2ff1"))

    def test_missing_fields_return_none(self) -> None:
        """缺部首或缺笔画就不进桶，绝不猜。"""
        self.assertIsNone(bucket_key(make_character(radical=None)))
        self.assertIsNone(bucket_key(make_character(stroke=0)))


class BlankedKeyTests(unittest.TestCase):
    def test_two_chars_differing_by_one_part_share_a_key(self) -> None:
        """只差一个部件的两个字必须落到同一个挖空键上——T12 的等价类基础。"""
        left = blanked_key("\u2ff1\u597d\u4e00", 1)
        right = blanked_key("\u2ff1\u597d\u4e8c", 1)
        self.assertIsNotNone(left)
        self.assertEqual(left, right)
        self.assertIn(BLANK, left or "")

    def test_leaf_ids_has_no_key(self) -> None:
        """独体字没有可挖的叶子，返回 None。"""
        self.assertIsNone(blanked_key("\u4e00", 0))

    def test_out_of_range_returns_none(self) -> None:
        """下标越界返回 None，不抛栈。"""
        self.assertIsNone(blanked_key("\u2ff1\u597d\u4e00", 9))


class CharacterIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hao = make_character(ziid="U1", zi="\u597d", ids=("\u2ff0\u5973\u5b50",), stroke=6, radical="\u5973")
        self.miao = make_character(ziid="U2", zi="\u5999", ids=("\u2ff0\u5973\u5c11",), stroke=7, radical="\u5973")
        self.zi = make_character(ziid="U3", zi="\u5b50", ids=("\u2ff1\u4e86\u4e00",), stroke=3, radical="\u5b50")
        self.index = CharacterIndex([self.hao, self.miao, self.zi])

    def test_lookup_char(self) -> None:
        """字符反查命中，未知字符返回 None。"""
        self.assertIs(self.index.lookup_char("\u597d"), self.hao)
        self.assertIsNone(self.index.lookup_char("\u4e0d\u5b58\u5728"))

    def test_lookup_ids(self) -> None:
        """IDS 反查命中，未知 IDS 返回空元组。"""
        self.assertEqual(self.index.lookup_ids("\u2ff0\u5973\u5b50"), (self.hao,))
        self.assertEqual(self.index.lookup_ids("\u2ff0\u4e0d\u5b58"), ())

    def test_same_bucket_excludes_different_stroke(self) -> None:
        """同部首但笔画不同 -> 不同桶。"""
        self.assertEqual([c.ziid for c in self.index.same_bucket(self.hao)], ["U1"])

    def test_same_radical_never_returns_self(self) -> None:
        """同部首查询绝不返回目标字自己。"""
        picked = self.index.same_radical(self.hao)
        self.assertTrue(all(c.ziid != self.hao.ziid for c in picked))
        self.assertIn(self.miao.ziid, [c.ziid for c in picked])

    def test_blank_mates_finds_real_replacement(self) -> None:
        """挖空索引能找出「换掉某个部件后真实存在」的字。"""
        mates = self.index.blank_mates("\u2ff0\u5973\u5b50", 1, exclude_zi="\u597d")
        self.assertEqual([c.ziid for c, _pos, _cand in mates], ["U2"])

    def test_by_stroke_groups(self) -> None:
        """按笔画分桶，供 T11 排序题取严格同侧的同伴。"""
        self.assertEqual([c.ziid for c in self.index.by_stroke[3]], ["U3"])


if __name__ == "__main__":
    unittest.main()
