"""码位 / 编码类构造器（T14 码位、T15 UTF-8、T16 码位反查）与 surface 变换的单元测试。"""

from __future__ import annotations

import random
import unittest

from hanzi_synth.facts.surface import codepoint_surfaces
from hanzi_synth.tasks.base import Difficulty
from hanzi_synth.tasks.registry import CONSTRUCTORS, build_all
from hanzi_synth.tasks.unicode import build_unicode_bytes, build_unicode_codepoint, build_unicode_reverse
from tests.support import make_character, make_index

# 测试用字：一(U+4E00) / 二(U+4E8C) / 三(U+4E09) / 十(U+5341) / 𠀀(U+20000，非 BMP)
YI = make_character(ziid="U4E00", zi="\u4e00", ids=("\u2ff0\u4e00\u4e00",), stroke=1, radical="\u4e00")
ER = make_character(ziid="U4E8C", zi="\u4e8c", ids=("\u2ff1\u4e00\u4e00",), stroke=2, radical="\u4e00")
SAN = make_character(ziid="U4E09", zi="\u4e09", ids=("\u2ff1\u4e00\u4e8c",), stroke=3, radical="\u4e00")
SHI = make_character(ziid="U5341", zi="\u5341", ids=("\u2ff0\u4e00\u4e28",), stroke=2, radical="\u4e00")
EXT = make_character(ziid="U20000", zi="\U00020000", ids=("\u2ff0\u4e00\u4e00",), stroke=2, radical="\u4e00")
NO_CODE = make_character(ziid="P80004", zi="", ids=("\u2ff0\u4e00\u4e28",), stroke=2, radical="\u4e00")

SMALL_INDEX = make_index([YI, ER, SAN, SHI, EXT])


class CodepointSurfaceTests(unittest.TestCase):
    def test_bmp_has_four_forms_in_fixed_order(self) -> None:
        """BMP 字给 4 条等价写法，顺序固定：带前缀 hex / 裸 hex / 十进制 / UTF-8 字节。"""
        self.assertEqual(codepoint_surfaces("\u4e00"), ("U+4E00", "4E00", "19968", "E4 B8 80"))

    def test_non_bmp_adds_utf16_surrogate_pair(self) -> None:
        """补充平面字多一条 UTF-16 代理对，UTF-8 是 4 字节。"""
        forms = codepoint_surfaces("\U00020000")
        self.assertEqual(forms[0], "U+20000")
        self.assertEqual(forms[2], "131072")
        self.assertEqual(forms[3], "F0 A0 80 80")
        self.assertEqual(forms[4], "D840 DC00")

    def test_non_single_char_returns_empty(self) -> None:
        """空串与多字符串都不是一个码位，返回空元组而不是瞎猜。"""
        self.assertEqual(codepoint_surfaces(""), ())
        self.assertEqual(codepoint_surfaces("\u4e00\u4e8c"), ())


class CodepointTaskTests(unittest.TestCase):
    def test_answer_is_one_of_the_forms_and_aliases_cover_the_rest(self) -> None:
        """T14：主答案是**数值写法**之一，三种数值写法全部进判分集合（多写法都算对）。"""
        spec = build_unicode_codepoint(YI, rng=random.Random(7), difficulty=Difficulty(), index=SMALL_INDEX)
        assert spec is not None
        forms = set(codepoint_surfaces("\u4e00")[:3])
        self.assertIn(spec.answer, forms)
        self.assertEqual(forms, set(spec.accepted))

    def test_codepointless_char_is_skipped(self) -> None:
        """Z/P 组无码位字不出码位题——没有码位就没有可学的编码。"""
        self.assertIsNone(build_unicode_codepoint(NO_CODE, rng=random.Random(1), difficulty=Difficulty(), index=SMALL_INDEX))
        self.assertIsNone(build_unicode_bytes(NO_CODE, rng=random.Random(1), difficulty=Difficulty(), index=SMALL_INDEX))
        self.assertIsNone(build_unicode_reverse(NO_CODE, rng=random.Random(1), difficulty=Difficulty(), index=SMALL_INDEX))

    def test_distractor_distance_follows_the_dial(self) -> None:
        """干扰码位与正确码位的距离随 distractor_depth 变大（near <= 2 < far）。"""
        code = ord("\u4e00")

        def gaps(depth: str) -> list[int]:
            spec = build_unicode_codepoint(
                YI, rng=random.Random(3), difficulty=Difficulty(distractor_depth=depth), index=SMALL_INDEX
            )
            assert spec is not None
            return [abs(int(d.text[2:], 16) - code) for d in spec.distractors]

        self.assertTrue(all(gap <= 2 for gap in gaps("near")))
        self.assertTrue(all(gap >= 0x100 for gap in gaps("far")))

    def test_utf8_answer_and_aliases(self) -> None:
        """T15：答案是 UTF-8 字节序列（空格或紧凑写法），十进制码位不算对（那是 T14 的题）。"""
        spec = build_unicode_bytes(YI, rng=random.Random(5), difficulty=Difficulty(), index=SMALL_INDEX)
        assert spec is not None
        self.assertIn(spec.answer, ("E4 B8 80", "E4B880"))
        self.assertIn("E4 B8 80", spec.accepted)
        self.assertIn("E4B880", spec.accepted)
        self.assertNotIn("19968", spec.accepted)
        self.assertEqual(spec.facts["byte_count"], 3)
        self.assertTrue(spec.distractors)
        self.assertNotIn(spec.answer, [d.text for d in spec.distractors])

    def test_non_bmp_teaches_utf16_too(self) -> None:
        """补充平面字：facts 里带 UTF-16 代理对，编码这一课讲全。"""
        spec = build_unicode_bytes(EXT, rng=random.Random(5), difficulty=Difficulty(), index=SMALL_INDEX)
        assert spec is not None
        self.assertEqual(spec.facts["utf16_units"], "D840 DC00")
        self.assertIn(spec.answer, ("F0 A0 80 80", "F0A08080"))
        self.assertIn("F0 A0 80 80", spec.accepted)

    def test_reverse_is_a_choice_question_with_the_answer_in_options(self) -> None:
        """T16：选择题，答案在选项里、选项不重复、至少 3 个选项。"""
        spec = build_unicode_reverse(YI, rng=random.Random(11), difficulty=Difficulty(), index=SMALL_INDEX)
        assert spec is not None
        self.assertEqual(spec.answer, "\u4e00")
        self.assertIn(spec.answer, spec.options)
        self.assertEqual(len(set(spec.options)), len(spec.options))
        self.assertGreaterEqual(len(spec.options), 3)

    def test_reverse_only_asks_with_numeric_codepoint_forms(self) -> None:
        """「码位」只许是**数值写法**（U+XXXX / 裸 hex / 十进制）——UTF-8 字节序列不是码位。

        预期：40 个种子下 `facts["codepoint"]` 与 `visible_subject`（题干必须写出的那个串）都落在
              前三条写法里；旧版从全部写法轮换，实测会问成「码位 E3 90 8B」（主人点名过）。
        """
        allowed = set(codepoint_surfaces("\u4e00")[:3])
        self.assertEqual(len(allowed), 3)
        for seed in range(40):
            with self.subTest(seed=seed):
                spec = build_unicode_reverse(
                    YI, rng=random.Random(seed), difficulty=Difficulty(), index=SMALL_INDEX
                )
                assert spec is not None
                self.assertIn(spec.facts["codepoint"], allowed)
                self.assertIn(spec.visible_subject, allowed)

    def test_reverse_needs_enough_neighbours(self) -> None:
        """凑不满 3 个选项（两选一是抛硬币）就不出题。"""
        lonely = make_index([YI, ER])
        self.assertIsNone(build_unicode_reverse(YI, rng=random.Random(11), difficulty=Difficulty(), index=lonely))

    def test_registry_exposes_the_three_new_tasks(self) -> None:
        """三个新题型必须真的注册进 registry，且 build_all 能一次带出来。"""
        for task_id in ("T14_unicode_codepoint", "T15_unicode_bytes", "T16_unicode_reverse"):
            self.assertIn(task_id, CONSTRUCTORS)
        specs = build_all(YI, rng=random.Random(2), index=SMALL_INDEX, difficulty=Difficulty())
        ids = {spec.task_id for spec in specs}
        self.assertLessEqual(
            {"T14_unicode_codepoint", "T15_unicode_bytes", "T16_unicode_reverse"}, ids
        )


if __name__ == "__main__":
    unittest.main()
