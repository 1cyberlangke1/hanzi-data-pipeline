"""关系与溯源类构造器（T5 / T6 / T7）的单元测试。"""

from __future__ import annotations

import random
import unittest

from hanzi_synth.tasks.base import Difficulty
from hanzi_synth.tasks.relations import (
    build_meaning_source,
    build_reference,
    build_standard_form,
    build_variant,
)
from tests.support import make_character, make_index

D = Difficulty()


class StandardFormTests(unittest.TestCase):
    def test_answer_is_standard_form_field(self) -> None:
        """答案 = standard_form 字段直接值。"""
        character = make_character(zi="\u3402", standard_form="\u559c")
        spec = build_standard_form(character, rng=random.Random(1), difficulty=D, index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "\u559c")
        self.assertEqual(spec.facts["standard_form"], "\u559c")

    def test_skips_when_standard_form_is_self(self) -> None:
        """标准字形就是自己 -> 不成题。"""
        character = make_character(zi="\u597d", standard_form="\u597d")
        self.assertIsNone(build_standard_form(character, rng=random.Random(1), difficulty=D, index=make_index([character])))

    def test_skips_multi_char_standard_form(self) -> None:
        """标准字形不是单字就不出题（题面没法展示）。"""
        character = make_character(zi="\u3402", standard_form="\u559c\u6b22")
        self.assertIsNone(build_standard_form(character, rng=random.Random(1), difficulty=D, index=make_index([character])))

    def test_distractors_exclude_answer_and_self(self) -> None:
        """干扰项既不是答案也不是本字。"""
        character = make_character(zi="\u3402", standard_form="\u559c", radical="\u53e3", stroke=12)
        other = make_character(ziid="U9", zi="\u55e8", radical="\u53e3", stroke=12)
        spec = build_standard_form(character, rng=random.Random(1), difficulty=D, index=make_index([character, other]))
        assert spec is not None
        texts = [d.text for d in spec.distractors]
        self.assertNotIn("\u559c", texts)
        self.assertNotIn("\u3402", texts)


class VariantTests(unittest.TestCase):
    def test_all_variants_are_accepted(self) -> None:
        """多个异体全部算对——「一个字当然有多种写法」。"""
        character = make_character(zi="\u3406", variants=("\u5f9d", "\u5d0d"))
        spec = build_variant(character, rng=random.Random(1), difficulty=D, index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "\u5f9d")
        self.assertIn("\u5d0d", spec.accepted)

    def test_skips_without_single_char_variants(self) -> None:
        """只有码位型异体（Z/P 组）时出不了题。"""
        character = make_character(zi="\u3402", variants=("PA0098",))
        self.assertIsNone(build_variant(character, rng=random.Random(1), difficulty=D, index=make_index([character])))

    def test_distractors_are_never_variants(self) -> None:
        """干扰项绝不能其实也是异体（那就是错题）。"""
        character = make_character(zi="\u3406", variants=("\u5f9d",), radical="\u4eba", stroke=8)
        pool = [character] + [
            make_character(ziid=f"U{index}", zi=chr(0x4E10 + index), radical="\u4eba", stroke=8) for index in range(6)
        ]
        spec = build_variant(character, rng=random.Random(4), difficulty=D, index=make_index(pool))
        assert spec is not None
        self.assertNotIn("\u5f9d", [d.text for d in spec.distractors])


class ReferenceTests(unittest.TestCase):
    def test_answer_is_page_field(self) -> None:
        """答案 = refs['zh']['page'] 字段直接值。"""
        character = make_character(zi="\u3400", refs={"zh": {"page": 431, "number": 6}})
        spec = build_reference(character, rng=random.Random(1), difficulty=D, index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "431")
        self.assertEqual(spec.facts["page"], 431)

    def test_skips_without_reference(self) -> None:
        """没有 zh_ref 就不出题（其它三个 ref 字段书名无据可查，一律不用）。"""
        character = make_character(zi="\u3400", refs={"hd": {"page": 1, "number": 1}})
        self.assertIsNone(build_reference(character, rng=random.Random(1), difficulty=D, index=make_index([character])))

    def test_zero_page_is_skipped(self) -> None:
        """页数非法时不出题。"""
        character = make_character(zi="\u3400", refs={"zh": {"page": 0, "number": 0}})
        self.assertIsNone(build_reference(character, rng=random.Random(1), difficulty=D, index=make_index([character])))


class MeaningSourceTests(unittest.TestCase):
    def test_single_source_is_unambiguous(self) -> None:
        """三本库里恰好一本有释义 -> 答案唯一。"""
        character = make_character(zi="\u3400", meanings={"mean_h": ["\u62fc\u97f3yi1<br>\u8868\u793a\u4e00"]})
        spec = build_meaning_source(character, rng=random.Random(1), difficulty=D, index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "\u4e2d\u534e\u5b57\u6d77")
        self.assertEqual(len(spec.distractors), 2)

    def test_zero_sources_skipped(self) -> None:
        """一本都没有 -> 没材料。"""
        character = make_character(zi="\u3400", meanings={"mean_z": ["\u7efc\u5408\u91ca\u4e49"]})
        self.assertIsNone(build_meaning_source(character, rng=random.Random(1), difficulty=D, index=make_index([character])))

    def test_two_sources_skipped_as_ambiguous(self) -> None:
        """两本都有 -> 题目有歧义，丢弃。"""
        character = make_character(zi="\u3400", meanings={"mean_h": ["\u5b57\u6d77\u91ca\u4e49"], "mean_k": ["\u5eb7\u7199\u91ca\u4e49"]})
        self.assertIsNone(build_meaning_source(character, rng=random.Random(1), difficulty=D, index=make_index([character])))

    def test_never_quotes_meaning_text(self) -> None:
        """选项只有书名，facts 里不含任何释义原文（边界要求）。"""
        character = make_character(zi="\u3400", meanings={"mean_h": ["\u62fc\u97f3yi1<br>\u8fd9\u662f\u4e00\u6bb5\u539f\u6587\u91ca\u4e49"]})
        spec = build_meaning_source(character, rng=random.Random(1), difficulty=D, index=make_index([character]))
        assert spec is not None
        self.assertNotIn("\u539f\u6587\u91ca\u4e49", str(spec.facts))


if __name__ == "__main__":
    unittest.main()
