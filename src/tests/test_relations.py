"""关系与溯源类构造器（T5 / T6）的单元测试。"""

from __future__ import annotations

import random
import unittest

from hanzi_synth.tasks.base import Difficulty
from hanzi_synth.tasks.relations import (
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


if __name__ == "__main__":
    unittest.main()
