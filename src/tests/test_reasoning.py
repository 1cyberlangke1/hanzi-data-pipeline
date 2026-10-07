"""推理类构造器（T8 约束 / T9 部件链 / T10 反例 / T11 排序 / T12 变换）的单元测试。"""

from __future__ import annotations

import random
import unittest

from hanzi_synth.tasks.base import Difficulty
from hanzi_synth.tasks.reasoning import (
    build_component_chain,
    build_constraint,
    build_ids_transform,
    build_odd_one_out,
    build_stroke_extreme,
)
from tests.support import make_character, make_index

# 测试用字形：㐀 = 上下(卝, 一)；卝 = 左中右(一, 丨, ⺊)；㐁 = 上下(卝, 二)；㐃 = 上下(卝, 三)
A = make_character(ziid="UA", zi="\u3400", ids=("\u2ff1\u535d\u4e00",), stroke=5, radical="\u4e00")
X = make_character(ziid="UX", zi="\u535d", ids=("\u2ff2\u4e00\u4e28\u2e8a",), stroke=4, radical="\u4e28")
B = make_character(ziid="UB", zi="\u3401", ids=("\u2ff1\u535d\u4e8c",), stroke=6, radical="\u4e8c")
C = make_character(ziid="UC", zi="\u3403", ids=("\u2ff1\u535d\u4e09",), stroke=7, radical="\u4e09")


class ConstraintTests(unittest.TestCase):
    def test_unique_bucket_produces_task(self) -> None:
        """(部首, 笔画) 桶里只剩本字 -> 唯一解，出题。"""
        other = make_character(ziid="UZ", zi="\u597d", ids=("\u2ff0\u5973\u5b50",), stroke=6, radical="\u5973")
        spec = build_constraint(A, rng=random.Random(1), difficulty=Difficulty(constraints=1), index=make_index([A, other]))
        assert spec is not None
        self.assertEqual(spec.answer, "\u3400")
        self.assertIn("\u90e8\u9996", spec.facts["conditions"])

    def test_ambiguous_bucket_is_rejected(self) -> None:
        """同部首同笔画有两个字 -> 歧义，直接丢弃。"""
        twin = make_character(ziid="UT", zi="\u3402", ids=("\u2ff1\u4e8c\u4e00",), stroke=5, radical="\u4e00")
        self.assertIsNone(
            build_constraint(A, rng=random.Random(1), difficulty=Difficulty(constraints=1), index=make_index([A, twin]))
        )

    def test_three_constraints_adds_structure_and_count(self) -> None:
        """constraints=3 时条件里出现结构与部件个数——证明旋钮真的生效。"""
        spec = build_constraint(A, rng=random.Random(1), difficulty=Difficulty(constraints=3), index=make_index([A]))
        assert spec is not None
        self.assertIn("component_count", spec.facts)
        self.assertIn("structure", spec.facts)
        self.assertIn("\u4e0a\u4e0b", spec.facts["conditions"])

    def test_answer_never_in_radical_description(self) -> None:
        """答案等于部首名时不出题（否则题干里直接有答案）。"""
        same = make_character(ziid="US", zi="\u4e28", ids=("\u2ff1\u4e00\u4e8c",), stroke=2, radical="\u4e28")
        self.assertIsNone(
            build_constraint(same, rng=random.Random(1), difficulty=Difficulty(constraints=1), index=make_index([same]))
        )

    def test_distractors_do_not_satisfy_constraints(self) -> None:
        """干扰项必须真的不满足条件。"""
        other = make_character(ziid="UZ", zi="\u597d", ids=("\u2ff0\u5973\u5b50",), stroke=6, radical="\u5973")
        spec = build_constraint(A, rng=random.Random(2), difficulty=Difficulty(constraints=1), index=make_index([A, other]))
        assert spec is not None
        self.assertTrue(spec.distractors)
        self.assertTrue(all(d.text != A.zi for d in spec.distractors))


class ComponentChainTests(unittest.TestCase):
    def test_single_hop_asks_for_direct_component(self) -> None:
        """hops=1：路径只有一层，答案是顶层某个部件。"""
        index = make_index([A, X])
        spec = build_component_chain(A, rng=random.Random(1), difficulty=Difficulty(hops=1), index=index)
        assert spec is not None
        self.assertEqual(len(spec.facts["path"]), 1)
        self.assertIn(spec.answer, ("\u535d", "\u4e00"))

    def test_two_hops_never_name_the_answer(self) -> None:
        """hops=2：路径两层，答案是内层部件，且与题干里出现过的字不撞车。"""
        index = make_index([A, X])
        spec = build_component_chain(A, rng=random.Random(1), difficulty=Difficulty(hops=2), index=index)
        assert spec is not None
        self.assertEqual(len(spec.facts["path"]), 2)
        self.assertIn(spec.answer, ("\u4e00", "\u4e28", "\u2e8a"))
        self.assertNotIn(spec.answer, (A.zi, X.zi))
        self.assertEqual(spec.facts["depth"], 2)

    def test_stem_must_ask_for_a_component(self) -> None:
        """问的必须是答案那件事：答案是一个**部件**，问「这个字的结构是什么」读者必然答错
        （实测 16p 的 T9 就是这么写的）。"""
        index = make_index([A, X])
        spec = build_component_chain(A, rng=random.Random(1), difficulty=Difficulty(hops=1), index=index)
        assert spec is not None
        self.assertIn("部件", spec.stem_asks_any)
        self.assertIn("哪个字", spec.stem_asks_any)

    def test_proof_has_no_internal_reference(self) -> None:
        """推理链是自足陈述，不含「题干」这类内部指代（模型会把 proof 抄进解析）。"""
        index = make_index([A, X])
        spec = build_component_chain(A, rng=random.Random(1), difficulty=Difficulty(hops=2), index=index)
        assert spec is not None
        for step in spec.proof:
            self.assertNotIn("题干", step)

    def test_chain_stops_when_component_not_in_pool(self) -> None:
        """中间件不在池里 -> 链子接不起来，返回 None。"""
        lonely = make_character(ziid="UL", zi="\u3400", ids=("\u2ff1\u4e0d\u5728\u4e00",), stroke=5, radical="\u4e00")
        self.assertIsNone(
            build_component_chain(lonely, rng=random.Random(1), difficulty=Difficulty(hops=2), index=make_index([lonely]))
        )

    def test_leaf_character_has_no_chain(self) -> None:
        """独体字没有部件可拆。"""
        leaf = make_character(ziid="UF", zi="\u4e00", ids=("\u4e00",))
        self.assertIsNone(
            build_component_chain(leaf, rng=random.Random(1), difficulty=Difficulty(hops=1), index=make_index([leaf]))
        )


class OddOneOutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mates = [
            make_character(ziid=f"UM{index}", zi=chr(0x5974 + index), ids=("\u2ff0\u5973\u5b50",), stroke=6, radical="\u5973")
            for index in range(4)
        ]

    def test_answer_is_the_odd_character(self) -> None:
        """答案是那个部首不同的字；选项 4 个、答案在选项里。"""
        spec = build_odd_one_out(A, rng=random.Random(1), difficulty=Difficulty(), index=make_index([A, *self.mates]))
        assert spec is not None
        self.assertEqual(spec.answer, A.zi)
        self.assertEqual(len(spec.options), 4)
        self.assertIn(spec.answer, spec.options)
        self.assertEqual(spec.facts["radical"], "\u5973")
        self.assertEqual(len(spec.distractors), 3)

    def test_stem_must_name_the_comparison_target(self) -> None:
        """「部首」是本题的比较对象，必须作为程序侧的必现串声明——实测 16z 的产出只写
        「请从下列选项中选出哪一个：㐀、…」，读者不知道该按什么挑。
        """
        spec = build_odd_one_out(A, rng=random.Random(1), difficulty=Difficulty(), index=make_index([A, *self.mates]))
        assert spec is not None
        self.assertEqual(spec.must_contain, ("部首",))

    def test_skips_when_no_group_is_big_enough(self) -> None:
        """凑不齐 3 个同部首同伴就不出题。"""
        self.assertIsNone(
            build_odd_one_out(A, rng=random.Random(1), difficulty=Difficulty(), index=make_index([A, self.mates[0]]))
        )


class StrokeExtremeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = make_character(ziid="UT", zi="\u3400", ids=("\u2ff1\u535d\u4e00",), stroke=10, radical="\u4e00")
        self.pool = [self.target] + [
            make_character(ziid=f"US{stroke}", zi=chr(0x4E00 + stroke), ids=("\u2ff1\u4e00\u4e8c",), stroke=stroke, radical="\u4e8c")
            for stroke in (1, 2, 3, 11, 12, 13)
        ]

    def test_extreme_is_unique_and_in_options(self) -> None:
        """极值唯一：同侧同伴笔画严格更少/更多，且答案在选项里。"""
        spec = build_stroke_extreme(self.target, rng=random.Random(7), difficulty=Difficulty(), index=make_index(self.pool))
        assert spec is not None
        self.assertEqual(spec.answer, self.target.zi)
        self.assertIn(spec.answer, spec.options)
        strokes = spec.facts["strokes"]
        others = [value for key, value in strokes.items() if key != self.target.zi]
        if spec.facts["direction"] == "most":
            self.assertTrue(all(value < 10 for value in others))
        else:
            self.assertTrue(all(value > 10 for value in others))

    def test_stem_must_name_the_comparison_target(self) -> None:
        """「笔画」是本题的比较对象，必须作为程序侧的必现串声明（与 T10 同因）。"""
        spec = build_stroke_extreme(self.target, rng=random.Random(7), difficulty=Difficulty(), index=make_index(self.pool))
        assert spec is not None
        self.assertEqual(spec.must_contain, ("笔画",))

    def test_skips_when_not_enough_distinct_strokes(self) -> None:
        """同侧凑不出 3 个不同笔画就不出题。"""
        thin = [self.target, make_character(ziid="U1", zi="\u4e00", ids=("\u2ff1\u4e00\u4e8c",), stroke=1, radical="\u4e8c")]
        self.assertIsNone(
            build_stroke_extreme(self.target, rng=random.Random(1), difficulty=Difficulty(), index=make_index(thin))
        )


class IdsTransformTests(unittest.TestCase):
    def test_single_edit_is_verifiable(self) -> None:
        """一次替换：答案必须是池里真实收录的那个字，编辑描述与字段一致。"""
        # 池里只放 A 与 B，替换位置唯一（下边的「一」-> 「二」），结果才确定
        spec = build_ids_transform(A, rng=random.Random(1), difficulty=Difficulty(hops=1), index=make_index([A, B]))
        assert spec is not None
        self.assertEqual(spec.answer, B.zi)
        self.assertEqual(spec.facts["edits"], ["\u628a\u4e0b\u8fb9\u7684\u90e8\u4ef6\u300c\u4e00\u300d\u6362\u6210\u300c\u4e8c\u300d"])
        self.assertNotIn(spec.answer, spec.facts["start_zi"])

    def test_skips_when_only_edit_leads_back_to_start(self) -> None:
        """唯一可行的替换会绕回起点 -> 答案泄露，直接丢弃。"""
        spec = build_ids_transform(B, rng=random.Random(1), difficulty=Difficulty(hops=2), index=make_index([A, B]))
        self.assertIsNone(spec)

    def test_leaf_character_has_no_edit(self) -> None:
        """独体字没有可替换的部件。"""
        leaf = make_character(ziid="UF", zi="\u4e00", ids=("\u4e00",))
        self.assertIsNone(
            build_ids_transform(leaf, rng=random.Random(1), difficulty=Difficulty(hops=1), index=make_index([leaf]))
        )


if __name__ == "__main__":
    unittest.main()
