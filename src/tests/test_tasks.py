"""L2 字形/读音类构造器与注册表的单元测试。"""

from __future__ import annotations

import random
import unittest

from hanzi_synth.tasks.base import DISTRACTOR_DEPTHS, MIXED_DEPTH, Difficulty, resolve_depth
from hanzi_synth.tasks.glyph import build_ids_forward, build_ids_reverse, build_stroke_total
from hanzi_synth.tasks.reading import build_reading, canonical_reading, extract_readings
from hanzi_synth.tasks.registry import CONSTRUCTORS, TASK_STEM_ASKS, build_all
from tests.support import make_character, make_index


class StrokeTotalTests(unittest.TestCase):
    def test_answer_is_field_value(self) -> None:
        """答案必须是 stroke_total 的直接值，干扰项不含答案本身。"""
        character = make_character(stroke=7)
        spec = build_stroke_total(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "7")
        self.assertNotIn("7", [d.text for d in spec.distractors])

    def test_returns_none_without_visible_char(self) -> None:
        """无码位字（zi 为空）出不了看字形的题。"""
        character = make_character(ziid="Z00001", zi="")
        self.assertIsNone(
            build_stroke_total(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        )

    def test_distractor_depth_changes_offsets(self) -> None:
        """near 档干扰项只差 1~2；far 档明显更远——证明难度旋钮真的接上了。"""
        character = make_character(stroke=20)
        index = make_index([character])
        near = build_stroke_total(character, rng=random.Random(3), difficulty=Difficulty(distractor_depth="near"), index=index)
        far = build_stroke_total(character, rng=random.Random(3), difficulty=Difficulty(distractor_depth="far"), index=index)
        assert near is not None and far is not None
        self.assertTrue(all(abs(int(d.text) - 20) <= 2 for d in near.distractors))
        self.assertTrue(any(abs(int(d.text) - 20) > 2 for d in far.distractors))


class IdsForwardTests(unittest.TestCase):
    def test_multi_candidate_all_accepted(self) -> None:
        """一字多解全部算对——「一个字当然有多种写法」。"""
        character = make_character(ids=("\u2ff0\u5973\u5b50", "\u2ff1\u597d\u4e00"))
        spec = build_ids_forward(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "\u2ff0\u5973\u5b50")
        self.assertIn("\u2ff1\u597d\u4e00", spec.accepted)

    def test_surface_forms_are_accepted(self) -> None:
        """描述式 surface 也在判分集合里。"""
        character = make_character()
        spec = build_ids_forward(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertTrue(any("\u5de6\u53f3" in item for item in spec.accepted))

    def test_proof_is_self_contained(self) -> None:
        """推理链是自足陈述，不含「题干给出」这类内部指代。"""
        character = make_character()
        spec = build_ids_forward(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        for step in spec.proof:
            self.assertNotIn("\u9898\u5e72", step)

    def test_stem_must_ask_for_the_component_structure(self) -> None:
        """题干问的必须是「结构拆解」那件事：T3 漏过「问这个字一共有几画」这种换了问题的产出。

        说明与闸门里都不出现 `IDS`：那是英文缩写，2B 会连英文夹注一起抄进中文题干
        （实测产出「请说明字符㐀的 IDS（ideographic description sequence）分解方式」）。
        闸门只认中文说法，说明也就用中文说法去描述要问什么。
        """
        character = make_character()
        spec = build_ids_forward(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertIn("拆解", spec.stem_asks_any)
        self.assertNotIn("IDS", spec.stem_asks_any)
        self.assertNotIn("IDS", spec.instruction)

    def test_distractors_never_repeat_the_answer(self) -> None:
        """干扰项的 IDS 绝不能等于本字的任一候选（否则「其实也对」）。"""
        target = make_character(ziid="U1", zi="\u597d", ids=("\u2ff0\u5973\u5b50",))
        other = make_character(ziid="U2", zi="\u5999", ids=("\u2ff0\u5973\u5c11",), radical="\u5973")
        spec = build_ids_forward(target, rng=random.Random(2), difficulty=Difficulty(), index=make_index([target, other]))
        assert spec is not None
        self.assertNotIn("\u2ff0\u5973\u5b50", [d.text for d in spec.distractors])


class IdsReverseTests(unittest.TestCase):
    def test_requires_meaning(self) -> None:
        """没有释义就出不了反向题。"""
        character = make_character()
        self.assertIsNone(
            build_ids_reverse(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        )

    def test_carries_meaning_source_for_restating(self) -> None:
        """有释义时把原文交给 L3 去重述，答案仍是 IDS 候选集合。"""
        character = make_character(meanings={"mean_h": ["\u6307\u4e8b\u7269\u4ee4\u4eba\u6ee1\u610f\u3002"]})
        spec = build_ids_reverse(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertIn("meaning_source", spec.facts)
        self.assertEqual(spec.answer, "\u2ff0\u5973\u5b50")

    def test_carries_reading_and_requires_it_in_the_stem(self) -> None:
        """释义里带读音标注时，读音必须进 facts 并被写进要求——反向题不写那个字，
        读者只能靠「读音 + 释义」认出是哪个字（实测缺了识别条件就被裁判判「未指定具体字」）。
        """
        character = make_character(
            meanings={"mean_h": ["ㄨㄩ拼音hao3<br>①指事物令人满意。"]}
        )
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertEqual(spec.facts["reading"], ("hao3",))
        self.assertTrue(spec.facts["reading_display"])
        # 识别条件是**程序侧闸门**（`must_contain`），不是散文要求：写进要求里模型照样漏。
        self.assertEqual(spec.must_contain, tuple(spec.facts["reading_display"]))
        self.assertIn("reading", spec.instruction)

    def test_declares_structure_words_and_forbids_them_in_stem(self) -> None:
        """T13 必须声明答案真实用到的位置词，并要求题干里一个都不出现——实测 2B 用
        「字形结构属于左上包围」把答案那棵树用散文说了一遍，读者照抄就能得分。
        """
        character = make_character(
            ids=("\u2ff8\u5e7f\u6728",),
            meanings={"mean_h": ["\u62fc\u97f3yi1<br>\u8868\u793a\u4e00\u4e2a\u610f\u601d\u3002"]},
        )
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertEqual(spec.structure_words, ("\u5de6\u4e0a\u5305\u56f4",))
        # 问的必须是答案那件事：答案是结构拆解，题干就得问拆解（实测漏网：「这个字是什么？」）
        self.assertIn("\u62c6\u89e3", spec.stem_asks_any)

    def test_requires_own_words_and_forbids_cross_references(self) -> None:
        """题干必须禁抄释义原文的分点标记与互见条目——实测模型把「同【依】」原样抄进题干，
        等于把答案那个字直接写出来；同时也要禁 IDS（那才是答案）。
        """
        character = make_character(
            meanings={"mean_h": ["ㄨㄩ拼音yi1<br>①归依。<br>②同【依】。<br>ㄨㄩ拼音yin3<br>①转身。"]}
        )
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        # 不许照抄释义原文（含互见条目）、不许出现 IDS、不许写出这个字本身——三条都由
        # 程序侧闸门兜（`no_source_copy` / `no_answer_leak` / `hidden_subject`），
        # 说明里不写禁令句（写了会被 2B 翻译进题干）。
        self.assertEqual(spec.hidden_subject, character.zi)
        self.assertNotIn("\u540c\u3010", spec.instruction)

    def test_without_reading_still_builds(self) -> None:
        """释义里抽不出读音时不砍这批字：只是不加「必须写读音」那条要求。
        """
        character = make_character(meanings={"mean_h": ["指事物令人满意。"]})
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertEqual(spec.facts["reading"], ())
        # 抽不出读音就不加这条必现串——不能拿「读音」两个字当代理：隐藏主语那一条本来
        # 就该提读音，用代理会把好数据判死
        self.assertEqual(spec.must_contain, ())

    def test_cross_references_are_stripped_before_wording(self) -> None:
        """释义里的互见条目必须在构造时删掉：它会把答案那个字直接写出来
        （实测 2B 把「同『依』」原样抄进题干）。删掉靠字符串处理，不靠求模型别抄。
        """
        character = make_character(
            meanings={
                "mean_h": [
                    "ㄨㄩ拼音yi1<br>①归。<br>②同【依】。<br>"
                    "ㄨㄩ拼音yin3<br>①归依。<br>② 转身。<br>③同【隐】。"
                ]
            }
        )
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        source = spec.facts["meaning_source"]
        self.assertNotIn("同【", source)
        self.assertNotIn("隐", source)
        self.assertIn("归", source)
        self.assertEqual(spec.facts["reading"], ("yi1", "yin3"))

    def test_short_meaning_is_skipped(self) -> None:
        """释义太短（<6 字）没材料可重述，直接不出。"""
        character = make_character(meanings={"mean_h": ["\u597d"]})
        self.assertIsNone(
            build_ids_reverse(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        )


    def test_reading_is_declared_as_must_contain(self) -> None:
        """读音不只是「写作要求」，而是程序侧必须出现的串（`must_contain`）——
        反向题不写那个字，读者只能靠读音认出是哪个字。"""
        character = make_character(meanings={"mean_h": ["ㄨㄩ拼音hao3<br>①指事物令人满意。"]})
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertEqual(spec.must_contain, ("hao3",))

    def test_without_reading_has_no_must_contain(self) -> None:
        """抽不出读音时（朝鲜吏读那类注音）不强求：`must_contain` 为空，不能因此砍掉整批字。"""
        character = make_character(meanings={"mean_h": ["读音gul。音译字。古文書所見奴婢名也。"]})
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertEqual(spec.must_contain, ())
        self.assertEqual(spec.facts["reading"], ())

    def test_headword_only_meaning_is_skipped(self) -> None:
        """释义只讲「本字与某字组成的词」时不出题：`[～毒]` 里的 `～` 就是本字本身，删掉词头后
        剩下的「印度的古译名」无所指，出了题也只能得到「拼音对应的是印度的古译名」这种不成话的句子。"""
        character = make_character(
            ziid="U03416", zi="㐖", ids=("⿰吉乚",),
            meanings={"mean_h": ["拼音xie2<br>[～毒]印度的古译名。"]},
        )
        self.assertIsNone(
            build_ids_reverse(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        )

    def test_meaning_naming_the_character_itself_is_skipped(self) -> None:
        """释义里出现本字本身时不出反向题：题干把字写出来了，读者根本不用认。
        实测 㐂 的释义就是「【喜】的草体写成「㐂」三个七，……」。"""
        character = make_character(
            zi="㐂", ids=("⿱七⿰七七",),
            meanings={"mean_h": ["77歳。【喜】的草体写成「㐂」三个七，成“品”字形相叠。"]},
        )
        self.assertIsNone(
            build_ids_reverse(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        )

    def test_falls_through_to_next_meaning_block(self) -> None:
        """首块只有排版残片时改用后面有正文的释义库——实测大量字的首块就是孤零零一个 `<br>`。"""
        character = make_character(meanings={"mean_h": ["<br>"], "mean_z": ["指事物令人满意。"]})
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertEqual(spec.facts["meaning_source"], "指事物令人满意。")

    def test_reading_marker_is_stripped_from_meaning_source(self) -> None:
        """读音标注不进释义原文：留着它模型会把紧随其后的解释挂到读音上；
        但读音仍要从**原始块**里抽得到，否则反向题就少了识别条件。"""
        character = make_character(meanings={"mean_h": ["ㄨㄩ拼音hao3<br>①指事物令人满意。"]})
        spec = build_ids_reverse(
            character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character])
        )
        assert spec is not None
        self.assertNotIn("拼音", spec.facts["meaning_source"])
        self.assertEqual(spec.facts["reading"], ("hao3",))


class ReadingTests(unittest.TestCase):
    def test_extract_and_canonicalize(self) -> None:
        """从释义标注里抽读音，ü 归一成 v。"""
        self.assertEqual(extract_readings("\u62fc\u97f3hao4<br>\u597d\u7684\u610f\u601d"), ("hao4",))
        self.assertEqual(canonical_reading("\u00fc", "4"), "v4")

    def test_multi_reading_character_is_skipped(self) -> None:
        """多音字会让题目歧义，直接不出。"""
        character = make_character(meanings={"mean_h": ["\u62fc\u97f3hao3<br>\u62fc\u97f3hao4"]})
        self.assertIsNone(
            build_reading(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        )

    def test_single_reading_produces_spec(self) -> None:
        """单音字正常出题，答案集合含声调符写法。"""
        character = make_character(meanings={"mean_h": ["\u62fc\u97f3hao4"]})
        spec = build_reading(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertEqual(spec.answer, "hao4")
        self.assertIn("h\u00e0o", spec.accepted)

    def test_instruction_only_says_what_to_ask(self) -> None:
        """T4 的出题说明只说「问什么」，一条写作要求 / 禁令都不写。

        实测：说明里写「列出所有候选」「请用自己的话解释…保留…」时，2B 会把这些句子**翻译**
        进中文题干（`--limit 8` 探针里 T11/T13 的题干就是说明的要求句），数据于是不再是自然语言。
        所以要求句一律不进说明：禁令由程序侧闸门（`no_answer_leak` 等）兜，
        「题干里有没有夹要求」由 L4.6 往返检验判（那是 LLM 的事）。
        """
        character = make_character(meanings={"mean_h": ["\u62fc\u97f3hao4"]})
        spec = build_reading(character, rng=random.Random(1), difficulty=Difficulty(), index=make_index([character]))
        assert spec is not None
        self.assertEqual(
            spec.instruction,
            "Ask the reader how the character \u300c\u597d\u300d is pronounced.",
        )
        self.assertNotIn("hao4", spec.instruction)
        self.assertIn("hao4", spec.accepted)


class DifficultyTests(unittest.TestCase):
    def test_rejects_out_of_range(self) -> None:
        """越界难度当场报错，不静默降级。"""
        with self.assertRaises(ValueError):
            Difficulty(hops=0)
        with self.assertRaises(ValueError):
            Difficulty(constraints=9)
        with self.assertRaises(ValueError):
            Difficulty(distractor_depth="ultra")

    def test_depth_values(self) -> None:
        """三档具体距离 + 一个分布档 `mixed` 是全部合法取值。"""
        self.assertEqual(DISTRACTOR_DEPTHS, ("near", "mid", "far", "mixed"))

    def test_mixed_depth_resolves_to_concrete_tiers(self) -> None:
        """`mixed` 必须落成具体档位，且抽样覆盖 near/mid/far（分布档，不是第四种距离）。"""
        rng = random.Random(20261006)
        seen = {resolve_depth(MIXED_DEPTH, rng) for _ in range(200)}
        self.assertEqual(seen, {"near", "mid", "far"})
        self.assertEqual(resolve_depth("far", rng), "far")

    def test_mixed_depth_is_reproducible(self) -> None:
        """同一随机源抽出的档位序列必须一致（全链路可追溯）。"""
        first_rng, second_rng = random.Random(7), random.Random(7)
        first = [resolve_depth(MIXED_DEPTH, first_rng) for _ in range(20)]
        second = [resolve_depth(MIXED_DEPTH, second_rng) for _ in range(20)]
        self.assertEqual(first, second)

    def test_mixed_depth_is_recorded_per_spec(self) -> None:
        """`build_all` 必须把 mixed 落定后的真实档位写进 TaskSpec.resolved_depth。"""
        character = make_character(stroke=7)
        specs = build_all(
            character,
            rng=random.Random(3),
            index=make_index([character]),
            difficulty=Difficulty(distractor_depth=MIXED_DEPTH),
        )
        self.assertTrue(specs)
        self.assertTrue(all(spec.resolved_depth in {"near", "mid", "far"} for spec in specs))
        self.assertEqual(len({spec.resolved_depth for spec in specs}), 1)  # 同一个字只抽一次

    def test_every_task_declares_a_machine_facing_instruction(self) -> None:
        """每个题型都必须给出**英文**的机器面出题说明——程序侧文本一律英文，模型抄不进
        中文题干（实测 instruction 写中文时 2B 一字不改地把它当题干输出）。"""
        character = make_character(
            zi="\u3400",
            ids=("\u2ff1\u535d\u4e00",),
            standard_form="\u4e18",
            variants=("\u4e18",),
            refs={"zh": {"page": 11, "number": 1}},
            meanings={"mean_h": ["\u62fc\u97f3yi1<br>\u8868\u793a\u4e00\u4e2a\u610f\u601d\u3002"]},
        )
        index = make_index([character])
        checked = 0
        for task_id, constructor in CONSTRUCTORS.items():
            spec = constructor(character, rng=random.Random(1), difficulty=Difficulty(), index=index)
            if spec is None:
                continue
            checked += 1
            self.assertGreaterEqual(len(spec.instruction), 40, f"{task_id} 的出题说明太短")
            for mark in ("\u3002", "\uff0c", "\uff1b", "\uff1a"):
                self.assertNotIn(mark, spec.instruction, f"{task_id} 的出题说明里出现中文句读 {mark!r}")
            # 说明**只说问什么**：以 Ask the reader 开头、只有一个句子。
            # 实测多出来的那几句（「List every candidate」「No digit may appear in the stem」）会被
            # 2B 翻译成中文题干里的要求句，产出就不再是自然语言。
            self.assertTrue(
                spec.instruction.startswith("Ask the reader"), f"{task_id} 的出题说明不是「问什么」句"
            )
            self.assertEqual(
                spec.instruction.count(". "), 0, f"{task_id} 的出题说明不止一句话（多半夹了写作要求）"
            )
            self.assertTrue(spec.instruction.endswith("."), f"{task_id} 的出题说明没有收尾句号")
        self.assertGreater(checked, 0)


class RegistryTests(unittest.TestCase):
    def test_only_filter(self) -> None:
        """白名单只产出指定题型。"""
        character = make_character()
        specs = build_all(character, rng=random.Random(1), index=make_index([character]), only=("T1_stroke_total",))
        self.assertEqual([s.task_id for s in specs], ["T1_stroke_total"])

    def test_all_designed_task_ids_are_registered(self) -> None:
        """README §6.2 的任务族必须全部注册（T2 两分已删除）。"""
        expected = [
            "T10_odd_one_out",
            "T11_stroke_extreme",
            "T12_ids_transform",
            "T13_ids_reverse",
            "T14_unicode_codepoint",
            "T15_unicode_bytes",
            "T16_unicode_reverse",
            "T1_stroke_total",
            "T3_ids_forward",
            "T4_reading",
            "T5_standard_form",
            "T5_variant",
            "T6_zh_ref",
            "T7_meaning_source",
            "T8_constraint",
            "T9_component_chain",
        ]
        self.assertEqual(sorted(CONSTRUCTORS), expected)

    def test_build_all_returns_multiple_tasks(self) -> None:
        """一个字能同时产出多个题型（不是一次只出一种）。"""
        character = make_character(meanings={"mean_h": ["\u62fc\u97f3hao4<br>\u8868\u793a\u7f8e\u597d\u7684\u610f\u601d\u3002"]})
        specs = build_all(character, rng=random.Random(1), index=make_index([character]))
        self.assertGreaterEqual(len(specs), 3)


if __name__ == "__main__":
    unittest.main()


class IdsTransformChainTests(unittest.TestCase):
    """输入：若干字的 T12 构造。输出：断言。预期：题干里每一步替换，读者按题干重建后都对得上——
    第 i 步的「旧部件」必须是第 i-1 步重建出的那串 IDS 里该位置的叶子。
    这是「结果字有多个 IDS 候选」踩过的坑：候选[0] 未必是上一步渲染出的那串。
    """

    def test_chain_matches_reader_reconstruction(self) -> None:
        import random
        import re as _re
        from hanzi_synth.facts.index import CharacterIndex
        from hanzi_synth.facts.loader import iter_characters
        from hanzi_synth.tasks.base import Difficulty
        from hanzi_synth.tasks.reasoning import (
            _leaf_at, _position_labels, build_ids_transform, leaves, parse_tree, render, replace_leaf,
        )

        characters = list(iter_characters())
        index = CharacterIndex(characters)
        pattern = _re.compile(r"^把(.+?)的部件「(.+?)」换成「(.+?)」$")
        rng = random.Random(7)
        checked = 0
        for character in characters[:800]:
            spec = build_ids_transform(character, rng=rng, difficulty=Difficulty(hops=3), index=index)
            if spec is None:
                continue
            checked += 1
            ids = character.ids_candidates[0]
            for edit in spec.facts["edits"]:
                match = pattern.match(edit)
                self.assertIsNotNone(match, edit)
                label, old, new = match.group(1), match.group(2), match.group(3)
                tree = parse_tree(ids)
                labels = _position_labels(tree.op or "", len(tree.children))
                self.assertIn(label, labels, edit)
                child = labels.index(label)
                offset = 0
                leaf_position = None
                for position, node in enumerate(tree.children):
                    if position == child:
                        leaf_position = offset
                        break
                    offset += len(leaves(node))
                self.assertIsNotNone(leaf_position, edit)
                self.assertEqual(_leaf_at(ids, leaf_position), old, edit)
                ids = render(replace_leaf(tree, leaf_position, new))
        self.assertGreater(checked, 0)


class PerTaskStructuralInvariantTests(unittest.TestCase):
    """输入：全池前 200 个字的构造结果。输出：断言。预期：每个题型各自的**结构性判据**成立，
    而不是「L4 那 11 条通用防线过了」就算对：facts 必须与源数据一致、answer 必须是 facts 的直接值、
    唯一性 / 链式必须真成立（T8 唯一解、T9 路径可达、T10 唯一反例、T11 唯一极值、T12 链自洽）。
    """

    @classmethod
    def setUpClass(cls) -> None:
        from hanzi_synth.facts.index import CharacterIndex
        from hanzi_synth.facts.loader import iter_characters

        characters = list(iter_characters())
        cls.sample = characters[:200]
        cls.index = CharacterIndex(characters)
        cls.by_zi = {c.zi: c for c in characters if c.zi}
        cls.by_ziid = {c.ziid: c for c in characters}

    def _source(self, spec):
        """输入：TaskSpec。输出：它的来源字（facts.zi 优先，否则 provenance.seed_ziid）。"""
        zi = spec.facts.get("zi")
        if isinstance(zi, str) and zi in self.by_zi:
            return self.by_zi[zi]
        return self.by_ziid.get(str(spec.provenance.get("seed_ziid", "")))

    def test_facts_are_faithful_and_answer_is_a_direct_value(self) -> None:
        """facts 必须能在源记录里对上，answer 必须是 facts 的直接值（不是字段间推导）。"""
        import random

        from hanzi_synth.tasks import glyph, reading, relations
        from hanzi_synth.tasks import unicode as uni
        from hanzi_synth.tasks.base import Difficulty

        builders = (
            glyph.build_stroke_total, glyph.build_ids_forward, glyph.build_ids_reverse,
            reading.build_reading, relations.build_standard_form, relations.build_variant,
            relations.build_reference, relations.build_meaning_source,
            uni.build_unicode_codepoint, uni.build_unicode_bytes, uni.build_unicode_reverse,
        )
        checked = 0
        for character in self.sample:
            rng = random.Random(5)
            for builder in builders:
                spec = builder(character, rng=rng, difficulty=Difficulty(), index=self.index)
                if spec is None:
                    continue
                checked += 1
                facts, tid, answer = spec.facts, spec.task_id, spec.answer
                source = self._source(spec)
                if tid == "T1_stroke_total":
                    self.assertEqual(facts["stroke_total"], source.stroke_total, spec.task_id)
                    self.assertEqual(facts["radical"], source.radical, spec.task_id)
                    self.assertEqual(answer, str(facts["stroke_total"]))
                elif tid == "T3_ids_forward":
                    self.assertIn(answer, facts["ids"])
                    for candidate in facts["ids"]:
                        self.assertIn(candidate, source.ids_candidates)
                    self.assertEqual(facts["stroke_total"], source.stroke_total)
                elif tid == "T13_ids_reverse":
                    self.assertIn(answer, facts["ids"])
                    for candidate in facts["ids"]:
                        self.assertIn(candidate, source.ids_candidates)
                elif tid == "T4_reading":
                    self.assertEqual(answer, facts["reading"])
                elif tid == "T5_standard_form":
                    self.assertEqual(answer, facts["standard_form"])
                    self.assertEqual(source.standard_form, facts["standard_form"])
                elif tid == "T5_variant":
                    self.assertIn(answer, facts["variants"])
                    self.assertIn(answer, source.variants)
                elif tid == "T6_zh_ref":
                    self.assertEqual(answer, str(facts["page"]))
                    self.assertIn(facts["page"], {v for ref in source.refs.values() for v in ref.values()})
                elif tid == "T7_meaning_source":
                    self.assertEqual(answer, facts["source"])
                    self.assertIn(answer, facts["options"])
                elif tid == "T14_unicode_codepoint":
                    self.assertIn(answer, facts["codepoint_forms"])  # 两种写法（带/不带 U+）都合法
                    self.assertIn(f"U+{ord(source.zi):04X}", facts["codepoint_forms"])
                elif tid == "T15_unicode_bytes":
                    self.assertEqual(answer.replace(" ", ""), facts["utf8_bytes"].replace(" ", ""))
                    self.assertEqual(source.zi.encode("utf-8").hex().upper(), answer.replace(" ", ""))
                elif tid == "T16_unicode_reverse":
                    self.assertEqual(answer, facts["zi"])
                    self.assertIn(f"U+{ord(answer):04X}", facts["codepoint_forms"])
        self.assertGreater(checked, 0)

    def test_t8_answer_is_unique_in_pool(self) -> None:
        """T8 的约束必须在全池里唯一确定答案（独立复算，不信构造器的自证）。"""
        import random

        from hanzi_synth.tasks.base import Difficulty
        from hanzi_synth.tasks.reasoning import IDC_LABELS, _leaf_total, build_constraint, top_operator

        checked = 0
        for character in self.sample:
            spec = build_constraint(
                character, rng=random.Random(9), difficulty=Difficulty(constraints=3), index=self.index
            )
            if spec is None:
                continue
            checked += 1
            facts = spec.facts
            answer = self.by_zi.get(spec.answer)
            self.assertIsNotNone(answer, spec.answer)
            self.assertEqual(facts["radical"], answer.radical)
            self.assertEqual(facts["stroke_total"], answer.stroke_total)
            pool = [c for c in self.index.by_radical.get(facts["radical"], ())
                    if c.zi and c.stroke_total == facts["stroke_total"] and c.ids_candidates]
            if "structure" in facts:
                pool = [c for c in pool
                        if IDC_LABELS.get(top_operator(c.ids_candidates[0])) == facts["structure"]]
            if "component_count" in facts:
                pool = [c for c in pool if _leaf_total(c) == facts["component_count"]]
            self.assertEqual({c.zi for c in pool}, {spec.answer})
        self.assertGreater(checked, 0)

    def test_t9_path_reaches_answer(self) -> None:
        """T9 的位置链必须真能走下去，终点就是答案（读者按候选[0] 逐层下钻）。"""
        import random

        from hanzi_synth.tasks.base import Difficulty
        from hanzi_synth.tasks.reasoning import (
            _position_labels, build_component_chain, leaves, parse_tree, render,
        )

        checked = 0
        for character in self.sample:
            spec = build_component_chain(
                character, rng=random.Random(13), difficulty=Difficulty(hops=3), index=self.index
            )
            if spec is None:
                continue
            checked += 1
            path = spec.facts["path"]
            cursor = self.by_zi[spec.facts["start_zi"]]
            for position, label in enumerate(path):
                tree = parse_tree(cursor.ids_candidates[0])
                labels = _position_labels(tree.op or "", len(tree.children))
                self.assertIn(label, labels, (cursor.zi, label, cursor.ids_candidates[0]))
                child = tree.children[labels.index(label)]
                self.assertEqual(len(leaves(child)), 1, (cursor.zi, label))
                component = render(child)
                if position == len(path) - 1:
                    self.assertEqual(component, spec.answer)
                    break
                # 中间层必须是池里真实收录的字，否则读者根本走不到下一层
                cursor = self.index.lookup_char(component)
                self.assertIsNotNone(cursor, component)
        self.assertGreater(checked, 0)

    def test_t9_path_is_declared_as_must_contain(self) -> None:
        """位置描述是程序侧必须出现的串（`must_contain`）——题干不写描述，读者无从判断走哪一层
        （实测 16r 的 T9 题干只写「按照描述逐层拆解」，描述一个都没写）。"""
        import random

        from hanzi_synth.tasks.base import Difficulty
        from hanzi_synth.tasks.reasoning import build_component_chain

        checked = 0
        for character in self.sample:
            spec = build_component_chain(
                character, rng=random.Random(13), difficulty=Difficulty(hops=3), index=self.index
            )
            if spec is None:
                continue
            checked += 1
            self.assertEqual(spec.must_contain, tuple(spec.facts["path"]), character.zi)
            self.assertGreater(len(spec.must_contain), 0, character.zi)
        self.assertGreater(checked, 0)


class ChoiceTaskRequirementTests(unittest.TestCase):
    """输入：真实字表（`chars.jsonl`）。输出：断言。
    预期：凡是声明了 `options` 的题，候选都必须**原样进 `stem_must_contain`**——那是模型
          真正看到的硬约束，也是 `options_listed` 闸门的判据。只写在散文说明里模型会漏，
          实测 16b 的 T11 产出就是「下面是一些有趣的汉字。请问总笔画数最少的是哪一个？」
          （一个候选都没列，读者根本没得选）。
    """

    def test_options_are_always_demanded_in_the_stem(self) -> None:
        """四种选择题都必须把候选送进 `stem_must_contain`（题干必须逐个原样列出）。"""
        from hanzi_synth.facts.index import CharacterIndex
        from hanzi_synth.facts.loader import iter_characters
        from hanzi_synth.llm.verbalizer import payload_for
        from hanzi_synth.tasks.base import Difficulty
        from hanzi_synth.tasks.reasoning import build_odd_one_out, build_stroke_extreme
        from hanzi_synth.tasks.relations import build_meaning_source
        from hanzi_synth.tasks.unicode import build_unicode_reverse

        builders = {
            "T10_odd_one_out": build_odd_one_out,
            "T11_stroke_extreme": build_stroke_extreme,
            "T7_meaning_source": build_meaning_source,
            "T16_unicode_reverse": build_unicode_reverse,
        }
        characters = list(iter_characters())
        index = CharacterIndex(characters)
        rng = random.Random(11)
        seen: set[str] = set()
        for character in characters:
            if len(seen) == len(builders):
                break
            for task_id, constructor in builders.items():
                if task_id in seen:
                    continue
                spec = constructor(character, rng=rng, difficulty=Difficulty(), index=index)
                if spec is None:
                    continue
                seen.add(task_id)
                self.assertTrue(spec.options, f"{task_id} 是选择题却没声明 options")
                payload = payload_for(spec)
                for option in spec.options:
                    self.assertIn(
                        option,
                        payload["stem_must_contain"],
                        f"{task_id} 的候选 {option!r} 没进 stem_must_contain——模型会照抄"
                        "「下列选项」这种没有清单的句式，把候选整个漏掉（实测 16b 的 T11）",
                    )
        self.assertEqual(seen, set(builders), "样本里没凑齐这四种选择题")

def _full_coverage_pool() -> list:
    """输入：无。输出：一批**能覆盖全部 16 个题型**的假字记录 + 索引。

    预期：这批字是给「每个题型都必须声明题干闸门」这条不变式用的——覆盖不全，
          闸门漏装也测不出来（第十轮 T12 的 `_TRANSFORM_ASK_MARKERS` 定义了却没接上，
          就是这么漏的）。各字的取材理由：
          - 㐀：有 IDS / 部首 / 标准字形 / 异体 / 页码，覆盖字形与字际关系一族；
          - 好：释义里带拼音，T4 才有唯一读音可出；
          - 一 / 丁 / 七 / 三 / 上：同一部首、笔画互有重叠，给 T8 约束桶与 T10/T11 凑同伴；
          - 㐄：部首「一」+ 5 画在池里独一份，T8 的唯一解条件才成立。
    """
    return [
        make_character(
            zi="\u3400",
            ziid="U03400",
            ids=("\u2ff0\u4e00\u4e01",),
            stroke=3,
            radical="\u4e00",
            standard_form="\u4e00",
            variants=("\u4e01",),
            meanings={"mean_h": ("\u53e4\u540c\u300c\u4e00\u300d\u3002",)},
            refs={"zh": {"page": 12, "number": 3}, "meaning_src": {"\u5b57\u6d77": 1}},
        ),
        make_character(zi="\u4e01", ziid="U04E01", ids=("\u2ff0\u4e00\u4e01",), stroke=2, radical="\u4e00",
                       meanings={"mean_h": ("\u5929\u5e72\u7b2c\u56db\u4f4d\u3002",)}),
        make_character(zi="\u4e03", ziid="U04E03", ids=("\u2ff0\u4e00\u4e01",), stroke=2, radical="\u4e00",
                       meanings={"mean_h": ("\u6570\u76ee\u3002",)}),
        make_character(zi="\u4e09", ziid="U04E09", ids=("\u2ff0\u4e00\u4e01",), stroke=3, radical="\u4e00",
                       meanings={"mean_h": ("\u6570\u76ee\u3002",)}),
        make_character(zi="\u4e0a", ziid="U04E0A", ids=("\u2ff0\u4e00\u4e01",), stroke=3, radical="\u4e00",
                       meanings={"mean_h": ("\u6570\u76ee\u3002",)}),
        make_character(zi="\u597d", ziid="U0597D", ids=("\u2ff0\u5973\u5b50",), stroke=6, radical="\u5973",
                       meanings={"mean_h": ("\u62fc\u97f3hao3\u7f8e\u597d\u3002",)}),
        make_character(zi="\u5982", ziid="U05982", ids=("\u2ff0\u5973\u53e3",), stroke=6, radical="\u5973",
                       meanings={"mean_h": ("\u987a\u4ece\u3002",)}),
        make_character(zi="\u59d0", ziid="U059D0", ids=("\u2ff0\u5973\u4e14",), stroke=8, radical="\u5973",
                       meanings={"mean_h": ("\u59d0\u59d0\u3002",)}),
        make_character(zi="\u59b9", ziid="U059B9", ids=("\u2ff0\u5973\u672a",), stroke=8, radical="\u5973",
                       meanings={"mean_h": ("\u59b9\u59b9\u3002",)}),
        make_character(zi="\u3404", ziid="U03404", ids=("\u2ff0\u4e00\u4e01",), stroke=5, radical="\u4e00",
                       meanings={"mean_h": ("\u53e4\u5b57\u3002",)}),
    ]


class StemAsksGateCoverageTests(unittest.TestCase):
    """题干闸门的不变式：每个题型都必须声明 `stem_asks_any`，否则 `stem_asks` 整条跳过。

    实测第十轮真机的 T1 U0340A 就是「题干问读音、答案是笔画数」的废题：闸门空着 = 不设防。
    """

    def test_every_task_declares_a_stem_gate(self) -> None:
        """输入：覆盖全部 16 个题型的假字池。输出：无。
        预期：池子真能出满 16 个题型（覆盖不全这条测试就没意义），且每条产出的
              `stem_asks_any` 都非空。
        """
        pool = _full_coverage_pool()
        index = make_index(pool)
        seen: set[str] = set()
        for character in pool:
            for spec in build_all(
                character, rng=random.Random(7), index=index, difficulty=Difficulty(distractor_depth="far")
            ):
                seen.add(spec.task_id)
                self.assertTrue(spec.stem_asks_any, f"{spec.task_id} 没声明 stem_asks_any")
        self.assertEqual(seen, set(CONSTRUCTORS))

    def test_task_table_matches_what_constructors_declare(self) -> None:
        """`registry.TASK_STEM_ASKS` 与构造器**实际**挂上的闸门不许漂移。

        预期：这张表是复验（`spec_from_record`）补闸门用的唯一来源，一旦构造器改了闸门而表没跟，
              复验就会拿旧闸门去查——「题干问错事」的废题会重新漏过去。
        """
        pool = _full_coverage_pool()
        index = make_index(pool)
        actual: dict[str, tuple[str, ...]] = {}
        for character in pool:
            for spec in build_all(
                character, rng=random.Random(7), index=index, difficulty=Difficulty(distractor_depth="far")
            ):
                actual[spec.task_id] = spec.stem_asks_any
        self.assertEqual(set(actual), set(TASK_STEM_ASKS))
        for task_id, markers in actual.items():
            self.assertEqual(markers, TASK_STEM_ASKS[task_id], task_id)
