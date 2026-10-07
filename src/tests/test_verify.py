"""L4 机械闸门的单元测试。

判据只有两类，测试也按这两类分：**载荷里的定死串**（答案、必现串、候选）与**产出自身的形态**
（中文自然语言、数字来源可查、没被截断、没有程序侧的说法）。写法不在这里判——那是 L4.5 / L4.6
（LLM）的事，本层只做子串 / 集合比对。
"""

from __future__ import annotations

import unittest

from hanzi_synth.llm.verbalizer import Verbalized
from hanzi_synth.tasks.base import TaskSpec
from hanzi_synth.verify.checks import verify


def _spec(**overrides) -> TaskSpec:
    """输入：要覆盖的字段。输出：一道最小可过闸门的题（答案 qiu1，主语 㐀）。
    预期：不传 overrides 时 `verify` 必须给 tier A——这样每条用例的失败原因都能归到被改的那一项。
    """
    base = dict(task_id="T_test", instruction="i", facts={}, answer="qiu1", visible_subject="㐀")
    base.update(overrides)
    return TaskSpec(**base)


def _verbalized(stem: str = "「㐀」的读音是什么？", explanation: str = "「㐀」读作 qiu1。") -> Verbalized:
    """输入：题干与解析。输出：L3 产出替身。预期：默认值能过全部机械闸门。"""
    return Verbalized(ok=True, stem=stem, explanation=explanation)


class BaselineTests(unittest.TestCase):
    def test_clean_record_passes_every_gate(self) -> None:
        """基线：干净产出必须是 tier A，且 failures 为空。"""
        result = verify(_spec(), _verbalized())
        self.assertEqual(result.tier, "A")
        self.assertEqual(result.failures, ())

    def test_schema_failure_short_circuits(self) -> None:
        """槽位为空：当场判 schema 不合格，不再往下判（下游判据都假设槽位有内容）。"""
        result = verify(_spec(), Verbalized(ok=False, stem="", explanation=""))
        self.assertFalse(result.schema)
        self.assertEqual(result.tier, "C")
        self.assertEqual(len(result.failures), 1)


class FactFidelityTests(unittest.TestCase):
    def test_answer_must_appear_in_explanation(self) -> None:
        """答案由代码定死，解析说不出它，这道题就没有解。"""
        result = verify(_spec(), _verbalized(explanation="「㐀」读作别的音。"))
        self.assertFalse(result.answer_in_explanation)
        self.assertEqual(result.tier, "C")

    def test_answer_leak_in_stem_is_rejected(self) -> None:
        """题干不得直接送出答案（选择题除外——候选本来就要列出来）。"""
        result = verify(_spec(), _verbalized(stem="「㐀」读作 qiu1 吗？"))
        self.assertFalse(result.no_answer_leak)
        self.assertEqual(result.tier, "C")

    def test_choice_options_are_exempt_from_answer_leak(self) -> None:
        """选择题：答案就是候选之一，必然出现在题干里，所以不做 no_answer_leak。"""
        spec = _spec(answer="㐀", options=("㐀", "㐁"), visible_subject="")
        result = verify(spec, _verbalized(stem="下面两个候选里，哪个读 qiu1？㐀、㐁。", explanation="答案是「㐀」。"))
        self.assertTrue(result.no_answer_leak)

    def test_choice_answer_stated_a_second_time_is_a_leak(self) -> None:
        """选择题的答案只准出现一次（候选名单里那次）。题干再说一遍「应选它」就是送答案——
        实测 T10 写「其中只有「㐀」的部首不是「風」，因此应选出「㐀」」，读者照抄就算答对。"""
        spec = _spec(answer="㐀", options=("㐀", "㐁"), visible_subject="")
        result = verify(
            spec,
            _verbalized(
                stem="下面两个候选里，哪个读 qiu1？㐀、㐁。其中只有「㐀」读 qiu1，因此应选㐀。",
                explanation="答案是「㐀」。",
            ),
        )
        self.assertFalse(result.no_answer_leak)
        self.assertEqual(result.tier, "C")

    def test_answer_declared_as_a_required_string_may_appear_once(self) -> None:
        """非选择题里答案若被题型声明成必现串（题干本来就得写出它），允许出现一次。"""
        spec = _spec(answer="㐀", visible_subject="", must_contain=("㐀",))
        result = verify(spec, _verbalized(stem="「㐀」的读音是哪一个？", explanation="答案是「㐀」。"))
        self.assertTrue(result.no_answer_leak)
        self.assertEqual(result.tier, "A")

    def test_leak_note_tells_the_writer_to_list_candidates_once(self) -> None:
        """选择题的反馈必须说清怎么改（候选各列一次、别说哪个对），否则模型只会删掉候选。"""
        spec = _spec(answer="㐀", options=("㐀", "㐁"), visible_subject="")
        result = verify(
            spec,
            _verbalized(
                stem="下面两个候选里，哪个读 qiu1？㐀、㐁。其中只有「㐀」读 qiu1，因此应选㐀。",
                explanation="答案是「㐀」。",
            ),
        )
        note = dict(zip((item.split(":")[0] for item in result.failures), result.notes))["no_answer_leak"]
        self.assertIn("exactly once", note)

    def test_subject_must_appear_in_stem(self) -> None:
        """读者看不见是哪个字，就答不了题。"""
        result = verify(_spec(), _verbalized(stem="这个字的读音是什么？"))
        self.assertFalse(result.subject_in_stem)
        self.assertEqual(result.tier, "C")

    def test_options_must_be_listed_one_by_one(self) -> None:
        """选择题必须把候选逐个列进题干——只写「从下列选项中选」读者根本没得选。"""
        spec = _spec(answer="㐀", options=("㐀", "㐁"), visible_subject="")
        result = verify(spec, _verbalized(stem="下面几个候选里，哪个读 qiu1？", explanation="答案是「㐀」。"))
        self.assertFalse(result.options_listed)
        self.assertEqual(result.tier, "C")

    def test_hidden_subject_must_stay_hidden(self) -> None:
        """反向题的隐藏主语一旦写进题干，题目就退化成查表题。"""
        spec = _spec(hidden_subject="㐆")
        result = verify(spec, _verbalized(stem="「㐆」的读音是什么？"))
        self.assertFalse(result.no_hidden_subject)
        self.assertEqual(result.tier, "C")


class OutputFormTests(unittest.TestCase):
    def test_stem_must_be_chinese(self) -> None:
        """程序侧文本一律英文；题干里一个汉字都没有，只可能是模型把载荷抄了回来。"""
        result = verify(_spec(), _verbalized(stem="Ask the reader how it is pronounced."))
        self.assertFalse(result.stem_is_chinese)
        self.assertEqual(result.tier, "C")

    def test_numbers_must_come_from_the_payload(self) -> None:
        """产出里的阿拉伯数字必须在载荷里找得到，否则就是凭空造的（笔画数、码位最容易编）。"""
        result = verify(_spec(), _verbalized(stem="「㐀」一共 7 画，读音是什么？"))
        self.assertFalse(result.numbers_from_facts)
        self.assertEqual(result.tier, "C")

    def test_truncated_output_is_rejected(self) -> None:
        """finish_reason=length 说明产出被砍断，不是完整的一句话。"""
        result = verify(_spec(), Verbalized(ok=True, stem="「㐀」的读音是什么？", explanation="「㐀」读作 qiu1。",
                                            finish_reason="length"))
        self.assertFalse(result.not_truncated)
        self.assertEqual(result.tier, "C")

    def test_copying_the_meaning_source_is_rejected(self) -> None:
        """释义原文是**素材**：连续照抄十个汉字及以上判不合格，正常改写不误杀。"""
        source = "这个字表示一种古代的礼器名称用来盛放祭品"
        spec = _spec(facts={"meaning_source": source})
        copied = verify(spec, _verbalized(stem=f"「㐀」的读音是什么？{source}"))
        self.assertFalse(copied.no_source_copy)
        rewritten = verify(spec, _verbalized(stem="古代有一种用来盛放祭品的礼器，这个字指的就是它，它读什么？"))
        self.assertTrue(rewritten.no_source_copy)


class InternalTermsTests(unittest.TestCase):
    """产出必须**上下文无关**：读者手里只有这道题本身，不知道什么叫「题干」「载荷」「槽位」「IDS」。

    实测：2B 会把 instruction 里的字段名（`stem_must_contain` / `facts.conditions`）意译成
    「题干描述的…」「提供的汉字」「下列字形」，也会把英文缩写连夹注一起抄进中文题干
    （「请说明字符㐀的 IDS（ideographic description sequence）分解方式」）。
    """

    def test_program_side_wording_in_stem_is_rejected(self) -> None:
        """题干里冒出程序侧的说法 / 指代出题过程的说法 → 不合格，逐个词都要抓到。

        两类：程序侧的行话（题干 / 载荷 / 槽位 / 校验器 / 必现串 / IDS …），以及指代出题过程或
        指代一份并不存在的东西的说法（本题 / 题目要求 / 提供的汉字 / 提供的资料）。
        实测真机产出里两类都出现过：T9「题干描述的位置链为…」、T3「其中包含了必现串 '㐀'」、
        T16「请在提供的汉字中找出…」、T4「但题目要求用拼音标注为 xi3」。
        """
        for term in ("题干", "载荷", "槽位", "校验器", "必现串", "本题", "题目要求",
                     "提供的汉字", "提供的资料", "IDS", "payload", "facts", "instruction"):
            with self.subTest(term=term):
                result = verify(_spec(), _verbalized(stem=f"「㐀」的读音是什么？{term}"))
                self.assertFalse(result.no_internal_terms, term)
                self.assertEqual(result.tier, "C", term)

    def test_program_side_wording_in_explanation_is_rejected(self) -> None:
        """解析里同样不许出现——产出是给读者的，不是给程序的。"""
        result = verify(_spec(), _verbalized(explanation="「㐀」读作 qiu1，这是本题的题干。"))
        self.assertFalse(result.no_internal_terms)
        self.assertEqual(result.tier, "C")

    def test_abbreviation_case_is_ignored(self) -> None:
        """大小写都算：写成 ids 一样是程序侧的说法。"""
        result = verify(_spec(), _verbalized(stem="「㐀」的 ids 拆解是什么？"))
        self.assertFalse(result.no_internal_terms)

    def test_clean_text_has_no_internal_terms(self) -> None:
        """基线：干净产出这一条必须过，否则这条闸门就是在误杀。"""
        self.assertTrue(verify(_spec(), _verbalized()).no_internal_terms)


class FieldNameTests(unittest.TestCase):
    """载荷**键名**不得出现在正文里：键名是程序侧的英文标识符，读者只该看到键对应的**值**。"""

    def test_facts_key_must_not_appear_in_prose(self) -> None:
        """实测 T1 解析写「「㐀」的 stroke_total 为 5」——字段名漏进了散文。"""
        spec = _spec(facts={"stroke_total": 5})
        result = verify(spec, _verbalized(explanation="「㐀」的 stroke_total 为 5，读音 qiu1。"))
        self.assertFalse(result.no_field_names)
        self.assertEqual(result.tier, "C")

    def test_facts_value_is_allowed(self) -> None:
        """比的是**键**不是**值**：值本来就该出现在正文里，写出来才叫忠实。"""
        spec = _spec(facts={"stroke_total": 5}, answer="5")
        result = verify(spec, _verbalized(stem="「㐀」有几画？", explanation="「㐀」有 5 画。"))
        self.assertTrue(result.no_field_names)
        self.assertEqual(result.tier, "A")

    def test_payload_field_name_is_rejected(self) -> None:
        """载荷字段名（`stem_must_contain` 这类）同样不得进正文——它比 facts 的键更不像人话。"""
        result = verify(_spec(), _verbalized(explanation="按 stem_must_contain 的要求，「㐀」读作 qiu1。"))
        self.assertFalse(result.no_field_names)
        self.assertEqual(result.tier, "C")

    def test_short_facts_key_is_not_gated(self) -> None:
        """`zi` 只有两个字符，容易在别的英文串里偶然命中，不入判据——只判长度 >= 3 的键名。"""
        spec = _spec(facts={"zi": "㐀"})
        result = verify(spec, _verbalized(stem="「㐀」的读音是什么？", explanation="「㐀」读作 qiu1。"))
        self.assertTrue(result.no_field_names)
        self.assertEqual(result.tier, "A")


class MarkerEnumerationTests(unittest.TestCase):
    """题干不许把 `stem_asks_any` 那张判据词表罗列出来——那是程序侧的判据，不是给读者看的清单。"""

    _MARKERS = ("读音", "读作", "怎么读", "念作", "读法", "拼音")

    def test_listing_the_marker_table_is_rejected(self) -> None:
        """实测 T4 写「读作、怎么读、念作、读法、拼音这些要求里都包含读音」。"""
        spec = _spec(stem_asks_any=self._MARKERS)
        result = verify(spec, _verbalized(stem="「㐀」读作、怎么读、念作、读法、拼音都是问读音吗？"))
        self.assertFalse(result.no_marker_enumeration)
        self.assertEqual(result.tier, "C")

    def test_a_normal_question_hits_few_markers(self) -> None:
        """正常问句命中一两个说法是闸门本来就要求的，不算罗列。"""
        spec = _spec(stem_asks_any=self._MARKERS)
        result = verify(spec, _verbalized(stem="「㐀」读作什么？"))
        self.assertTrue(result.no_marker_enumeration)
        self.assertEqual(result.tier, "A")

    def test_incidental_marker_hits_are_not_an_enumeration(self) -> None:
        """命中数多但**不相邻** = 正常措辞，不算罗列。

        实测 T16 写「请问下列哪个候选字的编码码点是 13323？…请根据码位与码点的对应关系判断
        哪一个候选字符合条件」，命中 5 个判据词但一个都不挨着——按命中总数判会误杀这道题。
        """
        spec = _spec(stem_asks_any=("码位", "码点", "对应", "哪个", "哪一个"))
        result = verify(
            spec,
            _verbalized(stem="请问下列哪个候选字的编码码点是 13323？请根据码位与码点的对应关系判断哪一个符合条件。"),
        )
        self.assertTrue(result.no_marker_enumeration)


class ForeignWordTests(unittest.TestCase):
    """正文必须是中文散文：夹进来的英文词要么是载荷里的数据，要么是标准术语。"""

    def test_stray_english_word_is_rejected(self) -> None:
        """实测 T3 写「结构 notation 字符串」——notation 既不是载荷里的值也不是标准术语。"""
        result = verify(_spec(), _verbalized(explanation="「㐀」读作 qiu1，notation 是英文词。"))
        self.assertFalse(result.no_foreign_words)
        self.assertEqual(result.tier, "C")

    def test_payload_value_in_ascii_is_allowed(self) -> None:
        """载荷里的 ASCII 值（码位、字节序列）本来就该原样出现。"""
        spec = _spec(facts={"codepoint": "U+3400"}, answer="U+3400")
        result = verify(spec, _verbalized(stem="「㐀」的码位是什么？", explanation="「㐀」的码位是 U+3400。"))
        self.assertTrue(result.no_foreign_words)
        self.assertEqual(result.tier, "A")

    def test_standard_technical_term_is_allowed(self) -> None:
        """Unicode / UTF-8 这类标准术语是中文技术写作的一部分，不该被当成英文词拦下。"""
        spec = _spec(facts={"codepoint": "U+3400"}, answer="U+3400")
        result = verify(spec, _verbalized(stem="「㐀」的 Unicode 码位是什么？", explanation="「㐀」的码位是 U+3400。"))
        self.assertTrue(result.no_foreign_words)
        self.assertEqual(result.tier, "A")


class EscapeArtifactTests(unittest.TestCase):
    """正文里不得有字面的反斜杠：实测 2B 会吐出字面的 \\n，发布出去读者看到的是「…的字。\\n选项…」。"""

    def test_literal_backslash_is_rejected(self) -> None:
        result = verify(_spec(), _verbalized(stem="「㐀」的读音是什么？\\n请作答。"))
        self.assertFalse(result.no_escape_artifacts)
        self.assertEqual(result.tier, "C")

    def test_plain_prose_passes(self) -> None:
        result = verify(_spec(), _verbalized())
        self.assertTrue(result.no_escape_artifacts)


class FeedbackNoteTests(unittest.TestCase):
    """`notes` 是 failures 的**英文**说法，只回灌给措辞模型。

    预期：与 failures 一一对应（同序）；模板是英文——中文失败原因回灌会被 2B 整段抄进题干
          （实测「题干列出了…」「请在提供的汉字中找出…」都来自上一轮的中文反馈）。
          提示里可以出现载荷里的汉字（缺的串、答案），那是数据不是模板。
    """

    def test_clean_output_has_no_notes(self) -> None:
        self.assertEqual(verify(_spec(), _verbalized()).notes, ())

    def test_notes_align_with_failures_and_are_english(self) -> None:
        result = verify(_spec(), _verbalized(stem="What is the reading?"))
        self.assertEqual(len(result.notes), len(result.failures))
        paired = dict(zip((item.split(":")[0] for item in result.failures), result.notes))
        self.assertEqual(
            paired["stem_is_chinese"],
            "The stem contains no Chinese characters. Write it as Chinese prose.",
        )

    def test_missing_required_string_is_named_in_the_note(self) -> None:
        """反馈必须指名道姓说出缺哪个串，否则模型只能猜（实测 T12/T13 反复在同一处摔）。"""
        result = verify(_spec(must_contain=("hao3",)), _verbalized())
        self.assertTrue(result.failures)
        self.assertTrue(any("hao3" in note for note in result.notes), result.notes)


if __name__ == "__main__":
    unittest.main()
