"""离线程序化渲染的测试：模板池覆盖、句式空间下限、确定性与闸门。

输入：`tests.support.full_coverage_pool()` 造的假字池（不读磁盘）。
输出：断言。
预期：模板池是「不调模型也能出题」那条路的地基，坏了要在这里炸——不能等到跑全量才发现。
"""

from __future__ import annotations

import random
import re
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from hanzi_synth.render import offline as O
from hanzi_synth.render.generate import generate
from hanzi_synth.render.stats import census
from hanzi_synth.tasks.base import Difficulty
from hanzi_synth.tasks.registry import CONSTRUCTORS, build_all
from hanzi_synth.verify.checks import (
    Verification,
    _INTERNAL_TERMS,
    answer_leak_limit,
    normalize_for_match,
)

from tests.support import full_coverage_pool, make_index

# 「每种题型至少能渲染出多少种不同句式」的下限：主人定的指标。
MIN_DISTINCT_STEMS = 500


def _sample_variants(space: int, cap: int = 240) -> range:
    """输入：句式空间大小 + 采样上限。输出：variant 序号序列。
    预期：**从头顺序取**，不按 `space // N` 跳步——混合进制的步长一旦是骨架数的倍数，
          每个样本都会落在同一根骨架上，看着采了 40 个，实际只测到 1 条模板。
    """
    return range(min(space, cap))


def _one_spec_per_type() -> dict:
    """输入：无。输出：{题型: 一条 TaskSpec}。预期：覆盖全部题型；少一个就断言失败。"""
    pool = full_coverage_pool()
    index = make_index(pool)
    rng = random.Random(7)
    found: dict = {}
    for character in pool:
        for spec in build_all(character, rng=rng, difficulty=Difficulty(), index=index):
            found.setdefault(spec.task_id, spec)
    return found


def _offline_specs() -> dict:
    """输入：无。输出：{题型: 一条 TaskSpec}，**只含模板池覆盖的题型**。
    预期：释义类题（`offline.LLM_ONLY_TASK_IDS`）不在模板池里，逐型断言前先摘掉它们，
          否则「至少渲染出一个变体」这类断言会在没有模板的题型上误炸。
    """
    return {
        task_id: spec
        for task_id, spec in _one_spec_per_type().items()
        if task_id not in O.LLM_ONLY_TASK_IDS
    }


class NoAnnouncementPrefixTests(unittest.TestCase):
    """题干不许带**播报式前缀**：前缀不提出任何新问题，只是把句式数凑大。

    主人问过「为什么要加前缀？不加前缀不也能看出是一个问题？」——`_OPENERS` 原先挂着
    「小测验：」「来一题：」「看这一题：」等 7 条播报前缀，覆盖了产出的 87.5%，
    而删掉它们之后每一条题干照样是一道完整的问题。所以前缀清空，句式数由骨架自己挣
    （T8 因此补到 130 条骨架）。这条单测挡住「哪天又把凑数前缀加回来」。
    """

    # 播报前缀：**短标签 + 冒号**，说了「这是一道题」却没说「问的是什么」。
    # 只认「标签：」这一种形状——「给定条件{conditions}，答案是什么字？」是自然句，
    # 不是播报（它把条件当句子成分说了出来）。
    _ANNOUNCEMENTS = (
        "小测验：", "来一题：", "看这一题：", "试着回答：", "汉字小考：", "练一练：",
        "下一题：", "问一个字：", "问个简单的：", "问码点：", "这字几画：", "这字的",
        "找字：", "按条件猜字：", "操作步骤：", "已知变换：", "条件如下：", "已知：",
        "给定条件：", "按这些线索找字：", "问 UTF-8 字节：", "请回答：",
    )

    def test_openers_are_empty(self) -> None:
        """`_OPENERS` 只能有空串：前缀是凑数，不是句式。"""
        self.assertEqual(O._OPENERS, ("",))

    def test_no_core_starts_with_an_announcement(self) -> None:
        """骨架不许以播报词开头（播报词在题干里不承担任何信息）。"""
        offenders = [
            f"{task_id} :: {core}"
            for task_id, cores in O._CORES.items()
            for core in cores
            if core.startswith(self._ANNOUNCEMENTS)
        ]
        self.assertEqual(offenders, [], f"这些骨架还在播报：{offenders}")


class ClausePlaceholderTests(unittest.TestCase):
    """从句型占位符只能当**整句**用，不许被当成名词短语去修饰后面的词。

    输入：`O._CORES` 骨架池。输出：断言。
    预期：`{conditions}`（值是「部首是「一」，总笔画是 5 画」）与 `{edits}`（值是「把上边的
          部件「九」换成「米」」）装的是**并列从句**。从句被当名词用时拼出的是「总笔画是 4 画
          这些条件」「总笔画是 4 画两点」这种残片——全量产出里实测 37 条与 11 条。
          从句必须被**标点或引出它的虚词**夹住，所以两侧的字符都要落在白名单里；
          改法是把从句挪到冒号后面（「有一个字满足这些条件：{conditions}。」），不是删句。
    """

    # 从句**后面**允许出现的字符：把从句收住的标点，或「的/吗」这类把它当成分句的虚词。
    _OK_AFTER = frozenset("，。？！：；、—…→ 的了吗呢\n")
    # 从句**前面**允许出现的字符：引出从句的标点与虚词（满足 / 符合 / 已知 / 具备 / 是 / 如果…）。
    _OK_BEFORE = frozenset("：，？→ 足合知备是果道据若再\n") | {""}

    def _clause_sites(self):
        """输入：无。输出：[(题型, 骨架, 槽位, 前一个字符, 后一个字符)]。
        预期：只扫两个从句槽位——`{subject}` / `{options}` 装的是名词，两侧接字面词是正常的。
        """
        for task_id, cores in O._CORES.items():
            for core in cores:
                for slot in ("{conditions}", "{edits}"):
                    start = 0
                    while True:
                        pos = core.find(slot, start)
                        if pos < 0:
                            break
                        head = core[pos - 1:pos] if pos else ""
                        tail = core[pos + len(slot):pos + len(slot) + 1]
                        yield task_id, core, slot, head, tail
                        start = pos + 1

    def test_clause_placeholders_are_delimited(self) -> None:
        """从句槽位两侧不许粘字面词：粘上就是「画这些条件」那类残片。"""
        offenders = [
            f"{task_id} :: {core}（{slot} 前 {head!r} 后 {tail!r}）"
            for task_id, core, slot, head, tail in self._clause_sites()
            if head not in self._OK_BEFORE or tail not in self._OK_AFTER
        ]
        self.assertEqual(offenders, [], f"从句占位符两侧粘了字面词：{offenders}")


class TemplatePoolTests(unittest.TestCase):
    """输入：假字池。输出：断言。预期：每个题型都有骨架与解析模板，且句式空间够大。"""

    def test_every_registered_type_has_templates(self) -> None:
        """题型在 `CONSTRUCTORS` 里注册了，模板池就必须有它的骨架与解析——缺一个就渲染不出题。
        例外只有释义类题：它们按 AGENTS.md §六 走 LLM 路径，模板池里**故意**没有它们。
        """
        for task_id in CONSTRUCTORS:
            if task_id in O.LLM_ONLY_TASK_IDS:
                continue
            self.assertTrue(O._CORES.get(task_id), f"{task_id} 没有问法骨架")
            self.assertTrue(O._EXPLANATIONS.get(task_id), f"{task_id} 没有解析模板")

    def test_llm_only_types_have_no_offline_pool(self) -> None:
        """释义类题**不许**在模板池里留骨架：`has_templates` 说没有，就真的一个变体都渲染不出。
        这条同时挡住两种漂移——有人把骨架加回来（纪律破了），或名单与模板池对不上（死代码）。
        """
        self.assertEqual(
            {task_id for task_id in CONSTRUCTORS if not O.has_templates(task_id)},
            set(O.LLM_ONLY_TASK_IDS),
            "`LLM_ONLY_TASK_IDS` 与「模板池里没有骨架的题型」必须完全一致",
        )
        for task_id in O.LLM_ONLY_TASK_IDS:
            self.assertFalse(O._CORES.get(task_id), f"{task_id} 不该有问法骨架")
            self.assertFalse(O._EXPLANATIONS.get(task_id), f"{task_id} 不该有解析模板")

    def test_every_context_key_is_used_by_a_template(self) -> None:
        """`_context` 交出去的每个占位符都必须真被模板用到——没人用的键就是删题型时留下的死键。

        预期：这条挡住「模板删了、`_context` 还留着它的键」这种半截清理（本轮删 T13 / T17 时
              `{reading_display}` 就成了死键，靠这条才发现）。
        """
        spec = _offline_specs()["T1_stroke_total"]
        keys = set(O._context(spec, O._QUOTES[0], "、"))
        templates = [
            *O._OPTION_STYLES,
            *(item for group in O._CORES.values() for item in group),
            *(item for group in O._EXPLANATIONS.values() for item in group),
        ]
        blob = "\n".join(templates)
        unused = sorted(key for key in keys if "{" + key + "}" not in blob)
        self.assertEqual(unused, [], f"这些上下文键没有任何模板在用：{unused}")

    def test_llm_only_specs_render_nothing(self) -> None:
        """释义类题的 spec 交到渲染器上必须一条都渲染不出（`variant_space` 为 0）。"""
        specs = _one_spec_per_type()
        for task_id in O.LLM_ONLY_TASK_IDS:
            spec = specs.get(task_id)
            if spec is None:
                continue
            self.assertEqual(O.variant_space(spec), 0, task_id)
            self.assertIsNone(O.render(spec, 0), task_id)

    def test_variant_space_is_at_least_500_per_type(self) -> None:
        """主人定的硬指标：每型至少能拼出 500 种不同句式（骨架 × 引号 × 候选写法 × 收尾；开场已清空）。"""
        specs = _offline_specs()
        self.assertEqual(
            set(specs), set(CONSTRUCTORS) - set(O.LLM_ONLY_TASK_IDS), "假字池没凑齐全部题型"
        )
        for task_id, spec in specs.items():
            self.assertGreaterEqual(O.variant_space(spec), MIN_DISTINCT_STEMS, task_id)

    def test_no_two_templates_in_a_type_are_the_same(self) -> None:
        """同型模板去标点后不许重复：重复 = 句式数注水，`variant_space` 报的 600 里有重复项。

        预期：判据是「剥掉标点与空白后仍不同」——「{subject}念作什么？」与「{subject}念作什么。」
              只差一个句号，读者看到的是同一句，不能算两种句式。实测改前有 10 组这种重复
              （T4 四组、T5 两型各一组、T6 / T8 / T15 / T16 各一组），每型骨架数虚高 1~4 条。
        """
        strip = lambda text: re.sub(r"[，。？！：；、（）「」『』“”《》\s]+", "", text)  # noqa: E731
        for pool_name, pool in (("骨架", O._CORES), ("解析", O._EXPLANATIONS)):
            for task_id, items in pool.items():
                seen: dict[str, int] = {}
                for index, text in enumerate(items):
                    key = strip(text)
                    if key in seen:
                        self.fail(
                            f"{task_id} 的{pool_name}第 {seen[key]} 条与第 {index} 条重复：{text}"
                        )
                    seen[key] = index

    def test_render_is_deterministic(self) -> None:
        """确定性：同一条 spec、同一个 variant，渲染两次必须一模一样（否则产出不可复现）。"""
        for task_id, spec in _offline_specs().items():
            space = O.variant_space(spec)
            for variant in (0, space // 3, space // 2, space - 1):
                self.assertEqual(O.render(spec, variant), O.render(spec, variant), task_id)

    def test_rendered_stems_hit_the_task_marker(self) -> None:
        """渲染出来的题干必须命中题型声明的判据词，否则 `stem_asks` 整条闸门形同虚设。"""
        for task_id, spec in _offline_specs().items():
            markers = spec.stem_asks_any
            self.assertTrue(markers, f"{task_id} 没声明 stem_asks_any")
            hits = 0
            for variant in _sample_variants(O.variant_space(spec)):
                rendered = O.render(spec, variant)
                if rendered is None:
                    continue
                stem = rendered[0]
                self.assertTrue(any(item in stem for item in markers), f"{task_id}: {stem}")
                hits += 1
            self.assertGreater(hits, 0, f"{task_id} 一个变体都渲染不出来")

    def test_rendered_text_carries_no_internal_term(self) -> None:
        """产出必须上下文无关：渲染出来的句子里不许出现程序侧的说法（题干 / 载荷 / facts 一类）。"""
        for task_id, spec in _offline_specs().items():
            for variant in _sample_variants(O.variant_space(spec)):
                rendered = O.render(spec, variant)
                if rendered is None:
                    continue
                blob = (rendered[0] + "\n" + rendered[1]).lower()
                for term in _INTERNAL_TERMS:
                    self.assertNotIn(term.lower(), blob, f"{task_id} 的产出里出现内部词 {term!r}")

    def test_answer_never_leaks_into_a_non_choice_stem(self) -> None:
        """非选择题的答案不许出现在题干里：`render` 直接把泄题的变体判成 None。"""
        for task_id, spec in _offline_specs().items():
            if spec.options or not spec.answer:
                continue
            # 泄题判据只读题干，所以只枚举**题干空间**：解析写法占高位，乘进来只是把同一批
            # 题干重复验 3~5 遍。
            for variant in range(O.stem_space(spec)):
                rendered = O.render(spec, variant)
                if rendered is None:
                    continue
                self.assertNotIn(str(spec.answer), rendered[0], f"{task_id} 泄题")


class EmptyPlaceholderTests(unittest.TestCase):
    """模板**用到的**占位符取到空串时，`render` 必须判 None，不许拼出缺成分的句子。

    输入：假字池里的一条真 spec，用 `dataclasses.replace` 把某个槽位改成空值。输出：断言。
    预期：这是「字海内部编号记录（写不出字形）」那批记录在渲染层的护栏。数据侧已经由 loader
          的准入闸门挡住，这里钉的是渲染器自己的不变式——将来换数据源、加题型，只要哪个槽位空了，
          句子就渲染不出来（由 `generate` 计入 `unrenderable` 并写进落选文件），不会漏进数据集。
    """

    def test_empty_subject_renders_nothing(self) -> None:
        """主语空了 = 句子里没有要问的那个字，整题不成立。"""
        spec = replace(_offline_specs()["T1_stroke_total"], visible_subject="")
        for variant in range(O.stem_space(spec)):
            self.assertIsNone(O.render(spec, variant), f"variant {variant} 拼出了空主语题干")

    def test_empty_clause_renders_nothing(self) -> None:
        """T8 的条件从句空了，就不许拼出「有一个字满足这些条件：。」这种残句。"""
        spec = _offline_specs()["T8_constraint"]
        facts = dict(spec.facts)
        facts["conditions"] = ""
        spec = replace(spec, facts=facts)
        for variant in range(O.stem_space(spec)):
            self.assertIsNone(O.render(spec, variant), f"variant {variant} 拼出了空从句题干")

    def test_empty_direction_renders_nothing(self) -> None:
        """T11 的 `{direction_word}` 来自 facts.direction，取不到时「笔画最少」整句都不成立。"""
        spec = _offline_specs()["T11_stroke_extreme"]
        facts = dict(spec.facts)
        facts["direction"] = ""
        spec = replace(spec, facts=facts)
        for variant in range(O.stem_space(spec)):
            self.assertIsNone(O.render(spec, variant), f"variant {variant} 拼出了空方向题干")

    def test_optional_note_may_be_empty(self) -> None:
        """补充说明（`{reading_note}` 一类）**允许**为空：主答案已经给出等价写法时按设计不出现。

        预期：这条挡住护栏误伤——把「可空」也当成「渲染不出」，会让 T4 这类题整型消失。
        """
        spec = _offline_specs()["T4_reading"]
        facts = dict(spec.facts)
        facts["reading_display"] = [spec.answer]
        spec = replace(spec, facts=facts)
        rendered = [O.render(spec, variant) for variant in range(O.stem_space(spec))]
        self.assertTrue(all(item is not None for item in rendered), "补充说明为空时整题被误判渲染不出")


class LeakRuleParityTests(unittest.TestCase):
    """渲染器与闸门必须共用同一条「答案在题干里能出现几次」判据。

    输入：假字池里每型一条 spec。输出：断言。
    预期：两处算法分叉过一次——闸门按 1/0 次判废，渲染器不看这条，于是渲染完直接被拒；
          全量实测 38 条 (题型, 字) 因此一条数据都出不来（其中 32 条换一条不撞答案的句式就能出，
          `synthesis/out/probe_leak_rescue.py`）。这条把「渲染器只会挑过得了闸门的句式」钉死。
    """

    def test_render_never_exceeds_the_gate_limit(self) -> None:
        """渲染出来的每一句都必须满足闸门的答案出现次数上限。"""
        for task_id, spec in _offline_specs().items():
            for variant in _sample_variants(O.stem_space(spec)):
                rendered = O.render(spec, variant)
                if rendered is None:
                    continue
                stem_norm = normalize_for_match(rendered[0])
                for item in spec.accepted:
                    needle = normalize_for_match(str(item))
                    if not needle:
                        continue
                    self.assertLessEqual(
                        stem_norm.count(needle),
                        answer_leak_limit(spec, item),
                        f"{task_id} 渲染出的题干超过了闸门允许的答案出现次数：{rendered[0]}",
                    )

    def test_colliding_choice_answer_switches_wording(self) -> None:
        """选择题的答案撞上模板用词时，渲染器要换一条不撞的句式，而不是让整条落选。

        预期：T16 的候选呈现方式里有「从 {options} 里挑一个。」——答案恰好是「从」时该变体的
              题干里「从」出现两次（一次是候选、一次是模板用词），必须判 None 换句式；
              其余候选呈现方式不受影响，所以这一型仍然出得来。
        """
        spec = replace(
            _offline_specs()["T16_unicode_reverse"],
            answer="从",
            answer_aliases=(),
            surface_forms=(),
            options=("从", "好", "女", "子"),
        )
        rendered = [O.render(spec, variant) for variant in range(O.stem_space(spec))]
        good = [item for item in rendered if item is not None]
        self.assertTrue(good, "答案撞上模板用词时整型都渲染不出来了")
        self.assertLess(len(good), len(rendered), "撞词的变体一条都没被判 None，这条测试白测")
        for stem, _explanation in good:
            self.assertLessEqual(
                normalize_for_match(stem).count("从"),
                answer_leak_limit(spec, "从"),
                f"撞词的题干漏出来了：{stem}",
            )


class ExplanationVarietyTests(unittest.TestCase):
    """解析写法必须真的被用到：每型 3~5 条解析骨架，一条都不许是死代码。

    输入：`O._EXPLANATIONS` + 假字池。输出：断言。
    预期：解析的混合进制原先接在题干进制之后，`variant` 走到那一步时已经被除到 0——六种开场与
          四种收尾永远只取第 0 个（实测全量 739,005 条的解析 100% 是空串开场），每型 3~5 条解析
          骨架也只用第 0 条，「不同解析」那个数字只反映引号写法。现在解析占一条独立高位轴，
          穷举一遍必须**每条**解析骨架都渲染出来过。
    """

    def test_every_explanation_body_is_reachable(self) -> None:
        """穷举整段空间，解析骨架的条数一条都不能少。"""
        for task_id, spec in _offline_specs().items():
            bodies = O._EXPLANATIONS[task_id]
            rendered = O.render_variants(spec, O.variant_space(spec))
            explanations = {explanation for _stem, explanation in rendered}
            self.assertGreaterEqual(
                len(explanations), len(bodies),
                f"{task_id} 只渲染出 {len(explanations)} 种解析，池子里有 {len(bodies)} 条",
            )

    def test_explanation_is_part_of_the_variant_space(self) -> None:
        """解析写法是空间的一部分：空间必须等于「题干空间 × 解析骨架条数」，不许虚报也不许漏乘。"""
        for task_id, spec in _offline_specs().items():
            self.assertEqual(
                O.variant_space(spec), O.stem_space(spec) * O.explanation_space(spec), task_id
            )
            self.assertEqual(O.explanation_space(spec), len(O._EXPLANATIONS[task_id]), task_id)


class RenderIndependenceTests(unittest.TestCase):
    """输入：假字池（把 loader 换掉，不读磁盘）。输出：断言。

    预期：钉住「各题型的随机源互不干扰」——同一 (题型, 字) 的产出不许因为池子里多一型、少一型而变。
          实测过反面（`synthesis/out/_probe_coupling_a/b.jsonl`）：`generate` 原先用**一个**跨
          (字, 题型) 共享的 random.Random，删掉 T1 之后 T10 的候选 0/343 相同、T11 0/240、T16 0/354
          ——数据集的内容取决于「当时池子里有哪几型」，那份落盘数据就不再是当前代码能复现的了。
    """

    def _render(self, pool, task_ids, out_dir: Path) -> dict:
        """输入：字池 + 题型白名单 + 输出目录。输出：{(题型, ziid): 记录}。
        预期：走真正的 `generate`（只把 `load_characters` 换成假池），拿到的就是生产路径的产出。
        """
        out = out_dir / f"render_{len(task_ids)}_{'_'.join(sorted(task_ids))[:40]}.jsonl"
        with mock.patch("hanzi_synth.render.generate.load_characters", lambda _path=None: list(pool)):
            generate(limit=100000, out_path=out, task_ids=task_ids, progress=lambda _msg: None)
        rows: dict = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[(row["task_id"], row["ziid"])] = row
        return rows

    def test_dropping_a_type_does_not_move_the_other_rows(self) -> None:
        """删掉一个题型，其余题型的产出必须逐字段不变（随机源按 (题型, 字) 独立）。"""
        pool = full_coverage_pool()
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        every = tuple(CONSTRUCTORS)
        without_t1 = tuple(tid for tid in every if tid != "T1_stroke_total")

        all_rows = self._render(pool, every, out_dir)
        fewer_rows = self._render(pool, without_t1, out_dir)
        shared = sorted(set(all_rows) & set(fewer_rows))
        self.assertTrue(shared, "两次渲染没有共有题型，这条测试白测")
        differing = [key for key in shared if all_rows[key] != fewer_rows[key]]
        self.assertEqual(differing[:3], [], f"{len(differing)} 条产出的内容被别的题型挪动了")

    def test_dropped_rows_are_kept_with_a_reason(self) -> None:
        """过不了闸门的产出不许静默消失：整条写进 `<out>.dropped.jsonl`，带 `drop_reason` 与 `failures`。

        预期：主数据集要干净（过不了的不进去），但落选样本必须能被翻出来查——「没进主数据集」
              和「丢了」是两件事。主人对回炉那条纪律是同一个要求（rejects 不许丢）。
        """
        pool = full_coverage_pool()
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        out = out_dir / "render.jsonl"
        failing = Verification(failures=("no_internal_terms: 题干里出现内部词",))
        with mock.patch(
            "hanzi_synth.render.generate.load_characters", lambda _path=None: list(pool)
        ), mock.patch("hanzi_synth.render.generate.verify", lambda *_args, **_kwargs: failing):
            stats = generate(limit=100000, out_path=out, progress=lambda _msg: None)

        self.assertEqual(stats["written"], 0, "闸门全判失败时不该有任何一条进主数据集")
        self.assertGreater(stats["unverified"], 0, "假池一条都没过闸门，这条测试白测")
        dropped = [
            json.loads(line)
            for line in out.with_suffix(".dropped.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(dropped), stats["unverified"])
        for row in dropped:
            self.assertEqual(row["drop_reason"], "unverified")
            self.assertEqual(
                row["verification"]["failures"], ["no_internal_terms: 题干里出现内部词"]
            )
            self.assertTrue(row["stem"] and row["explanation"], "落选样本也要留下它本来写成什么样")
            self.assertTrue(row["answer"])

    def test_dropped_file_is_always_written(self) -> None:
        """正常渲染也要落一个落选文件：文件缺席与「一条都没落选」是两件事，别让人去猜。"""
        pool = full_coverage_pool()
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        out = out_dir / "render.jsonl"
        with mock.patch("hanzi_synth.render.generate.load_characters", lambda _path=None: list(pool)):
            stats = generate(limit=100000, out_path=out, progress=lambda _msg: None)
        dropped_path = out.with_suffix(".dropped.jsonl")
        self.assertTrue(dropped_path.exists())
        rows = [line for line in dropped_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), stats["unverified"] + stats["unrenderable"])

    def test_same_spec_renders_identically_across_runs(self) -> None:
        """同一份字池渲染两次必须逐字节相同（同种子 → 同一份数据集）。"""
        pool = full_coverage_pool()
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        first = self._render(pool, tuple(CONSTRUCTORS), out_dir)
        second = self._render(pool, tuple(CONSTRUCTORS), out_dir)
        self.assertEqual(first, second)


class AxesTests(unittest.TestCase):
    """输入：假字池。输出：断言。预期：`axes` 拆出的下标必须与 `render` 真正用的模板一致。

    统计侧（`census` / 数据集句式扫描）拿 `axes` 的下标当「这是第几条骨架」，一旦它与
    `render` 的进制拆法漂开，报出来的句式数就是假的——所以逐字重拼一遍来钉死。
    """

    def test_axes_rebuilds_exactly_what_render_renders(self) -> None:
        """按 `axes` 手工重拼的句子与 `render` 的输出逐字相同（含候选呈现与解析）。"""
        rendered = 0
        for task_id, spec in _offline_specs().items():
            cores = O._CORES[task_id]
            bodies = O._EXPLANATIONS[task_id]
            for variant in _sample_variants(O.variant_space(spec)):
                picked = O.axes(task_id, variant, has_options=bool(spec.options))
                self.assertIsNotNone(picked, f"{task_id} 有模板却拆不出下标")
                quote = O._QUOTES[picked.quote]
                sep = O._LIST_STYLES[picked.sep] if spec.options else O._LIST_STYLES[0]
                context = O._context(spec, quote, sep)
                parts = [O._OPENERS[picked.opener] + cores[picked.core].format_map(context)]
                if spec.options:
                    parts.append(O._OPTION_STYLES[picked.option_style].format_map(context))
                if O._TAILS[picked.tail]:
                    parts.append(O._TAILS[picked.tail])
                expected = (
                    "".join(parts),
                    bodies[picked.explanation % len(bodies)].format_map(context),
                )
                got = O.render(spec, variant)
                if got is None:
                    # 唯一的 None 来源是泄题：答案落在题干里。用与闸门同一判据复核，
                    # 不许拿「跳过」把「axes 拆出来的句子其实渲染不出来」盖过去。
                    stem_norm = normalize_for_match(expected[0])
                    leaked = [
                        item
                        for item in spec.accepted
                        if normalize_for_match(str(item))
                        and stem_norm.count(normalize_for_match(str(item)))
                        > answer_leak_limit(spec, item)
                    ]
                    self.assertTrue(leaked, f"{task_id} variant={variant} 无端渲染不出：{expected[0]}")
                    continue
                self.assertEqual(got, expected, f"{task_id} variant={variant}")
                rendered += 1
        self.assertGreater(rendered, 0, "一条都没渲染出来，这条测试白测")

    def test_axes_is_none_without_templates(self) -> None:
        """释义类题（模板池里没有）拆不出下标：返回 None，不抛异常。"""
        for task_id in O.LLM_ONLY_TASK_IDS:
            self.assertIsNone(O.axes(task_id, 0, has_options=False))


class CensusTests(unittest.TestCase):
    """输入：假字池。输出：断言。预期：census 穷举整段句式空间并数出去重后的题干数。"""

    def test_census_counts_distinct_stems_and_verifies_them(self) -> None:
        """一条题的 census：不同题干数 ≤ 空间大小，且过闸门的题干是它的子集、不为空。"""
        specs = _one_spec_per_type()
        spec = specs["T1_stroke_total"]
        found = census(spec)
        self.assertEqual(found.space, O.variant_space(spec))
        self.assertGreater(len(found.stems), 1)
        self.assertTrue(found.verified, "一条都过不了闸门，模板池就是坏的")
        self.assertTrue(found.verified <= found.stems)
        self.assertTrue(found.explanations)
        # 三层句式数各报各的：问法骨架 ≤ 骨架×收尾 ≤ 真的渲染出来的题干 ≤ 整段空间。
        self.assertGreaterEqual(len(found.skeletons), 1)
        self.assertGreaterEqual(len(found.skeleton_tails), len(found.skeletons))
        self.assertGreaterEqual(len(found.stems), len(found.skeleton_tails))
        self.assertLessEqual(len(found.stems), found.space)

    def test_leaking_variants_are_skipped_not_counted(self) -> None:
        """答案落在题干里的变体（如 T9 的答案「一」撞上「来一题」）按设计跳过，不计进任何集合。"""
        pool = full_coverage_pool()
        index = make_index(pool)
        spec = next(
            s
            for s in build_all(pool[0], rng=random.Random(7), difficulty=Difficulty(), index=index)
            if s.task_id == "T9_component_chain"
        )
        found = census(spec)
        self.assertLess(
            len(found.stems), found.space, "这个夹具本该有渲不出来的变体，否则这条测试白测"
        )
        self.assertTrue(found.verified)
