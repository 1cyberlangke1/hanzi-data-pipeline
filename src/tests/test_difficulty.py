"""L5 难度标定的单元测试（不联网、不读真实数据）。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hanzi_synth.pipeline.difficulty import (
    DEFAULT_BRANCHES,
    _as_bool,
    branches_for,
    difficulty_note,
    estimate_irt_from_responses,
    load_responses,
    target_facility,
    write_calibration,
)
from hanzi_synth.tasks.base import MIXED_LEVEL, Difficulty


class TargetFacilityTests(unittest.TestCase):
    def test_single_hop_is_k_over_k_plus_one(self) -> None:
        """单跳先验 p = K/(K+1)：K=1 -> 0.5，K=4 -> 0.8。"""
        self.assertAlmostEqual(target_facility(hops=1, branches=1), 0.5)
        self.assertAlmostEqual(target_facility(hops=1, branches=4), 0.8)

    def test_hops_compound(self) -> None:
        """多跳独立串联：hops=2、K=1 -> 1 - 0.5**2 = 0.75。"""
        self.assertAlmostEqual(target_facility(hops=2, branches=1), 0.75)
        self.assertAlmostEqual(target_facility(hops=3, branches=4), 1 - 0.2**3)

    def test_invalid_arguments_raise(self) -> None:
        """越界参数当场报错，不静默产出假难度。"""
        with self.assertRaises(ValueError):
            target_facility(hops=0)
        with self.assertRaises(ValueError):
            target_facility(branches=0)


class DifficultyNoteTests(unittest.TestCase):
    def test_branches_follow_the_task(self) -> None:
        """T8 的 K 用约束个数，其余题型用默认分支数。"""
        self.assertEqual(branches_for("T8_constraint", 3), 3)
        self.assertEqual(branches_for("T1_stroke_total", 3), DEFAULT_BRANCHES)

    def test_only_hop_tasks_report_hops(self) -> None:
        """不消费 hops 的题型如实写 hops=1，不假装旋钮起作用了。"""
        note = difficulty_note("T9_component_chain", Difficulty(hops=2))
        self.assertEqual(note["hops"], 2)
        self.assertEqual(difficulty_note("T1_stroke_total", Difficulty(hops=2))["hops"], 1)

    def test_constraints_only_for_t8(self) -> None:
        """constraints 只对 T8 生效，其余题型写 None。"""
        self.assertEqual(difficulty_note("T8_constraint", Difficulty(constraints=2))["constraints"], 2)
        self.assertIsNone(difficulty_note("T9_component_chain", Difficulty(constraints=2))["constraints"])

    def test_modes_are_recorded_so_sampling_is_visible(self) -> None:
        """注记要能区分「配置就是这档」和「从 mixed 抽 / 退档来的」——否则看不出难度是抽样来的。"""
        note = difficulty_note("T9_component_chain", Difficulty(hops=MIXED_LEVEL), resolved_hops=2)
        self.assertEqual(note["hops"], 2)
        self.assertEqual(note["hops_mode"], MIXED_LEVEL)
        note = difficulty_note("T8_constraint", Difficulty(constraints=2), resolved_constraints=2)
        self.assertEqual(note["constraints"], 2)
        self.assertEqual(note["constraints_mode"], 2)
        self.assertEqual(note["branches"], 2)

    def test_note_is_labelled_as_prior(self) -> None:
        """注记必须自称解析式先验，不能让人误以为这是实测标定。"""
        note = difficulty_note("T4_reading", Difficulty())
        self.assertEqual(note["kind"], "analytic-prior")
        self.assertEqual(note["distractor_depth"], "near")
        self.assertAlmostEqual(note["target_facility"], target_facility(hops=1, branches=DEFAULT_BRANCHES))


class AsBoolTests(unittest.TestCase):
    def test_accepts_bool_int_and_words(self) -> None:
        """兼容 bool / 0/1 / 中英文字面，无法判定时返回 None。"""
        self.assertIs(_as_bool(True), True)
        self.assertIs(_as_bool(0), False)
        self.assertIs(_as_bool("对"), True)
        self.assertIs(_as_bool("false"), False)
        self.assertIsNone(_as_bool("也许"))
        self.assertIsNone(_as_bool(2))


def responses(rows: list[tuple[str, str, int]]) -> list[dict]:
    """输入：(person, item, correct) 三元组列表。输出：作答记录列表。预期：测试用最小构造。"""
    return [{"person": person, "item": item, "correct": correct} for person, item, correct in rows]


class IrtCalibrationTests(unittest.TestCase):
    def test_no_data_is_honest(self) -> None:
        """没有作答数据就如实报未标定，绝不编参数。"""
        result = estimate_irt_from_responses([])
        self.assertFalse(result["calibrated"])
        self.assertIn("reason", result)

    def test_rows_missing_fields_are_filtered(self) -> None:
        """缺 person / item / correct 的行被过滤；全被过滤等于没数据。"""
        result = estimate_irt_from_responses([{"person": "p1"}, {"item": "T1"}])
        self.assertFalse(result["calibrated"])

    def test_orders_item_difficulty_by_error_rate(self) -> None:
        """答错越多的题难度越高，且以 mean(b)=0 做识别约束。"""
        data = responses(
            [
                ("p1", "easy", 1), ("p1", "mid", 1), ("p1", "hard", 1),
                ("p2", "easy", 1), ("p2", "mid", 1), ("p2", "hard", 0),
                ("p3", "easy", 1), ("p3", "mid", 1), ("p3", "hard", 0),
                ("p4", "easy", 1), ("p4", "mid", 0), ("p4", "hard", 0),
                ("p5", "easy", 0), ("p5", "mid", 0), ("p5", "hard", 0),
            ]
        )
        result = estimate_irt_from_responses(data)
        self.assertTrue(result["calibrated"])
        self.assertEqual(result["model"], "rasch-1pl-jmle")
        self.assertEqual(result["responses"], 15)
        items = result["item_parameters"]
        self.assertGreater(items["hard"]["difficulty"], items["mid"]["difficulty"])
        self.assertGreater(items["mid"]["difficulty"], items["easy"]["difficulty"])
        mean = sum(item["difficulty"] for item in items.values()) / len(items)
        self.assertAlmostEqual(mean, 0.0, places=5)

    def test_boundary_cases_are_flagged(self) -> None:
        """全对/全错的题参数无界，如实标 boundary=True，不假装估出来了。"""
        data = responses(
            [
                ("p1", "always", 1), ("p1", "mixed", 1),
                ("p2", "always", 1), ("p2", "mixed", 0),
            ]
        )
        result = estimate_irt_from_responses(data)
        self.assertTrue(result["calibrated"])
        self.assertTrue(result["item_parameters"]["always"]["boundary"])
        self.assertFalse(result["item_parameters"]["mixed"]["boundary"])
        self.assertTrue(result["person_parameters"]["p1"]["boundary"])


class ResponseFileTests(unittest.TestCase):
    def test_load_and_write_calibration(self) -> None:
        """作答文件读得进、校准结果写得下，且是 UTF-8 JSON。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in responses([("p1", "T1", 1), ("p2", "T1", 0)]):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            loaded = load_responses(path)
            self.assertEqual(len(loaded), 2)
            result = estimate_irt_from_responses(loaded)
            target = write_calibration(Path(tmp) / "nested" / "calib.json", result)
            self.assertTrue(target.exists())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["calibrated"], True)


if __name__ == "__main__":
    unittest.main()
