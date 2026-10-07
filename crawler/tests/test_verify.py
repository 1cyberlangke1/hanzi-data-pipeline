"""verify.py 单元测试：完整性对账能否抓出漏抓、重复、坏行、缺图。全部离线。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeResponse, RoutingSession
from zisea.api import ZiseaApi
from zisea.client import ZiseaClient
from zisea.crawl import CHARS_FILE, CHARS_INDEX_FILE
from zisea.sprites import SPRITES_DIR
from zisea.verify import prefix_lengths, run_verify, scan_details, verify_report_to_dict

silent = lambda _message: None  # noqa: E731 - 测试里不需要日志

GROUPS = {
    "series": [
        {"series": "U", "count": 3, "groups": [{"prefix": "U04", "count": 2}, {"prefix": "U05", "count": 1}]},
        {"series": "Z", "count": 1, "groups": [{"prefix": "Z4", "count": 1}]},
    ]
}
STATS = {"total_characters": 4, "series": {"U": 3, "Z": 1}, "ids": {"records": 4, "with_ids": 2}}


def sample_records() -> list[dict]:
    """输入：无。输出：4 条样例记录：U04E00/U05B50 有 IDS，Z40001 无码位（要精灵图）。"""
    return [
        {"ziid": "U04E00", "property": {"zi": "一", "ids": "#(H)"}, "meanings": {"mean_h": ["a"]}},
        {"ziid": "U04E01", "property": {"zi": "丁"}, "meanings": {}},
        {"ziid": "U05B50", "property": {"zi": "中", "ids": "#(H)(G)"}, "meanings": {}},
        {"ziid": "Z40001", "property": {"zi": "", "ids": ""}, "meanings": {}},
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    """输入：目标路径与记录列表。输出：无。预期：写 UTF-8 JSONL（LF 行尾）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class Fixture(unittest.TestCase):
    """搭一套「与站点计数完全对得上」的本地产出，供各用例按需破坏。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        self.write_dataset(sample_records())

    def write_dataset(self, records: list[dict], make_sprites: bool = True) -> None:
        """输入：记录列表、是否同时生成精灵图。输出：无。预期：写出 chars/chars_index/精灵图。"""
        write_jsonl(self.out / CHARS_FILE, records)
        write_jsonl(self.out / CHARS_INDEX_FILE, [{"ziid": r["ziid"], "prefix": ""} for r in records])
        sprite_dir = self.out / SPRITES_DIR
        sprite_dir.mkdir(parents=True, exist_ok=True)
        for old in sprite_dir.glob("*.png"):
            old.unlink()
        if make_sprites:
            for record in records:
                if not (record.get("property") or {}).get("zi"):
                    (sprite_dir / f"{record['ziid']}.png").write_bytes(b"png")

    def api(self) -> ZiseaApi:
        """输入：无。输出：接了假 session 的 ZiseaApi（stats/groups 走真实解析路径）。"""
        session = RoutingSession(
            {"/api/stats": FakeResponse(200, STATS), "/api/groups": FakeResponse(200, GROUPS)}
        )
        return ZiseaApi(ZiseaClient(base_url="http://example.test", session=session, backoff=0, delay=0))

    def check(self, name: str):
        """输入：检查项名称。输出：该项 Check 对象。预期：名字写错时立刻报错而不是静默通过。"""
        report = run_verify(self.out, self.api(), log=silent)
        self._last_report = report
        for item in report.checks:
            if item.name == name:
                return item
        raise AssertionError(f"没有名为 {name} 的检查项，实际有 {[c.name for c in report.checks]}")


class PrefixLengthTests(unittest.TestCase):
    def test_reads_length_per_series(self) -> None:
        """U 组前缀 3 位、Z 组 2 位。"""
        self.assertEqual(prefix_lengths(GROUPS), {"U": 3, "Z": 2})

    def test_inconsistent_length_raises(self) -> None:
        """同系列前缀长度不一致说明对账口径失效，必须抛错而不是给出错误的「全绿」。"""
        broken = {"series": [{"series": "U", "groups": [{"prefix": "U04", "count": 1}, {"prefix": "U0", "count": 1}]}]}
        with self.assertRaises(ValueError):
            prefix_lengths(broken)


class ScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / CHARS_FILE

    def test_counts_series_and_prefix(self) -> None:
        """按系列与前缀长度分别计数，并统计有 IDS 的字与需要精灵图的字。"""
        write_jsonl(self.path, sample_records())
        scan = scan_details(self.path, {"U": 3, "Z": 2})
        self.assertEqual(scan.total_lines, 4)
        self.assertEqual(scan.series_counts, {"U": 3, "Z": 1})
        self.assertEqual(scan.prefix_counts, {"U04": 2, "U05": 1, "Z4": 1})
        self.assertEqual(scan.with_ids, 2)
        self.assertEqual(scan.needs_sprite, {"Z40001"})

    def test_flags_bad_lines_and_missing_fields(self) -> None:
        """坏行与缺字段都要被记下来，且不影响其余行统计。"""
        with self.path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("{不是 JSON\n")
            handle.write(json.dumps({"property": {}, "meanings": {}}, ensure_ascii=False) + "\n")
            handle.write(json.dumps({"ziid": "U04E00", "property": {}}, ensure_ascii=False) + "\n")
        scan = scan_details(self.path, {"U": 3})
        self.assertEqual(len(scan.bad_lines), 1)
        self.assertEqual(len(scan.missing_fields), 2)
        self.assertEqual(scan.keys, {"U04E00"})

    def test_detects_duplicates(self) -> None:
        """重复主键要被单独记下来。"""
        records = sample_records()
        write_jsonl(self.path, records + [records[0]])
        scan = scan_details(self.path, {"U": 3, "Z": 2})
        self.assertEqual(scan.duplicates, ["U04E00"])
        self.assertEqual(len(scan.keys), 4)

    def test_missing_file_is_empty_scan(self) -> None:
        """产出文件不存在时返回空统计而不是抛错。"""
        scan = scan_details(Path(self._tmp.name) / "nope.jsonl", {})
        self.assertEqual(scan.total_lines, 0)
        self.assertEqual(scan.keys, set())


class RunVerifyTests(Fixture):
    def test_all_green_on_complete_dataset(self) -> None:
        """本地与站点逐项对上时 ok=True，且 8 项检查全部通过。"""
        report = run_verify(self.out, self.api(), log=silent)
        self.assertTrue(report.ok)
        self.assertEqual(len(report.checks), 8)
        self.assertTrue(all(check.ok for check in report.checks))
        self.assertEqual(report.totals["detail_records"], 4)
        self.assertEqual(report.totals["sprites"], 1)

    def test_missing_character_fails_prefix_check(self) -> None:
        """少抓一个字：逐前缀计数与总数都应报出差异，ok=False。"""
        self.write_dataset([r for r in sample_records() if r["ziid"] != "U04E01"])
        report = run_verify(self.out, self.api(), log=silent)
        self.assertFalse(report.ok)
        detail = {c.name: c for c in report.checks}
        self.assertFalse(detail["详情条数"].ok)
        self.assertFalse(detail["逐前缀计数（完整性核心）"].ok)
        self.assertIn("U04", detail["逐前缀计数（完整性核心）"].detail)

    def test_stale_index_fails_index_check(self) -> None:
        """索引文件与详情主键不一致要被抓出来。"""
        write_jsonl(self.out / CHARS_INDEX_FILE, [{"ziid": "U04E00"}, {"ziid": "U99999"}])
        self.assertFalse(self.check("chars_index 与 chars 主键一致").ok)

    def test_missing_sprite_fails(self) -> None:
        """无码位字少了字形图片要被抓出来。"""
        (self.out / SPRITES_DIR / "Z40001.png").unlink()
        self.assertFalse(self.check("精灵图齐全").ok)

    def test_orphan_sprite_fails(self) -> None:
        """多出来的精灵图（对应字已不在产出里）同样要报。"""
        (self.out / SPRITES_DIR / "Z99999.png").write_bytes(b"png")
        self.assertFalse(self.check("精灵图齐全").ok)

    def test_duplicate_line_fails(self) -> None:
        """重复行会被去重检查抓到。"""
        records = sample_records()
        write_jsonl(self.out / CHARS_FILE, records + [records[0]])
        self.assertFalse(self.check("重复 ZIID").ok)

    def test_report_to_dict_is_serializable(self) -> None:
        """报告要能直接 json 序列化（CLI 会打印它）。"""
        payload = verify_report_to_dict(run_verify(self.out, self.api(), log=silent))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["totals"]["unique_ziids"], 4)
        json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
