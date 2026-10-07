"""抓取编排单元测试：增量跳过、404、失败记录、状态合并。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeResponse, RoutingSession, make_request_exception
from zisea.api import ZiseaApi
from zisea.client import ZiseaClient
from zisea.crawl import (
    CHARS_FILE,
    CHARS_INDEX_FILE,
    enumerate_light,
    report_to_dict,
    run_chars,
    write_failed,
    write_state,
)
from zisea.store import JsonlStore, load_json


def build_api(prefix_ids: dict[str, list[str]], details: dict[str, object]) -> ZiseaApi:
    """输入：前缀->ZIID 列表，ZIID->详情（dict 表示 200，404 表示未收录，"boom" 表示网络异常）。
    输出：接了假 session 的 ZiseaApi。预期：完整走真实的分页/错误归一化代码路径。
    """
    groups = {
        "series": [
            {
                "series": next(iter(prefix_ids))[0],
                "count": sum(len(ids) for ids in prefix_ids.values()),
                "groups": [{"prefix": prefix, "count": len(ids)} for prefix, ids in prefix_ids.items()],
            }
        ]
    }

    def z_range_handler(url, params):
        prefix = params["prefix"]
        ids = prefix_ids[prefix]
        page = params["page"]
        size = params["page_size"]
        start = (page - 1) * size
        chunk = ids[start : start + size]
        return FakeResponse(200, {"items": [{"ziid": z} for z in chunk], "total": len(ids)})

    def detail_handler(url, params):
        ziid = url.rsplit("/", 1)[-1]
        value = details[ziid]
        if value == "boom":
            raise make_request_exception()
        if value == 404:
            return FakeResponse(404, {"detail": f"Character not found: {ziid}"})
        return FakeResponse(200, value)

    session = RoutingSession(
        {"/api/groups": FakeResponse(200, groups), "/api/z-range": z_range_handler, "/api/z/": detail_handler}
    )
    return ZiseaApi(ZiseaClient(base_url="http://example.test", session=session, backoff=0, delay=0))


def silent(_message: str) -> None:
    """输入：日志文本。输出：无。预期：测试时不打印进度。"""


class EnumerateLightTests(unittest.TestCase):
    def test_returns_ziid_and_prefix_in_order(self) -> None:
        """枚举结果应带上所属前缀，顺序与 groups 一致。"""
        api = build_api({"U04": ["U04E00", "U04E01"], "Z4": ["Z43651"]}, {})
        self.assertEqual(
            enumerate_light(api),
            [
                {"ziid": "U04E00", "prefix": "U04"},
                {"ziid": "U04E01", "prefix": "U04"},
                {"ziid": "Z43651", "prefix": "Z4"},
            ],
        )


class RunCharsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_first_run_fetches_all_and_writes_index(self) -> None:
        """首次运行应抓全部详情，并把轻量索引写进 chars_index.jsonl。"""
        api = build_api({"U04": ["U04E00", "U04E01"]}, {"U04E00": {"ziid": "U04E00"}, "U04E01": {"ziid": "U04E01"}})
        report = run_chars(self.out, api, workers=2, log=silent)
        self.assertEqual(report.enumerated, 2)
        self.assertEqual(report.fetched, 2)
        self.assertEqual(report.failed, 0)
        self.assertEqual(JsonlStore(self.out / CHARS_FILE).count(), 2)
        self.assertEqual(JsonlStore(self.out / CHARS_INDEX_FILE).count(), 2)

    def test_second_run_is_incremental(self) -> None:
        """已入库的 ZIID 不应重复抓取，fetched 归零。"""
        api = build_api({"U04": ["U04E00", "U04E01"]}, {"U04E00": {"ziid": "U04E00"}, "U04E01": {"ziid": "U04E01"}})
        run_chars(self.out, api, workers=2, log=silent)
        report = run_chars(self.out, api, workers=2, log=silent)
        self.assertEqual(report.fetched, 0)
        self.assertEqual(report.already_done, 2)
        self.assertEqual(JsonlStore(self.out / CHARS_FILE).count(), 2)

    def test_new_ziid_only_fetches_the_delta(self) -> None:
        """站点新增一个字后，增量模式只抓那一个。"""
        api = build_api({"U04": ["U04E00"]}, {"U04E00": {"ziid": "U04E00"}})
        run_chars(self.out, api, workers=1, log=silent)
        api2 = build_api(
            {"U04": ["U04E00", "U04E01"]},
            {"U04E00": {"ziid": "U04E00"}, "U04E01": {"ziid": "U04E01"}},
        )
        report = run_chars(self.out, api2, workers=1, log=silent)
        self.assertEqual(report.fetched, 1)
        self.assertEqual(report.added, 1)
        self.assertEqual(report.removed, 0)
        self.assertEqual(JsonlStore(self.out / CHARS_FILE).count(), 2)

    def test_refresh_refetches_everything(self) -> None:
        """--refresh 应无视已入库记录重抓，文件不出现重复行。"""
        api = build_api({"U04": ["U04E00"]}, {"U04E00": {"ziid": "U04E00"}})
        run_chars(self.out, api, workers=1, log=silent)
        report = run_chars(self.out, api, workers=1, refresh=True, log=silent)
        self.assertEqual(report.fetched, 1)
        self.assertEqual(JsonlStore(self.out / CHARS_FILE).count(), 1)

    def test_missing_character_counted_not_stored(self) -> None:
        """404 记入 missing，不写入 chars.jsonl。"""
        api = build_api({"P8": ["P80004", "P80015"]}, {"P80004": {"ziid": "P80004"}, "P80015": 404})
        report = run_chars(self.out, api, workers=1, log=silent)
        self.assertEqual(report.missing, 1)
        self.assertEqual(report.fetched, 1)
        self.assertEqual(JsonlStore(self.out / CHARS_FILE).load_keys(), {"P80004"})

    def test_network_failure_recorded_and_run_continues(self) -> None:
        """单条网络失败只记录，不影响其他条目，并落到 failed_ids。"""
        api = build_api(
            {"U04": ["U04E00", "U04E01"]},
            {"U04E00": {"ziid": "U04E00"}, "U04E01": "boom"},
        )
        report = run_chars(self.out, api, workers=1, log=silent)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.failed_ids, ["U04E01"])
        self.assertEqual(report.fetched, 1)

    def test_failed_ziid_is_retried_next_run(self) -> None:
        """上一轮失败的 ZIID 因为没入库，下一轮会被自动重试。"""
        api = build_api({"U04": ["U04E00", "U04E01"]}, {"U04E00": {"ziid": "U04E00"}, "U04E01": "boom"})
        run_chars(self.out, api, workers=1, log=silent)
        api2 = build_api(
            {"U04": ["U04E00", "U04E01"]},
            {"U04E00": {"ziid": "U04E00"}, "U04E01": {"ziid": "U04E01"}},
        )
        report = run_chars(self.out, api2, workers=1, log=silent)
        self.assertEqual(report.fetched, 1)
        self.assertEqual(report.failed, 0)

    def test_limit_caps_this_run(self) -> None:
        """--limit 只影响本轮抓取数量，枚举仍是全量。"""
        api = build_api(
            {"U04": ["U04E00", "U04E01", "U04E02"]},
            {"U04E00": {"ziid": "U04E00"}, "U04E01": {"ziid": "U04E01"}, "U04E02": {"ziid": "U04E02"}},
        )
        report = run_chars(self.out, api, workers=1, limit=2, log=silent)
        self.assertEqual(report.enumerated, 3)
        self.assertEqual(report.fetched, 2)

    def test_recovers_from_partial_last_line(self) -> None:
        """上一轮崩溃留下半行时，重跑要能修复并继续。"""
        api = build_api({"U04": ["U04E00", "U04E01"]}, {"U04E00": {"ziid": "U04E00"}, "U04E01": {"ziid": "U04E01"}})
        run_chars(self.out, api, workers=1, log=silent)
        with (self.out / CHARS_FILE).open("a", encoding="utf-8") as handle:
            handle.write('{"ziid": "U04E02"')
        report = run_chars(self.out, api, workers=1, log=silent)
        self.assertEqual(report.failed, 0)
        self.assertEqual(JsonlStore(self.out / CHARS_FILE).count(), 2)


class StateAndReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_write_state_merges_with_existing(self) -> None:
        """state.json 多次写入应合并字段而不是整体覆盖。"""
        write_state(self.out, {"chars": {"fetched": 1}})
        write_state(self.out, {"sprite_version": "abc"})
        state = load_json(self.out / "state.json")
        self.assertEqual(state["chars"], {"fetched": 1})
        self.assertEqual(state["sprite_version"], "abc")

    def test_write_failed_creates_then_removes_file(self) -> None:
        """有失败时写文件，无失败时删除旧文件，避免误导。"""
        write_failed(self.out, ["U04E01"])
        self.assertEqual((self.out / "failed_ziids.txt").read_text(encoding="utf-8"), "U04E01\n")
        write_failed(self.out, [])
        self.assertFalse((self.out / "failed_ziids.txt").exists())

    def test_report_to_dict_drops_failed_ids(self) -> None:
        """报告序列化时去掉明细列表（明细另写文件）。"""
        from zisea.crawl import CrawlReport

        data = report_to_dict(CrawlReport(target="chars", fetched=3, failed_ids=["A"]))
        self.assertEqual(data["target"], "chars")
        self.assertNotIn("failed_ids", data)


if __name__ == "__main__":
    unittest.main()
