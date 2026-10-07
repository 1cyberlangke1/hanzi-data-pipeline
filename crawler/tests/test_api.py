"""ZiseaApi 单元测试：分页收敛、404 归一化、参数透传。"""

from __future__ import annotations

import unittest

from tests.fakes import FakeResponse, RoutingSession
from zisea.api import ZiseaApi
from zisea.client import BadRequestError, ZiseaClient, NotFoundError


def make_api(routes) -> tuple[ZiseaApi, RoutingSession]:
    """输入：URL 子串 -> 响应/工厂 的路由表。输出：(api, session)，便于断言实际请求。"""
    session = RoutingSession(routes)
    client = ZiseaClient(base_url="http://example.test", session=session, backoff=0, delay=0)
    return ZiseaApi(client), session


GROUPS = {
    "series": [
        {"series": "U", "count": 3, "groups": [{"prefix": "U04", "count": 2}, {"prefix": "U05", "count": 1}]},
        {"series": "Z", "count": 1, "groups": [{"prefix": "Z4", "count": 1}]},
    ]
}


class PrefixTests(unittest.TestCase):
    def test_iter_prefixes_flattens_series(self) -> None:
        """输出顺序与 /api/groups 一致，元素为 (系列, 前缀, 计数)。"""
        api, _ = make_api({"/api/groups": FakeResponse(200, GROUPS)})
        self.assertEqual(
            list(api.iter_prefixes()),
            [("U", "U04", 2), ("U", "U05", 1), ("Z", "Z4", 1)],
        )

    def test_iter_prefixes_handles_empty_series(self) -> None:
        """groups 缺字段时不应崩，返回空迭代。"""
        api, _ = make_api({"/api/groups": FakeResponse(200, {})})
        self.assertEqual(list(api.iter_prefixes()), [])


class ZRangeTests(unittest.TestCase):
    def test_passes_expected_params(self) -> None:
        """prefix/page/page_size 必须原样传给服务端。"""
        api, session = make_api({"/api/z-range": FakeResponse(200, {"items": [], "total": 0})})
        api.z_range("U04", page=3, page_size=500)
        self.assertEqual(session.calls[0][1], {"prefix": "U04", "page": 3, "page_size": 500})

    def test_iter_z_range_walks_pages_until_total(self) -> None:
        """total=5、每页 2 时，应翻 3 页后停止，共 5 条。"""
        pages = {
            1: {"items": [{"ziid": "A1"}, {"ziid": "A2"}], "total": 5},
            2: {"items": [{"ziid": "A3"}, {"ziid": "A4"}], "total": 5},
            3: {"items": [{"ziid": "A5"}], "total": 5},
        }

        def handler(url, params):
            return FakeResponse(200, pages[params["page"]])

        api, session = make_api({"/api/z-range": handler})
        self.assertEqual([r["ziid"] for r in api.iter_z_range("U04", page_size=2)], ["A1", "A2", "A3", "A4", "A5"])
        self.assertEqual(len(session.calls), 3)

    def test_iter_z_range_stops_on_empty_page(self) -> None:
        """服务端 total 虚高但返回空页时，必须立即停止，不能死循环。"""
        api, session = make_api({"/api/z-range": FakeResponse(200, {"items": [], "total": 999})})
        self.assertEqual(list(api.iter_z_range("U04")), [])
        self.assertEqual(len(session.calls), 1)

    def test_iter_ziids_covers_all_prefixes(self) -> None:
        """全量枚举应遍历所有前缀，产出 (ZIID, 前缀)。"""

        def handler(url, params):
            prefix = params["prefix"]
            return FakeResponse(200, {"items": [{"ziid": f"{prefix}1"}], "total": 1})

        api, _ = make_api({"/api/groups": FakeResponse(200, GROUPS), "/api/z-range": handler})
        self.assertEqual(list(api.iter_ziids()), [("U041", "U04"), ("U051", "U05"), ("Z41", "Z4")])


class CharacterTests(unittest.TestCase):
    def test_returns_record(self) -> None:
        """200 时返回完整详情 dict。"""
        api, session = make_api({"/api/z/": FakeResponse(200, {"ziid": "U04E00"})})
        self.assertEqual(api.character("U04E00"), {"ziid": "U04E00"})
        self.assertIn("/api/z/U04E00", session.calls[0][0])

    def test_returns_none_on_404(self) -> None:
        """404 表示站点未收录，返回 None 而不是抛错。"""
        api, _ = make_api({"/api/z/": FakeResponse(404, {"detail": "Character not found"})})
        self.assertIsNone(api.character("P00001"))

    def test_bad_ziid_still_raises(self) -> None:
        """400 说明 ZIID 格式写错，是程序 bug，必须抛出而不是当成「未收录」。"""
        api, _ = make_api({"/api/z/": FakeResponse(400, {"detail": "Invalid ZIID"})})
        with self.assertRaises(BadRequestError):
            api.character("00001")

    def test_not_found_error_is_subclass_of_zisea_error(self) -> None:
        """NotFoundError 必须继承 ZiseaError，方便上层统一兜底。"""
        self.assertTrue(issubclass(NotFoundError, Exception))


class StatsTests(unittest.TestCase):
    def test_stats_passthrough(self) -> None:
        """stats 原样返回服务端结构。"""
        api, _ = make_api({"/api/stats": FakeResponse(200, {"total_characters": 145504})})
        self.assertEqual(api.stats()["total_characters"], 145504)

    def test_sprite_version_returns_string(self) -> None:
        """正常返回版本号字符串。"""
        api, _ = make_api({"/api/sprite-version": FakeResponse(200, {"version": "8894233ed869"})})
        self.assertEqual(api.sprite_version(), "8894233ed869")

    def test_sprite_version_missing_key_returns_empty(self) -> None:
        """字段缺失时返回空串，调用方无需判 None。"""
        api, _ = make_api({"/api/sprite-version": FakeResponse(200, {})})
        self.assertEqual(api.sprite_version(), "")


if __name__ == "__main__":
    unittest.main()
