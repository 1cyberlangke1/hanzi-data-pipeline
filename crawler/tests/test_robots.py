"""robots.py 单元测试：校验判定、保守策略、URL 拼接。全部离线。"""

from __future__ import annotations

import unittest

from tests.fakes import FakeResponse, RoutingSession
from zisea.client import ZiseaClient
from zisea.robots import check_robots, fetch_robots, robots_url_for

DISALLOW_ALL = "User-agent: *\nDisallow: /\n"
ALLOW_ALL = "User-agent: *\nDisallow:\n"


def make_client(routes: dict) -> ZiseaClient:
    """输入：URL 子串 -> 响应的路由表。输出：接了假 session 的 ZiseaClient。
    预期：backoff=0 让重试路径在测试里瞬间走完。
    """
    session = RoutingSession(routes)
    return ZiseaClient(base_url="http://example.test", session=session, backoff=0, delay=0)


class UrlTests(unittest.TestCase):
    def test_joins_plain_base(self) -> None:
        """基址无尾斜杠时也要拼出正确 URL。"""
        self.assertEqual(robots_url_for("http://zisea.com"), "http://zisea.com/robots.txt")

    def test_joins_base_with_trailing_slash(self) -> None:
        """基址带尾斜杠时不产生双斜杠。"""
        self.assertEqual(robots_url_for("http://zisea.com/"), "http://zisea.com/robots.txt")


class FetchTests(unittest.TestCase):
    def test_returns_status_and_text(self) -> None:
        """200 时返回 (200, 原文)。"""
        client = make_client({"/robots.txt": FakeResponse(200, None, content=DISALLOW_ALL.encode("utf-8"))})
        self.assertEqual(fetch_robots(client, "http://example.test"), (200, DISALLOW_ALL))

    def test_missing_file_is_404_with_empty_text(self) -> None:
        """站点没有 robots.txt 时返回 (404, "")，由调用方按「不限制」处理。"""
        client = make_client({})
        self.assertEqual(fetch_robots(client, "http://example.test"), (404, ""))


class CheckTests(unittest.TestCase):
    def test_disallow_all_blocks(self) -> None:
        """Disallow: / 必须判定为禁止，并给出可读原因。"""
        client = make_client({"/robots.txt": FakeResponse(200, None, content=DISALLOW_ALL.encode("utf-8"))})
        decision = check_robots(client, base_url="http://example.test")
        self.assertFalse(decision.allowed)
        self.assertIn("禁止", decision.reason)
        self.assertEqual(decision.status, 200)
        self.assertIn("Disallow: /", decision.raw_text)

    def test_missing_file_allows(self) -> None:
        """没有 robots.txt 视为不限制（RFC 9309）。"""
        client = make_client({})
        decision = check_robots(client, base_url="http://example.test")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, 404)

    def test_explicit_allow_allows(self) -> None:
        """Disallow 留空即显式放行。"""
        client = make_client({"/robots.txt": FakeResponse(200, None, content=ALLOW_ALL.encode("utf-8"))})
        self.assertTrue(check_robots(client, base_url="http://example.test").allowed)

    def test_matching_user_agent_rule_blocks(self) -> None:
        """命中与自身 UA 同名的规则时禁止（UA 取 '/' 前的 token 做匹配）。"""
        body = "User-agent: zisea-offline-crawler\nDisallow: /\n"
        client = make_client({"/robots.txt": FakeResponse(200, None, content=body.encode("utf-8"))})
        decision = check_robots(
            client, base_url="http://example.test", user_agent="zisea-offline-crawler/1.0 (personal offline study)"
        )
        self.assertFalse(decision.allowed)

    def test_unrelated_user_agent_rule_does_not_block(self) -> None:
        """别的 bot 被禁不等于本工具被禁，只有 * 或同名规则才生效。"""
        body = "User-agent: otherbot\nDisallow: /\n\nUser-agent: *\nDisallow:\n"
        client = make_client({"/robots.txt": FakeResponse(200, None, content=body.encode("utf-8"))})
        decision = check_robots(
            client, base_url="http://example.test", user_agent="zisea-offline-crawler/1.0 (personal offline study)"
        )
        self.assertTrue(decision.allowed)

    def test_network_failure_is_conservative(self) -> None:
        """取不到 robots.txt 时保守拒绝，绝不默认放行。"""
        client = make_client({"/robots.txt": FakeResponse(500, None)})
        decision = check_robots(client, base_url="http://example.test")
        self.assertFalse(decision.allowed)
        self.assertIn("保守", decision.reason)


if __name__ == "__main__":
    unittest.main()
