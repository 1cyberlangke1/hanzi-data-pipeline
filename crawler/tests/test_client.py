"""ZiseaClient 单元测试：状态码映射、重试、URL 拼接、session 归属。"""

from __future__ import annotations

import unittest

from tests.fakes import FakeResponse, FakeSession, make_request_exception
from zisea.client import BadRequestError, NotFoundError, ZiseaClient, ZiseaError


def make_client(session, **overrides) -> ZiseaClient:
    """输入：假 session 与可选覆盖参数。输出：退避/延迟归零的客户端，保证测试不等待。"""
    params = {"base_url": "http://example.test", "session": session, "backoff": 0, "delay": 0}
    params.update(overrides)
    return ZiseaClient(**params)


class GetJsonTests(unittest.TestCase):
    def test_returns_parsed_payload(self) -> None:
        """200 时原样返回解析后的 JSON。"""
        session = FakeSession([FakeResponse(200, {"ziid": "U04E00"})])
        self.assertEqual(make_client(session).get_json("/api/z/U04E00"), {"ziid": "U04E00"})

    def test_builds_url_and_forwards_params(self) -> None:
        """路径以 / 开头与否都要拼成 base + path，且 params 原样透传。"""
        session = FakeSession([FakeResponse(200, {})])
        make_client(session).get_json("api/z-range", {"prefix": "U04", "page": 2})
        self.assertEqual(session.calls[0][0], "http://example.test/api/z-range")
        self.assertEqual(session.calls[0][1], {"prefix": "U04", "page": 2})

    def test_trailing_slash_in_base_url_is_normalized(self) -> None:
        """基址带尾斜杠不应产生双斜杠。"""
        session = FakeSession([FakeResponse(200, {})])
        make_client(session, base_url="http://example.test/").get_json("/api/stats")
        self.assertEqual(session.calls[0][0], "http://example.test/api/stats")

    def test_sets_user_agent_header(self) -> None:
        """构造时应写入 UA 与 Accept 头，避免被当成无头脚本。"""
        session = FakeSession([FakeResponse(200, {})])
        make_client(session, user_agent="UA-TEST")
        self.assertEqual(session.headers["User-Agent"], "UA-TEST")
        self.assertEqual(session.headers["Accept"], "application/json")


class StatusCodeTests(unittest.TestCase):
    def test_404_raises_not_found(self) -> None:
        """404 归一到 NotFoundError，供上层判断「未收录」。"""
        session = FakeSession([FakeResponse(404, {"detail": "Character not found: P00001"})])
        with self.assertRaises(NotFoundError):
            make_client(session).get_json("/api/z/P00001")

    def test_400_raises_bad_request_without_retry(self) -> None:
        """400 是参数写错，属于 bug，不重试。"""
        session = FakeSession([FakeResponse(400, {"detail": "Invalid ZIID: x"})])
        with self.assertRaises(BadRequestError):
            make_client(session, retries=5).get_json("/api/z/x")
        self.assertEqual(len(session.calls), 1)

    def test_422_raises_bad_request(self) -> None:
        """422（FastAPI 校验失败，如 page_size>500）同样归一到 BadRequestError。"""
        session = FakeSession([FakeResponse(422, {"detail": []})])
        with self.assertRaises(BadRequestError):
            make_client(session).get_json("/api/z-range", {"page_size": 99999})

    def test_retries_on_500_then_succeeds(self) -> None:
        """5xx 先失败后成功，应重试并返回成功结果。"""
        session = FakeSession([FakeResponse(500), FakeResponse(200, {"ok": True})])
        self.assertEqual(make_client(session, retries=3).get_json("/api/stats"), {"ok": True})
        self.assertEqual(len(session.calls), 2)

    def test_exhausted_retries_raises_zisea_error(self) -> None:
        """一直 5xx 时，重试次数用尽后抛 ZiseaError，且调用次数等于 retries。"""
        session = FakeSession([FakeResponse(503)])
        with self.assertRaises(ZiseaError):
            make_client(session, retries=3).get_json("/api/stats")
        self.assertEqual(len(session.calls), 3)

    def test_invalid_json_raises_zisea_error(self) -> None:
        """返回 HTML 错误页时 json() 抛 ValueError，应收敛成 ZiseaError 而不是裸异常。"""
        session = FakeSession([FakeResponse(200, invalid_json=True)])
        with self.assertRaises(ZiseaError):
            make_client(session, retries=1).get_json("/api/stats")


class NetworkErrorTests(unittest.TestCase):
    def test_retries_network_error_then_succeeds(self) -> None:
        """网络异常按重试计数，恢复后正常返回。"""
        calls = {"n": 0}

        def handler(url, params):
            calls["n"] += 1
            if calls["n"] == 1:
                raise make_request_exception()
            return FakeResponse(200, {"ok": 1})

        client = make_client(FakeSession(handler), retries=3)
        self.assertEqual(client.get_json("/api/stats"), {"ok": 1})
        self.assertEqual(calls["n"], 2)

    def test_network_error_exhausts_retries(self) -> None:
        """网络一直不通时抛 ZiseaError。"""
        def handler(url, params):
            raise make_request_exception()

        with self.assertRaises(ZiseaError):
            make_client(FakeSession(handler), retries=2).get_json("/api/stats")


class SessionOwnershipTests(unittest.TestCase):
    def test_injected_session_is_not_closed(self) -> None:
        """外部注入的 session 归调用方所有，client.close() 不应关闭它。"""
        session = FakeSession([FakeResponse(200, {})])
        client = make_client(session)
        client.close()
        self.assertFalse(session.closed)

    def test_context_manager_closes_owned_session(self) -> None:
        """自建 session 时，退出 with 块要关闭它。"""
        client = ZiseaClient(base_url="http://example.test")
        fake = FakeSession([FakeResponse(200, {})])
        client._session = fake
        client._owns_session = True
        with client:
            pass
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
