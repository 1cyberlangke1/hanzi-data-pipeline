"""测试用的假 HTTP 层。

只实现 ZiseaClient 真正用到的接口（get / headers / close），
让单测不碰网络，也不依赖 requests 的真实 Session 行为。
"""

from __future__ import annotations

from typing import Any, Callable, Iterable


class FakeResponse:
    """输入：状态码、JSON 载荷、原始字节、是否模拟「不是 JSON」。输出：可被 client 消费的响应对象。"""

    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        invalid_json: bool = False,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json
        self.content = content

    def json(self) -> Any:
        """输入：无。输出：预置载荷；invalid_json 为真时抛 ValueError（模拟 HTML 错误页）。"""
        if self._invalid_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class FakeSession:
    """按顺序吐出预置响应；用完后重复最后一个，方便测「一直 500」这类场景。"""

    def __init__(self, responses: Iterable[FakeResponse] | Callable[[str, dict | None], FakeResponse]) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict | None]] = []
        self.closed = False
        self._responses = list(responses) if not callable(responses) else None
        self._handler = responses if callable(responses) else None

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> FakeResponse:
        """输入：URL 与查询参数。输出：下一个预置响应。预期：记录每次调用供断言。"""
        self.calls.append((url, params))
        if self._handler is not None:
            return self._handler(url, params)
        if not self._responses:
            raise AssertionError("FakeSession 没有更多预置响应了")
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def close(self) -> None:
        """输入：无。输出：无。预期：标记已关闭，供 client.close 的归属断言。"""
        self.closed = True


class RoutingSession:
    """按 URL 子串路由到对应响应工厂，用于模拟一整套 API。"""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.headers: dict[str, str] = {}
        self.routes = routes
        self.calls: list[tuple[str, dict | None]] = []
        self.closed = False

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> FakeResponse:
        """输入：URL 与查询参数。输出：第一个 URL 含路由键的响应。
        预期：路由值可以是 FakeResponse 或 (url, params) -> FakeResponse 的函数。
        """
        self.calls.append((url, params))
        for key, value in self.routes.items():
            if key in url:
                if callable(value):
                    return value(url, params)
                return value
        return FakeResponse(404, {"detail": f"no route for {url}"})

    def close(self) -> None:
        """输入：无。输出：无。"""
        self.closed = True


class NetworkError(Exception):
    """占位：真实实现里由 requests.RequestException 承担。"""


def make_request_exception() -> Exception:
    """输入：无。输出：一个真正的 requests.RequestException 实例（client 只捕获这个基类）。"""
    import requests

    return requests.ConnectionError("连接被重置")
