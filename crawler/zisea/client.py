"""字海网（新叶典）HTTP 客户端。

职责：统一基址 / 请求头 / 重试 / 限速，并把 HTTP 层异常收敛成 ZiseaError 子类。
不负责：业务字段的解析与校验（见 api.py）。

注意：站点 https 在当前网络下 TLS 握手会被干扰（curl rc=35），http 正常，
因此默认基址用 http://。
"""

from __future__ import annotations

import random
import time
from typing import Any

import requests

# 站点默认基址。https 在当前网络不可用，见模块 docstring。
DEFAULT_BASE_URL = "http://zisea.com"

# 默认 UA 不冒充浏览器：公开仓库里的爬虫应当能被站点识别出来（实测该 UA 返回 200）。
# 需要自定义（例如带上联系方式）时用 --user-agent 覆盖。
DEFAULT_USER_AGENT = "zisea-offline-crawler/1.0 (personal offline study)"


class ZiseaError(RuntimeError):
    """字海网请求失败：网络异常，或重试耗尽后仍非预期状态码。"""


class NotFoundError(ZiseaError):
    """资源不存在（HTTP 404）。调用方可据此判断「该 ZIID 确实没收录」。"""


class BadRequestError(ZiseaError):
    """请求参数不合法（HTTP 400 / 422）。属于程序 bug，不做重试。"""


class ZiseaClient:
    """带重试与限速的 JSON 客户端。

    输入：基址、超时、重试次数、退避基数、每次请求前的固定延迟、可注入的 session。
    输出：get_json 返回解析后的 JSON（dict 或 list）。
    预期：4xx 直接抛错不重试；5xx 与网络异常按指数退避重试；重试耗尽抛 ZiseaError。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        delay: float = 0.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.backoff = backoff
        self.delay = delay
        self.user_agent = user_agent
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

    def close(self) -> None:
        """输入：无。输出：无。预期：关闭自建的 session，外部注入的 session 不动。"""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "ZiseaClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def get_json(self, path: str, params: dict | None = None) -> Any:
        """输入：API 路径（如 "/api/z/U04E00"）与查询参数 dict。
        输出：解析后的 JSON 对象（dict 或 list）。
        预期：404 -> NotFoundError；400/422 -> BadRequestError；其余失败重试耗尽 -> ZiseaError。
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self.retries):
            if self.delay > 0:
                # 加一点抖动，避免多线程同时打点形成脉冲
                time.sleep(self.delay * random.uniform(0.5, 1.5))
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        last_error = ZiseaError(f"{url} 返回的不是合法 JSON: {exc}")
                elif status == 404:
                    raise NotFoundError(f"资源不存在(404): {url}")
                elif status in (400, 422):
                    raise BadRequestError(f"参数不合法({status}): {url} params={params}")
                else:
                    last_error = ZiseaError(f"非预期状态码 {status}: {url}")

            if attempt < self.retries - 1:
                time.sleep(self.backoff * (2**attempt))

        raise ZiseaError(f"请求失败（已重试 {self.retries} 次）: {url} 最后错误: {last_error}")

    def get_bytes(self, path: str) -> bytes:
        """输入：资源路径（如 "/sprites/Z43651.png"）。
        输出：原始字节内容。
        预期：404 -> NotFoundError（该字有 Unicode 字形，站点不提供图片）；
              其余失败重试耗尽 -> ZiseaError。
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self.retries):
            if self.delay > 0:
                time.sleep(self.delay * random.uniform(0.5, 1.5))
            try:
                response = self._session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                status = response.status_code
                if status == 200:
                    return response.content
                if status == 404:
                    raise NotFoundError(f"资源不存在(404): {url}")
                if status in (400, 422):
                    raise BadRequestError(f"参数不合法({status}): {url}")
                last_error = ZiseaError(f"非预期状态码 {status}: {url}")

            if attempt < self.retries - 1:
                time.sleep(self.backoff * (2**attempt))

        raise ZiseaError(f"请求失败（已重试 {self.retries} 次）: {url} 最后错误: {last_error}")
