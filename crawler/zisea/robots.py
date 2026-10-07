"""robots.txt 校验。

公开仓库里的爬虫默认尊重站点 robots.txt（RFC 9309）：任何抓取动作之前先读
{base_url}/robots.txt，命中 Disallow 就中止，除非调用方显式加 --ignore-robots
声明自己已获授权、或已自行评估过风险。

现状（2026-10 实测）：zisea.com 与 yedict.com 的 robots.txt 都是
「User-agent: * / Disallow: /」，即全站禁止自动抓取，所以默认路径会拒绝执行 ——
这是预期行为，不是故障。

输入：HTTP 客户端、站点基址、User-Agent、待检查路径。
输出：RobotsDecision（是否放行、原因、robots.txt 原文）。
预期：每次判定只发一个请求；取不到 robots.txt 时保守拒绝，不默认放行。
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

from .client import DEFAULT_BASE_URL, DEFAULT_USER_AGENT, NotFoundError, ZiseaClient, ZiseaError

# robots.txt 的固定路径。
ROBOTS_PATH = "/robots.txt"


@dataclass(frozen=True)
class RobotsDecision:
    """一次 robots.txt 判定结果。输入：无。输出：可直接打印。"""

    allowed: bool
    reason: str
    robots_url: str
    status: int | None = None
    raw_text: str = ""


def robots_url_for(base_url: str) -> str:
    """输入：站点基址。输出：该站 robots.txt 的绝对 URL。"""
    return urljoin(base_url.rstrip("/") + "/", ROBOTS_PATH.lstrip("/"))


def fetch_robots(client: ZiseaClient, base_url: str) -> tuple[int, str]:
    """输入：HTTP 客户端、站点基址。输出：(HTTP 状态码, robots.txt 文本)。
    预期：404 表示站点没有 robots.txt（RFC 9309 视为不限制），返回 (404, "")；
          其余网络异常抛 ZiseaError，由调用方决定保守策略。
    """
    try:
        payload = client.get_bytes(ROBOTS_PATH)
    except NotFoundError:
        return 404, ""
    return 200, payload.decode("utf-8", errors="replace")


def check_robots(
    client: ZiseaClient,
    base_url: str = DEFAULT_BASE_URL,
    user_agent: str = DEFAULT_USER_AGENT,
    path: str = "/",
) -> RobotsDecision:
    """输入：HTTP 客户端、站点基址、User-Agent、待检查路径。
    输出：RobotsDecision。
    预期：用标准库 RobotFileParser 判定；取不到 robots.txt 时拒绝（宁可不抓）。
    """
    url = robots_url_for(base_url)
    try:
        status, text = fetch_robots(client, base_url)
    except ZiseaError as exc:
        return RobotsDecision(
            allowed=False,
            reason=f"无法获取 {url}（{exc}），无法确认站点意愿，保守起见不抓取",
            robots_url=url,
        )

    parser = RobotFileParser()
    parser.parse(text.splitlines())
    target = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))

    if parser.can_fetch(user_agent, target):
        note = "站点未提供 robots.txt，视为不限制" if status == 404 else "robots.txt 未禁止本次访问"
        return RobotsDecision(True, f"{note}（{url}）", url, status=status, raw_text=text)

    return RobotsDecision(
        allowed=False,
        reason=f"{url} 禁止 User-Agent「{user_agent}」访问 {path}",
        robots_url=url,
        status=status,
        raw_text=text,
    )


__all__ = ["ROBOTS_PATH", "RobotsDecision", "check_robots", "fetch_robots", "robots_url_for"]
