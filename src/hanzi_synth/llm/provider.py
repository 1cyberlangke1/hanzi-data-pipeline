"""多 provider 的 OpenAI 兼容 Chat 客户端。

设计要点（findings.md §9.2 / 主人指令）：
- **不设 `max_tokens`**：思考段会吃光额度；不设时模型自己以 finish_reason=stop 收尾；
- 限额来自配置（`ProviderConfig.rpm`），由 SlidingWindowLimiter 在客户端侧挡住，
  所以正常情况不会打到网关的 429；
- 429 / 5xx 做**有界退避重试**，仍失败则**故障转移到下一个 provider**；
- **引擎侧临时故障伪装成的 4xx 也重试**：本地推理服务在并发下会把「KV 缓存挤爆」这类
  引擎内部错误报成 HTTP 400（实测 LM Studio：「Context size has been exceeded」，
  而同一份 payload 单独重发立刻成功），所以按报文特征识别，不能只看状态码；
- 报错说人话，且**永不回显 api_key**。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import requests

from ..config import ProviderConfig
from .ratelimit import SlidingWindowLimiter
from .thinking import strip_thinking

# 报成 4xx 但其实是**引擎侧临时故障**的报文特征（小写比对）。
# 判据来自实测：LM Studio 在 4 路并发下偶发 `HTTP 400 ... Context size has been exceeded`，
# 而提示词只有 3000 字符、`loaded_context_length` 是 32000——不可能是真的超长；
# 同一份 payload 单独重发一次就成功，说明它是引擎侧的瞬态状态，重发有意义。
_TRANSIENT_4XX_MARKERS = ("context size has been exceeded",)


def is_transient_4xx(response: requests.Response) -> bool:
    """输入：HTTP 响应。输出：这个 4xx 是不是引擎侧临时故障（重发有意义）。

    预期：只在报文里出现**已知的临时故障特征**时判 True；认不出来一律按真 4xx 处理
          （真 4xx 重发是浪费配额，而且掩盖了请求本身的问题）。取不到正文也按真 4xx。
    """
    try:
        body = response.text or ""
    except Exception:  # noqa: BLE001 - 取不到正文不算致命，按真 4xx 处理
        return False
    lowered = body.lower()
    return any(marker in lowered for marker in _TRANSIENT_4XX_MARKERS)


@dataclass(frozen=True)
class Completion:
    """输入：一次调用的结果。输出：不可变结果对象。
    预期：`text` 已剥离思考段；`ok=False` 时 `error` 一定非空。
    """

    ok: bool
    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0


class ChatClient:
    """输入：provider 配置列表（顺序即优先级）+ 超时/重试参数。输出：可复用的 chat 客户端。
    预期：线程安全（限速器自带锁，requests.Session 每次调用独立），失败绝不抛到调用方。
    """

    def __init__(
        self,
        providers: Sequence[ProviderConfig],
        *,
        timeout: float = 180.0,
        max_retries: int = 2,
        backoff: float = 3.0,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个 provider 配置")
        self._providers = list(providers)
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff = backoff
        self._limiters = {p.name: SlidingWindowLimiter(p.rpm) for p in self._providers}

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.6,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        """输入：OpenAI 格式的 messages + 采样温度 + 额外字段。输出：Completion。
        预期：按 provider 顺序尝试；某个 provider 重试耗尽后换下一个；全部失败返回 ok=False。
        """
        last_error = "没有可用的 provider"
        total_attempts = 0
        for provider in self._providers:
            limiter = self._limiters.get(provider.name)
            for attempt in range(self._max_retries + 1):
                if limiter is not None:
                    limiter.acquire()
                total_attempts += 1
                try:
                    response = self._post(provider, messages, temperature, extra)
                except requests.RequestException as exc:
                    last_error = f"{provider.name}: 网络错误 {type(exc).__name__}"
                    time.sleep(self._backoff * (attempt + 1))
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"{provider.name}: HTTP {response.status_code}"
                    time.sleep(self._backoff * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    last_error = f"{provider.name}: HTTP {response.status_code} {self._safe_message(response)}"
                    if is_transient_4xx(response):
                        # 引擎侧瞬态：退避重发同一份请求（实测重发即成功），别把它当成拒收理由
                        time.sleep(self._backoff * (attempt + 1))
                        continue
                    break  # 4xx 是请求本身的问题，换 provider 也没用，但仍尝试下一个
                return self._parse(provider, response, total_attempts)

        return Completion(ok=False, text="", provider="", model="", error=last_error, attempts=total_attempts)

    # ---------- 内部 ----------

    def _post(
        self,
        provider: ProviderConfig,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        extra: Mapping[str, Any] | None,
    ) -> requests.Response:
        """输入：provider 与请求内容。输出：HTTP 响应。
        预期：请求体**不含 max_tokens**；api_key 只出现在请求头，不进日志；
              provider 的 `extra_body` 原样并进请求体（不配就不发这些字段，
              不认它们的端点不会因为未知字段吃 400）。
        """
        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if provider.extra_body:
            # provider 专属字段（不同端点的「思考」开关字段名不一样，所以只透传、不适配）。
            # 放在 extra 之前：调用方这一次显式给的 extra 优先级更高。
            payload.update(provider.extra_body)
        if extra:
            payload.update(extra)
        return requests.post(
            f"{provider.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._timeout,
        )

    def _parse(self, provider: ProviderConfig, response: requests.Response, attempts: int) -> Completion:
        """输入：provider 与 2xx 响应。输出：Completion。
        预期：结构异常按 ok=False 返回；思考段在这里被无条件剥离。
        """
        try:
            data = response.json()
        except ValueError:
            return Completion(ok=False, text="", provider=provider.name, model=provider.model,
                              error=f"{provider.name}: 响应不是 JSON", attempts=attempts)
        choices = data.get("choices") or []
        if not choices:
            return Completion(ok=False, text="", provider=provider.name, model=provider.model,
                              error=f"{provider.name}: 响应没有 choices", attempts=attempts)
        message = choices[0].get("message") or {}
        raw = message.get("content") or ""
        text, usable = strip_thinking(raw)
        return Completion(
            ok=usable,
            text=text,
            provider=provider.name,
            model=provider.model,
            finish_reason=choices[0].get("finish_reason"),
            usage=data.get("usage") or {},
            error=None if usable else f"{provider.name}: 剥离思考段后无有效内容",
            attempts=attempts,
        )

    @staticmethod
    def _safe_message(response: requests.Response) -> str:
        """输入：错误响应。输出：截断后的可读信息。预期：不回显任何凭据（响应体里没有 key）。"""
        try:
            body = response.text or ""
        except Exception:  # noqa: BLE001 - 取不到正文不算致命
            return ""
        return body[:200].replace("\n", " ")
