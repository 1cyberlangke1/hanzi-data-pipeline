"""按请求数计的滑动窗口限速。

实测（findings.md §9.2）：白山 edgefn 网关是 **12 QPM / 滑动 60s 窗口 / 按请求数计**，
无 `x-ratelimit-*` 头、无 `Retry-After`。这里把它实现成**可配置的客户端限速**——
12 只是 secrets.toml 里的一个值，不是写死的常量。
"""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """输入：每分钟请求上限与窗口秒数（rpm <= 0 表示不限速）。输出：线程安全的 acquire()。
    预期：任意 60s 窗口内放行的请求数不超过 rpm；超限时阻塞到最早那次请求滑出窗口，
          不抛异常、不丢请求。
    """

    def __init__(self, rpm: int, window: float = 60.0) -> None:
        self._rpm = rpm
        self._window = window
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """输入：无。输出：无（拿到配额后返回）。
        预期：调用方拿到返回值即可立即发请求；rpm<=0 时直接返回。
        """
        if self._rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= self._window:
                    self._times.popleft()
                if len(self._times) < self._rpm:
                    self._times.append(now)
                    return
                wait = self._window - (now - self._times[0])
            time.sleep(max(wait, 0.05))
