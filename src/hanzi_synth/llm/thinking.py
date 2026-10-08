"""思考段剥离（硬规则：思考段一律不进产出）。

- **无条件切除**：不管模型有没有思考段，都先剥再判；
- **永不写入文件、不回灌、不参与校验**：合成产出里不允许出现思维链；
- 剥完为空、或残留**不闭合**的开标签（说明输出被截断）→ 该条判失败。

云端网关会把三种关思考的请求参数都忽略掉，所以**输出侧切除是无条件的**；本地端点则各家不同，
要关思考就得按**它自己的字段名**发（`secrets.toml` 的 provider `extra_body` 原样透传，
代码不认识也不适配任何具体开关名），关掉之后小模型不再把额度烧进思考段——实测同一份往返检验请求：
不发 325 秒且正文为空，发了 0.6 秒且判定完整。
"""

from __future__ import annotations

import re

# 成对标签：<think...>...</think...>，跨行匹配；同时兼容 <thinking> / <redacted_thinking>
_PAIR_RE = re.compile(r"<\s*(think|thinking|redacted_thinking)\s*>.*?<\s*/\s*\1\s*>", re.S | re.I)
# 未闭合的开标签：剥完还剩它 = 输出被截断
_OPEN_RE = re.compile(r"<\s*(think|thinking|redacted_thinking)\s*>", re.I)


def strip_thinking(text: str | None) -> tuple[str, bool]:
    """输入：模型原始输出（可能为 None）。输出：(剥离后的文本, 是否可用)。
    预期：`ok=False` 表示该条必须判失败——剥完为空，或有未闭合的开标签。
    """
    if not text:
        return "", False
    cleaned = _PAIR_RE.sub("", text)
    if _OPEN_RE.search(cleaned):
        return cleaned.strip(), False
    cleaned = cleaned.strip()
    return cleaned, bool(cleaned)
