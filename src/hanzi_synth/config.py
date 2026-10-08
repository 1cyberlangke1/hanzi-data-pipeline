"""本机配置读取（secrets.toml）。

约定：
- 凭据只存本机 secrets.toml，**不注入环境变量**；换机器手工放同名文件即可。
- 限额（rpm / batch）是**配置不是常量**：12 QPM 只是某个网关的观测值，换 provider 就改文件。
- 文件缺失或格式错时报人话，且**绝不回显 api_key**。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# 仓库根：src/hanzi_synth/config.py -> parents[0]=hanzi_synth, [1]=src, [2]=仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECRETS = REPO_ROOT / "secrets.toml"


@dataclass(frozen=True)
class ProviderConfig:
    """输入：secrets.toml 里一个 [[providers]] 块。输出：不可变的 provider 配置。
    预期：api_key 只留在内存；repr/日志里永远显示 ***，防止密钥泄漏到输出文件。
    """

    name: str
    base_url: str
    model: str
    api_key: str
    role: str = "verbalizer"
    rpm: int = 0  # 0 = 不限速（仅用于本地 mock）
    batch: int = 1
    # 该 provider 专属的请求体字段，**原样**并进请求 JSON。
    # 用途：不同提供商的「思考」开关根本不是同一个字段——有的端点认 `reasoning_effort`，
    # 别的端点可能认 `enable_thinking` / `chat_template_kwargs` / `thinking`……所以这里
    # 不做任何字段名适配，只做透传：secrets.toml 里怎么写，请求体里就怎么发。
    # 不配 = 请求体里没有这些字段（端点用模型自己的默认行为）。
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(name={self.name!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, role={self.role!r}, rpm={self.rpm}, "
            f"batch={self.batch}, extra_body={dict(self.extra_body)!r}, api_key='***')"
        )


class ConfigError(RuntimeError):
    """输入：配置问题描述。输出：可读异常。预期：调用方捕获后打印人话并退出码非 0。"""


def _extra_body(block: Mapping[str, Any]) -> dict[str, Any]:
    """输入：secrets.toml 里一个 provider 块。输出：要并进请求体的字段字典。
    预期：`[providers.extra_body]` 表里的键原样保留，值不改写（bool / str / 嵌套表都行）；
          顶层直接写 `reasoning_effort` 视为等价写法（老配置兼容，免得换写法时静默丢掉
          关思考的开关）；非表的值直接忽略——一行配置写错不该崩掉整条管线。
    """
    extra: dict[str, Any] = {}
    raw = block.get("extra_body")
    if isinstance(raw, dict):
        extra.update(raw)
    effort = block.get("reasoning_effort")
    if effort:
        extra.setdefault("reasoning_effort", effort)
    return extra


def load_providers(path: Path | str = DEFAULT_SECRETS, role: str | None = None) -> list[ProviderConfig]:
    """输入：secrets.toml 路径 + 可选 role 过滤。输出：ProviderConfig 列表（顺序即优先级）。
    预期：文件不存在时抛 ConfigError 并提示复制模板；缺字段的块被跳过而非崩溃。
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"找不到凭据文件 {path}。请在本机放一份 secrets.toml（该文件已被 .gitignore 覆盖，不入库）。"
        )
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path} 不是合法 TOML：{exc}") from exc

    providers: list[ProviderConfig] = []
    for block in data.get("providers", []):
        missing = [k for k in ("name", "base_url", "model", "api_key") if not block.get(k)]
        if missing:
            continue  # 跳过残缺块，不因为一行配置崩掉整条管线
        providers.append(
            ProviderConfig(
                name=str(block["name"]),
                base_url=str(block["base_url"]).rstrip("/"),
                model=str(block["model"]),
                api_key=str(block["api_key"]),
                role=str(block.get("role", "verbalizer")),
                rpm=int(block.get("rpm", 0) or 0),
                batch=max(1, int(block.get("batch", 1) or 1)),
                extra_body=_extra_body(block),
            )
        )
    if role is not None:
        providers = [p for p in providers if p.role == role]
    if not providers:
        raise ConfigError(f"{path} 里没有可用的 provider" + (f"（role={role}）" if role else ""))
    return providers


def default_data_path() -> Path:
    """输入：无。输出：爬取产出的 chars.jsonl 绝对路径。预期：只拼路径，不检查存在性。"""
    return REPO_ROOT / "crawler" / "data" / "chars.jsonl"
