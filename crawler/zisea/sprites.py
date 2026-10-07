"""精灵图下载：给「没有 Unicode 码位」的字补上字形图片。

背景：U 组字符本身就是 Unicode 码位，直接用字体渲染；Z 组（中华字海所见字形）
和 P 组（提案所见字形）没有对应码位，站点只提供 /sprites/{ZIID}.png，
离线查看时必须本地留一份，否则界面上只能显示占位符。

输入：crawler/data/chars.jsonl
输出：crawler/data/sprites/{ZIID}.png
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from .client import NotFoundError, ZiseaClient
from .crawl import DEFAULT_WORKERS, CrawlReport

SPRITES_DIR = "sprites"


@dataclass
class SpriteReport:
    """一次精灵图抓取的统计。输入：无。输出：可 json 序列化。"""

    needed: int = 0
    already_have: int = 0
    downloaded: int = 0
    unavailable: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    failed_ids: list[str] = field(default_factory=list)


def iter_sprite_ziids(chars_path: str | Path) -> Iterator[str]:
    """输入：chars.jsonl 路径。输出：迭代所有 property.zi 为空（无 Unicode 字形）的 ZIID。
    预期：流式读取，不把 14 万条记录读进内存。
    """
    import json

    path = Path(chars_path)
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except Exception:  # noqa: BLE001 - 脏行跳过，不阻塞整批
                continue
            prop = record.get("property") or {}
            if not prop.get("zi") and record.get("ziid"):
                yield str(record["ziid"])


def run_sprites(
    out_dir: str | Path,
    client: ZiseaClient,
    chars_path: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    progress_every: int = 500,
    log: Callable[[str], None] = print,
) -> SpriteReport:
    """输入：输出目录、HTTP 客户端、数据文件、并发数、条数上限。
    输出：SpriteReport。
    预期：已存在的 PNG 直接跳过，可反复运行；单张失败只记录不中断。
    """
    started = time.monotonic()
    out_path = Path(out_dir)
    sprite_dir = out_path / SPRITES_DIR
    sprite_dir.mkdir(parents=True, exist_ok=True)
    source = Path(chars_path) if chars_path else out_path / "chars.jsonl"

    log("[1/2] 扫描需要字形图片的字...")
    needed = list(iter_sprite_ziids(source))
    report = SpriteReport(needed=len(needed))

    todo = [ziid for ziid in needed if not (sprite_dir / f"{ziid}.png").exists()]
    report.already_have = len(needed) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    if not todo:
        log("[2/2] 没有需要下载的图片。")
        report.elapsed_seconds = round(time.monotonic() - started, 1)
        return report

    log(f"[2/2] 待下载 {len(todo)} 张（已有 {report.already_have} 张），并发 {workers} ...")
    done = 0

    def fetch(ziid: str) -> tuple[str, bytes | None]:
        """输入：ZIID。输出：(ZIID, 图片字节或 None)。预期：404 表示该字有字形，不算失败。"""
        try:
            return ziid, client.get_bytes(f"/sprites/{ziid}.png")
        except NotFoundError:
            return ziid, None
        except Exception:  # noqa: BLE001 - 交给上层计数
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, ziid): ziid for ziid in todo}
        for future in as_completed(futures):
            ziid = futures[future]
            try:
                _, payload = future.result()
            except Exception as exc:  # noqa: BLE001 - 单张失败不中断
                report.failed += 1
                report.failed_ids.append(ziid)
                log(f"      [失败] {ziid}: {exc}")
                continue
            if payload is None:
                report.unavailable += 1
                continue
            (sprite_dir / f"{ziid}.png").write_bytes(payload)
            report.downloaded += 1
            done += 1
            if done % progress_every == 0:
                log(f"      已下载 {done}/{len(todo)}")

    report.elapsed_seconds = round(time.monotonic() - started, 1)
    return report


def sprite_report_to_dict(report: SpriteReport) -> dict:
    """输入：SpriteReport。输出：去掉失败明细的 dict。"""
    return {
        "needed": report.needed,
        "already_have": report.already_have,
        "downloaded": report.downloaded,
        "unavailable": report.unavailable,
        "failed": report.failed,
        "elapsed_seconds": report.elapsed_seconds,
    }


__all__ = ["SPRITES_DIR", "SpriteReport", "iter_sprite_ziids", "run_sprites", "sprite_report_to_dict", "CrawlReport"]
