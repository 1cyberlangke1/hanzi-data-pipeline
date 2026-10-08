"""抓取编排：枚举 -> 差异 -> 并发抓详情 -> 写入文件。

增量策略：
  1. 用 /api/groups + /api/z-range 轻量枚举出全站 ZIID（约 300 次请求）
  2. 与已有 chars.jsonl 的主键集合求差，只抓新增的 ZIID（详情请求才是重活）
  3. 与上一轮 index 对比，报告新增/下架的 ZIID
记录本身不加时间戳，保持与 API 返回逐字节可比，便于后续做 diff。

并发与限速默认值刻意保守（4 线程 + 每次请求前抖动延迟，见 cli.py 的 --delay），
避免对这个小站造成压力；要更快请自行评估后显式调高。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .api import MAX_PAGE_SIZE, ZiseaApi
from .store import JsonlStore, write_json_atomic

CHARS_FILE = "chars.jsonl"
CHARS_INDEX_FILE = "chars_index.jsonl"
STATE_FILE = "state.json"
FAILED_FILE = "failed_ziids.txt"

# 默认并发。站点体量小，4 线程配 CLI 的默认限速即可，不追求压满带宽。
DEFAULT_WORKERS = 4


@dataclass
class CrawlReport:
    """一次抓取的统计结果。输入：无。输出：可直接 json 序列化。"""

    target: str = ""
    enumerated: int = 0
    already_done: int = 0
    fetched: int = 0
    missing: int = 0
    failed: int = 0
    added: int = 0
    removed: int = 0
    elapsed_seconds: float = 0.0
    failed_ids: list[str] = field(default_factory=list)


def enumerate_light(api: ZiseaApi, page_size: int = MAX_PAGE_SIZE) -> list[dict]:
    """输入：API 与每页条数。输出：全站轻量记录列表（含 ziid，按系列/前缀/页顺序）。
    预期：约 300 次请求覆盖 14.5 万条，远轻于抓详情。
    """
    records: list[dict] = []
    for item in api.iter_ziids(page_size=page_size):
        ziid, prefix = item
        records.append({"ziid": ziid, "prefix": prefix})
    return records


def _diff_keys(old_keys: set[str], new_keys: set[str]) -> tuple[int, int]:
    """输入：旧、新主键集合。输出：(新增数, 下架数)。"""
    return len(new_keys - old_keys), len(old_keys - new_keys)


def run_chars(
    out_dir: str | Path,
    api: ZiseaApi,
    workers: int = DEFAULT_WORKERS,
    page_size: int = MAX_PAGE_SIZE,
    refresh: bool = False,
    limit: int | None = None,
    progress_every: int = 200,
    log: Callable[[str], None] = print,
) -> CrawlReport:
    """输入：输出目录、API、并发数、枚举页大小、是否强制重抓、最多抓多少条。
    输出：CrawlReport（含新增/下架/失败统计）。
    预期：中断后重跑自动跳过已入库 ZIID；单条失败只记录不中断整体。
    """
    started = time.monotonic()
    out_path = Path(out_dir)
    index_store = JsonlStore(out_path / CHARS_INDEX_FILE, key_field="ziid")
    target_path = out_path / CHARS_FILE
    if refresh:
        # 重抓时先写临时文件，全部抓完再原子替换，避免中途失败留下半新半旧的数据文件
        refresh_path = target_path.with_name(target_path.name + ".refresh")
        refresh_path.unlink(missing_ok=True)
        detail_store = JsonlStore(refresh_path, key_field="ziid")
    else:
        detail_store = JsonlStore(target_path, key_field="ziid")
        detail_store.repair_tail()
    index_store.repair_tail()

    log("[1/3] 枚举全站 ZIID（轻量）...")
    enumerated = enumerate_light(api, page_size=page_size)
    new_keys = {record["ziid"] for record in enumerated}
    old_index_keys = index_store.load_keys()
    added, removed = _diff_keys(old_index_keys, new_keys)
    index_store.replace_all(enumerated)
    log(f"      枚举到 {len(enumerated)} 条；相对上轮 新增 {added} / 下架 {removed}")

    report = CrawlReport(target="chars", enumerated=len(enumerated), added=added, removed=removed)

    done_keys = set() if refresh else detail_store.load_keys()
    todo = [record["ziid"] for record in enumerated if record["ziid"] not in done_keys]
    report.already_done = len(enumerated) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    fetched = 0
    if not todo:
        log("[2/3] 没有需要抓取的新 ZIID，跳过。")
    else:
        log(f"[2/3] 待抓 {len(todo)} 条（已入库 {report.already_done} 条），并发 {workers} ...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(api.character, ziid): ziid for ziid in todo}
            for future in as_completed(futures):
                ziid = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整体
                    report.failed += 1
                    report.failed_ids.append(ziid)
                    log(f"      [失败] {ziid}: {exc}")
                    continue
                if record is None:
                    report.missing += 1
                    continue
                detail_store.append(record)
                fetched += 1
                if fetched % progress_every == 0:
                    log(f"      已抓 {fetched}/{len(todo)}")

    report.fetched = fetched
    if refresh and report.enumerated > 0:
        # enumerated 为 0 说明枚举环节异常，此时宁可保留旧文件也不要用空文件覆盖
        os.replace(detail_store.path, target_path)
    report.elapsed_seconds = round(time.monotonic() - started, 1)
    return report


def write_state(out_dir: str | Path, payload: dict) -> None:
    """输入：输出目录与要合并进 state.json 的字段。输出：无。预期：与旧 state 合并后原子写。"""
    from .store import load_json

    out_path = Path(out_dir)
    state = load_json(out_path / STATE_FILE, default={}) or {}
    state.update(payload)
    write_json_atomic(out_path / STATE_FILE, state)


def write_failed(out_dir: str | Path, failed_ids: Iterable[str]) -> None:
    """输入：输出目录与失败 ZIID 列表。输出：无。预期：写 failed_ziids.txt，供下一轮重试。"""
    ids = list(failed_ids)
    path = Path(out_dir) / FAILED_FILE
    if not ids:
        path.unlink(missing_ok=True)
        return
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def report_to_dict(report: CrawlReport) -> dict:
    """输入：CrawlReport。输出：去掉失败明细的 dict（明细另写文件）。"""
    data = asdict(report)
    data.pop("failed_ids", None)
    return data
