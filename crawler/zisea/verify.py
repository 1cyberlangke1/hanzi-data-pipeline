"""抓取完整性核对：证明「站点收录的每一个 ZIID 都已入库」。

对账口径全部取自站点权威接口：
  /api/stats   -> 全站总数、各系列总数、有 IDS 的字数
  /api/groups  -> 各系列下每个前缀桶的收录数
本地侧只读产出文件：chars.jsonl / chars_index.jsonl / sprites/。

核心结论来自「逐前缀计数」：站点把收录的字切成 prefix 桶（实测 U 组 3 位、Z/P 组 2 位，
如 U04 / Z4 / PA），只要每个桶的本地计数都等于站点计数，就说明枚举入口覆盖了全站，
不存在漏抓；其余检查用于交叉印证。

输入：产出目录、API。
输出：VerifyReport（逐项结果 + 是否全绿）。
预期：只读，不修改任何文件；任一项失败即 ok=False。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .api import ZiseaApi
from .crawl import CHARS_FILE, CHARS_INDEX_FILE
from .sprites import SPRITES_DIR
from .store import JsonlStore

# chars.jsonl 每条记录必须有的顶层字段（只查键存在，值可以为空字典）。
REQUIRED_FIELDS = ("property", "meanings")


@dataclass
class Check:
    """一项检查的结果。输入：无。输出：可直接打印/序列化。"""

    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict:
        """输入：无。输出：{name, ok, detail}。"""
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class VerifyReport:
    """一次完整核对的结果。输入：无。输出：ok 表示是否全绿。"""

    ok: bool = True
    checks: list[Check] = field(default_factory=list)
    totals: dict = field(default_factory=dict)


@dataclass
class DetailScan:
    """chars.jsonl 的一次流式扫描结果。输入：无。输出：各项统计。"""

    total_lines: int = 0
    keys: set[str] = field(default_factory=set)
    duplicates: list[str] = field(default_factory=list)
    bad_lines: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    series_counts: dict[str, int] = field(default_factory=dict)
    prefix_counts: dict[str, int] = field(default_factory=dict)
    with_ids: int = 0
    needs_sprite: set[str] = field(default_factory=set)


def prefix_lengths(groups: dict) -> dict[str, int]:
    """输入：/api/groups 返回。输出：{系列字母: 前缀长度}。
    预期：同一系列内前缀长度一致（实测 U 组 3 位、Z/P 组 2 位）；不一致时抛
          ValueError —— 逐前缀对账口径失效时必须报错，而不是给出错误的「全绿」。
    """
    lengths: dict[str, int] = {}
    for series in groups.get("series") or []:
        letter = str(series.get("series") or "")
        for group in series.get("groups") or []:
            prefix = str(group.get("prefix") or "")
            if not prefix:
                continue
            current = lengths.setdefault(letter, len(prefix))
            if current != len(prefix):
                raise ValueError(
                    f"系列 {letter} 的前缀长度不一致（{current} vs {len(prefix)}），逐前缀对账口径失效"
                )
    return lengths


def scan_details(path: str | Path, lengths: dict[str, int]) -> DetailScan:
    """输入：chars.jsonl 路径、各系列前缀长度。输出：DetailScan。
    预期：流式读取，14.5 万条也不吃内存；坏行只记录不中断整批。
    """
    scan = DetailScan()
    file_path = Path(path)
    if not file_path.exists():
        return scan

    with file_path.open("rb") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            scan.total_lines += 1
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                scan.bad_lines.append(f"第 {line_no} 行: {exc}")
                continue

            ziid = str(record.get("ziid") or "")
            if not ziid:
                scan.missing_fields.append(f"第 {line_no} 行: 缺 ziid")
                continue

            if ziid in scan.keys:
                scan.duplicates.append(ziid)
            scan.keys.add(ziid)

            absent = [name for name in REQUIRED_FIELDS if name not in record]
            if absent:
                scan.missing_fields.append(f"{ziid}: 缺 {'/'.join(absent)}")

            series = ziid[:1]
            scan.series_counts[series] = scan.series_counts.get(series, 0) + 1
            width = lengths.get(series)
            if width:
                prefix = ziid[:width]
                scan.prefix_counts[prefix] = scan.prefix_counts.get(prefix, 0) + 1

            prop = record.get("property") or {}
            if prop.get("ids"):
                scan.with_ids += 1
            if not prop.get("zi"):
                scan.needs_sprite.add(ziid)

    return scan


def _compare_counts(local: dict[str, int], expected: dict[str, int]) -> tuple[dict, dict]:
    """输入：本地计数、站点计数。输出：(计数不符项, 站点没有而本地多出的项)。
    预期：两个返回值都为空即完全一致。
    """
    mismatch = {
        key: {"local": local.get(key, 0), "site": value}
        for key, value in expected.items()
        if local.get(key, 0) != value
    }
    extra = {key: value for key, value in local.items() if key not in expected}
    return mismatch, extra


def run_verify(
    out_dir: str | Path,
    api: ZiseaApi,
    log: Callable[[str], None] = print,
) -> VerifyReport:
    """输入：产出目录、API。输出：VerifyReport。
    预期：先取站点权威计数，再扫本地产出，最后逐项对账；不写任何文件。
    """
    out_path = Path(out_dir)
    log("[1/2] 取站点权威计数（/api/stats + /api/groups）...")
    stats = api.stats()
    groups = api.groups()
    lengths = prefix_lengths(groups)

    log("[2/2] 扫描本地产出（chars.jsonl / chars_index.jsonl / sprites/）...")
    scan = scan_details(out_path / CHARS_FILE, lengths)
    index_keys = JsonlStore(out_path / CHARS_INDEX_FILE, key_field="ziid").load_keys()
    sprite_dir = out_path / SPRITES_DIR
    sprite_files = {path.name for path in sprite_dir.glob("*.png")} if sprite_dir.is_dir() else set()

    checks: list[Check] = []

    expected_total = int(stats.get("total_characters", 0))
    checks.append(
        Check(
            "详情条数",
            scan.total_lines == expected_total,
            f"本地 {scan.total_lines} 条 / 站点 {expected_total} 条",
        )
    )

    checks.append(
        Check(
            "重复 ZIID",
            not scan.duplicates,
            "无重复" if not scan.duplicates else f"{len(scan.duplicates)} 个，例如 {scan.duplicates[:5]}",
        )
    )

    malformed = len(scan.bad_lines) + len(scan.missing_fields)
    sample = (scan.bad_lines + scan.missing_fields)[:3]
    checks.append(
        Check(
            "行格式与必需字段",
            malformed == 0,
            "全部合法" if malformed == 0 else f"{malformed} 条异常，例如 {sample}",
        )
    )

    expected_series = {str(k): int(v) for k, v in (stats.get("series") or {}).items()}
    series_bad, series_extra = _compare_counts(scan.series_counts, expected_series)
    checks.append(
        Check(
            "各系列计数",
            not series_bad and not series_extra,
            f"逐系列一致 {scan.series_counts}"
            if not series_bad and not series_extra
            else f"不符 {series_bad} 多出 {series_extra}",
        )
    )

    expected_prefix: dict[str, int] = {}
    for series in groups.get("series") or []:
        for group in series.get("groups") or []:
            prefix = str(group.get("prefix") or "")
            if prefix:
                expected_prefix[prefix] = int(group.get("count", 0))
    prefix_bad, prefix_extra = _compare_counts(scan.prefix_counts, expected_prefix)
    checks.append(
        Check(
            "逐前缀计数（完整性核心）",
            not prefix_bad and not prefix_extra,
            f"{len(expected_prefix)} 个前缀桶全部对上"
            if not prefix_bad and not prefix_extra
            else f"不符 {prefix_bad} 多出 {prefix_extra}",
        )
    )

    missing_in_detail = sorted(index_keys - scan.keys)
    missing_in_index = sorted(scan.keys - index_keys)
    checks.append(
        Check(
            "chars_index 与 chars 主键一致",
            not missing_in_detail and not missing_in_index,
            f"两侧各 {len(scan.keys)} 个 ZIID 完全一致"
            if not missing_in_detail and not missing_in_index
            else f"详情缺 {len(missing_in_detail)} 个、索引缺 {len(missing_in_index)} 个",
        )
    )

    expected_ids = int((stats.get("ids") or {}).get("with_ids", 0))
    checks.append(
        Check(
            "有 IDS 的字数",
            scan.with_ids == expected_ids,
            f"本地 {scan.with_ids} / 站点 {expected_ids}",
        )
    )

    expected_sprites = {f"{ziid}.png" for ziid in scan.needs_sprite}
    sprite_missing = sorted(expected_sprites - sprite_files)
    sprite_orphan = sorted(sprite_files - expected_sprites)
    checks.append(
        Check(
            "精灵图齐全",
            not sprite_missing and not sprite_orphan,
            f"{len(expected_sprites)} 张全部就位"
            if not sprite_missing and not sprite_orphan
            else f"缺 {len(sprite_missing)} 张、多 {len(sprite_orphan)} 张",
        )
    )

    report = VerifyReport(
        ok=all(check.ok for check in checks),
        checks=checks,
        totals={
            "detail_records": scan.total_lines,
            "unique_ziids": len(scan.keys),
            "series": scan.series_counts,
            "prefixes": len(scan.prefix_counts),
            "with_ids": scan.with_ids,
            "sprites": len(sprite_files),
        },
    )

    for check in checks:
        mark = "通过" if check.ok else "失败"
        log(f"      [{mark}] {check.name}：{check.detail}")
    log(f"      结论：{'全部通过，站点收录的字已全量入库' if report.ok else '存在不符项，见上'}")
    return report


def verify_report_to_dict(report: VerifyReport) -> dict:
    """输入：VerifyReport。输出：可 json 序列化的 dict。"""
    return {
        "ok": report.ok,
        "checks": [check.to_dict() for check in report.checks],
        "totals": report.totals,
    }


__all__ = [
    "Check",
    "DetailScan",
    "VerifyReport",
    "prefix_lengths",
    "run_verify",
    "scan_details",
    "verify_report_to_dict",
]
