"""字海网爬虫命令行入口。

用法示例：
  python crawler/crawl_zisea.py stats                     查看全站统计（含 daily 增量线索）
  python crawler/crawl_zisea.py enumerate                 只枚举全站 ZIID，写 chars_index.jsonl
  python crawler/crawl_zisea.py chars --limit 200         小范围试抓 200 条详情
  python crawler/crawl_zisea.py chars                     增量抓取全部缺失详情
  python crawler/crawl_zisea.py sprites                   下载无码位字的字形图片
  python crawler/crawl_zisea.py verify                    核对站点收录的字是否已全部入库
  python crawler/crawl_zisea.py view                      启动离线查看器

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import MAX_PAGE_SIZE, ZiseaApi
from .client import DEFAULT_BASE_URL, DEFAULT_USER_AGENT, ZiseaClient
from .crawl import (
    CHARS_FILE,
    CHARS_INDEX_FILE,
    DEFAULT_WORKERS,
    report_to_dict,
    run_chars,
    write_failed,
    write_state,
)
from .robots import check_robots
from .store import JsonlStore

# 默认输出到 crawler/data/，与爬虫代码放在一起，避免和仓库根目录混在一起。
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data"

# 需要访问站点的子命令：这些命令一律先过 robots.txt 校验。view 只读本地数据，不在此列。
NETWORK_COMMANDS = frozenset({"stats", "enumerate", "chars", "sprites", "verify"})


def _add_common(parser: argparse.ArgumentParser) -> None:
    """输入：子命令 parser。输出：无。预期：挂上所有子命令共用的连接与输出参数。"""
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 crawler/data）")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="站点基址")
    parser.add_argument("--timeout", type=float, default=30.0, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("--delay", type=float, default=0.5, help="每次请求前的固定延迟秒数（默认 0.5，限速）")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="请求 UA（默认标识本工具，不冒充浏览器）")
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="跳过 robots.txt 检查（仅在你已获授权或已自行评估风险时使用）",
    )


def build_parser() -> argparse.ArgumentParser:
    """输入：无。输出：配置好的 ArgumentParser。"""
    parser = argparse.ArgumentParser(prog="crawl_zisea", description="字海网（新叶典）爬虫")
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="打印 /api/stats 全站统计")
    _add_common(p_stats)

    p_enum = sub.add_parser("enumerate", help="只枚举全站 ZIID（轻量），写 chars_index.jsonl")
    _add_common(p_enum)
    p_enum.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE, help="枚举每页条数（<=500）")

    p_chars = sub.add_parser("chars", help="抓取字符详情（增量）")
    _add_common(p_chars)
    p_chars.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    p_chars.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE, help="枚举每页条数（<=500）")
    p_chars.add_argument("--refresh", action="store_true", help="忽略已入库记录，全部重抓")
    p_chars.add_argument("--limit", type=int, default=None, help="本次最多抓多少条（试抓用）")

    p_verify = sub.add_parser("verify", help="核对站点收录的字是否已全部入库（只读，不写产出）")
    _add_common(p_verify)

    p_view = sub.add_parser("view", help="启动离线查看器（本机网页界面）")
    _add_common(p_view)
    p_view.add_argument("--host", default="127.0.0.1", help="监听地址（默认只监听本机）")
    p_view.add_argument("--port", type=int, default=8720, help="监听端口")
    p_view.add_argument("--rebuild", action="store_true", help="强制重建查看器索引")
    p_view.add_argument(
        "--allow-origin-fetch",
        action="store_true",
        help="本地缺精灵图时回源站点取（默认关，保持纯离线）",
    )

    p_sprite = sub.add_parser("sprites", help="下载无 Unicode 码位字的字形图片（增量）")
    _add_common(p_sprite)
    p_sprite.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    p_sprite.add_argument("--limit", type=int, default=None, help="本次最多下载多少张")

    return parser


def _guard_robots(client: ZiseaClient, args: argparse.Namespace) -> bool:
    """输入：HTTP 客户端与解析后的参数。输出：True 表示可以继续访问站点。
    预期：默认先读站点 robots.txt；被禁止时打印原因与替代做法并返回 False；
          --ignore-robots 时只提示一次风险后放行。
    """
    if args.ignore_robots:
        print(
            "[注意] 已跳过 robots.txt 检查（--ignore-robots）："
            "请自行确认你已获得站点授权，或本次使用属于个人学习研究。"
        )
        return True

    decision = check_robots(client, base_url=args.base_url, user_agent=args.user_agent)
    if decision.allowed:
        print(f"robots: {decision.reason}")
        return True

    print(f"[中止] 已按 robots.txt 拒绝执行：{decision.reason}", file=sys.stderr)
    if decision.raw_text.strip():
        print("       robots.txt 原文：", file=sys.stderr)
        for line in decision.raw_text.strip().splitlines():
            print(f"         {line}", file=sys.stderr)
    print(
        "       说明：robots.txt 是站点所有者对自动抓取的明确声明。若你已获得授权，或确认本次\n"
        "             使用属于个人学习研究，请自行评估后加 --ignore-robots 重跑。",
        file=sys.stderr,
    )
    return False


def main(argv: list[str] | None = None) -> int:
    """输入：命令行参数（默认取 sys.argv）。输出：进程退出码（0 成功，1 失败）。
    预期：任何异常都转成人话输出，不抛裸 traceback。
    """
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = ZiseaClient(
        base_url=args.base_url,
        timeout=args.timeout,
        retries=args.retries,
        delay=args.delay,
        user_agent=args.user_agent,
    )
    api = ZiseaApi(client)

    try:
        if args.command in NETWORK_COMMANDS and not _guard_robots(client, args):
            return 3

        if args.command == "stats":
            data = api.stats()
            print(json.dumps(data, ensure_ascii=False, indent=2))
            write_state(out_dir, {"last_stats": data, "sprite_version": api.sprite_version()})
            return 0

        if args.command == "enumerate":
            from .crawl import enumerate_light

            records = enumerate_light(api, page_size=args.page_size)
            store = JsonlStore(out_dir / CHARS_INDEX_FILE, key_field="ziid")
            store.replace_all(records)
            print(f"已枚举 {len(records)} 条 ZIID -> {store.path}")
            return 0

        if args.command == "chars":
            report = run_chars(
                out_dir,
                api,
                workers=args.workers,
                page_size=args.page_size,
                refresh=args.refresh,
                limit=args.limit,
            )
            write_failed(out_dir, report.failed_ids)
            write_state(
                out_dir,
                {
                    "chars": report_to_dict(report),
                    "chars_total": JsonlStore(out_dir / CHARS_FILE, key_field="ziid").count(),
                    "sprite_version": api.sprite_version(),
                },
            )
            print(json.dumps(report_to_dict(report), ensure_ascii=False, indent=2))
            return 1 if report.failed else 0

        if args.command == "view":
            from .viewer import serve

            client.close()
            serve(
                out_dir / CHARS_FILE,
                host=args.host,
                port=args.port,
                rebuild=args.rebuild,
                allow_origin_fetch=args.allow_origin_fetch,
            )
            return 0

        if args.command == "sprites":
            from .sprites import run_sprites, sprite_report_to_dict

            report = run_sprites(
                out_dir,
                client,
                chars_path=out_dir / CHARS_FILE,
                workers=args.workers,
                limit=args.limit,
            )
            write_failed(out_dir, report.failed_ids)
            payload = sprite_report_to_dict(report)
            write_state(out_dir, {"sprites": payload})
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1 if report.failed else 0

        if args.command == "verify":
            from .verify import run_verify, verify_report_to_dict

            report = run_verify(out_dir, api)
            payload = verify_report_to_dict(report)
            write_state(out_dir, {"verify": payload})
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if report.ok else 1

        print(f"未知命令: {args.command}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断，已入库的数据不会丢，重跑会自动续爬。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - 顶层保底，输出人话
        print(f"抓取失败：{exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
