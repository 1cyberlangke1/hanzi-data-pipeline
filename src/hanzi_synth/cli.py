"""合成管线命令行入口。

用法：
    .\.venv\Scripts\python.exe -m hanzi_synth.cli list            # 列出题型
    .\.venv\Scripts\python.exe -m hanzi_synth.cli dry-run         # 只跑 L0/L2，不联网
    .\.venv\Scripts\python.exe -m hanzi_synth.cli run --limit 40  # 端到端小批量
    .\.venv\Scripts\python.exe -m hanzi_synth.cli reject-stats    # 只读：rejects 按防线汇总
    .\.venv\Scripts\python.exe -m hanzi_synth.cli attempt-stats   # 只读：逐轮判定，看首轮死在哪
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# 允许 `python -m hanzi_synth.cli` 在未安装包时直接跑（src 加入搜索路径）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hanzi_synth.config import ConfigError, default_data_path, load_providers
from hanzi_synth.facts.loader import DataError
from hanzi_synth.pipeline.dedup import (
    DEFAULT_NGRAM,
    DEFAULT_THRESHOLD,
    contamination_file,
    dedup_file,
    write_kept,
    write_report,
)
from hanzi_synth.pipeline.audit import (
    UNIQUE_MODES,
    audit_sources,
    collect_sources,
    is_published_source,
    write_audit_report,
    write_jsonl,
)
from hanzi_synth.pipeline.difficulty import estimate_irt_from_responses, load_responses, write_calibration
from hanzi_synth.pipeline.roundtrip import run_roundtrip
from hanzi_synth.pipeline.run import (
    DEFAULT_OUT,
    PipelineBusy,
    PipelineConfig,
    coverage_report,
    replay_rejects,
    run_pipeline,
    summarize_attempts,
    summarize_rejects,
)
from hanzi_synth.tasks.base import DISTRACTOR_DEPTHS
from hanzi_synth.tasks.registry import CONSTRUCTORS


def _build_parser() -> argparse.ArgumentParser:
    """输入：无。输出：argparse 解析器。预期：注册全部子命令（list / dry-run / run / coverage /
    reject-stats / dedup / contamination / difficulty），未知子命令由 argparse 直接报错。"""
    parser = argparse.ArgumentParser(prog="hanzi_synth", description="汉字数据合成管线")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出已注册的题型")

    dry = sub.add_parser("dry-run", help="只跑 L0/L2 出题，不联网、不调 LLM")
    dry.add_argument("--limit", type=int, default=20)
    dry.add_argument("--tasks", default=None, help="逗号分隔的题型白名单")
    dry.add_argument("--out", default=str(DEFAULT_OUT.with_name("dry_run.jsonl")))
    dry.add_argument("--data", default=None, help="chars.jsonl 路径，默认 crawler/data/chars.jsonl")
    dry.add_argument("--overwrite", action="store_true", help="清空重跑（旧产出先备份成 .bak-<时间戳>）")
    dry.add_argument("--min-coverage", type=float, default=0.0, help="覆盖率下限，未达标返回退出码 1")
    _add_exclude_flag(dry)
    _add_difficulty_flags(dry)

    run = sub.add_parser("run", help="端到端合成（会调用 LLM）")
    run.add_argument("--limit", type=int, default=40)
    run.add_argument("--tasks", default=None, help="逗号分隔的题型白名单")
    run.add_argument("--out", default=str(DEFAULT_OUT))
    run.add_argument("--data", default=None, help="chars.jsonl 路径，默认 crawler/data/chars.jsonl")
    run.add_argument("--seed", type=int, default=20261006)
    run.add_argument("--temperature", type=float, default=0.6)
    run.add_argument("--overwrite", action="store_true", help="清空重跑（旧产出先备份成 .bak-<时间戳>）")
    run.add_argument("--min-coverage", type=float, default=0.0, help="覆盖率下限，未达标返回退出码 1")
    run.add_argument(
        "--rejects", default=None, help="L7 回炉：对 rejects 文件里的失败样本单条重试，成功追加进主产出"
    )
    run.add_argument(
        "--judge", action="store_true",
        help="L4.5：程序防线全过后再过一道 LLM 裁判（默认关；README §8.2 要求先做金标标定）",
    )
    run.add_argument(
        "--roundtrip", action=argparse.BooleanOptionalAction, default=True,
        help="L4.6 往返检验：把题干与解析交回模型答题，答案对不上载荷里定死的答案就否决重试"
             "（默认开；--no-roundtrip 关掉）",
    )
    _add_exclude_flag(run)
    _add_difficulty_flags(run)

    cov = sub.add_parser("coverage", help="只读：看现有产出覆盖了池子里的多少字")
    cov.add_argument("--out", default=str(DEFAULT_OUT))
    cov.add_argument("--data", default=None, help="chars.jsonl 路径，默认 crawler/data/chars.jsonl")
    cov.add_argument("--pool", type=int, default=0, help="只统计前 N 个字；0 = 全池")

    rej = sub.add_parser("reject-stats", help="只读：按防线名 / 题型汇总 rejects 与 replayed")
    rej.add_argument("--rejects", default=str(DEFAULT_OUT.with_suffix(".rejects.jsonl")))

    att = sub.add_parser("attempt-stats", help="只读：逐轮 L4 判定汇总（首轮死在哪、救回多少）")
    att.add_argument("--attempts", default=str(DEFAULT_OUT.with_suffix(".attempts.jsonl")))

    dedup = sub.add_parser("dedup", help="L6 只读：近似重复报告（不改原产出）")
    dedup.add_argument("--out", default=str(DEFAULT_OUT))
    dedup.add_argument("--key", default="stem", help="取哪一列做指纹，默认 stem")
    dedup.add_argument("--ngram", type=int, default=None, help=f"字符级 n-gram 长度，默认 {DEFAULT_NGRAM}")
    dedup.add_argument("--threshold", type=float, default=None, help=f"Jaccard 重复阈值，默认 {DEFAULT_THRESHOLD}")
    dedup.add_argument("--write-kept", action="store_true", help="另存去重副本 <out>.dedup.jsonl（原文件不动）")

    cont = sub.add_parser("contamination", help="L6 只读：查题干有没有连续照抄外部语料")
    cont.add_argument("--out", default=str(DEFAULT_OUT))
    cont.add_argument("--corpus", default=None, help="外部语料（.jsonl 或纯文本）；不给则如实报「没查」")
    cont.add_argument("--ngram", type=int, default=None, help="污染窗口长度，默认 13")

    audit = sub.add_parser(
        "audit",
        help="L6 只读：把一批已发布产出重过 L4（含往返判定必查），只留 A 级，组装最终数据集",
    )
    audit.add_argument(
        "--in", dest="inputs", action="append", default=None,
        help="可重复：产出 jsonl / 目录（目录展开一层 *.jsonl，自动剔掉旁路日志）",
    )
    audit.add_argument(
        "--out", default=str(DEFAULT_OUT.with_name("dataset.jsonl")),
        help="最终数据集路径；剔除记录写 <out>.dropped.jsonl，报告写 <out>.audit.json",
    )
    audit.add_argument(
        "--order", choices=("newest", "path"), default="newest",
        help="`--unique-by task-zi` 时留哪一版：newest = 最新文件优先（默认），path = 路径序",
    )
    audit.add_argument(
        "--unique-by", choices=UNIQUE_MODES, default="none",
        help="收唯一口径：none = 全收（默认，重复量只报出来）；task-zi = 每道题只留一条",
    )

    rt = sub.add_parser(
        "roundtrip",
        help="L4.6 只读：把产出交回模型，让它只凭题面与解析答题，答不出答案的记录就是读不通",
    )
    rt.add_argument(
        "--in", dest="inputs", action="append", default=None,
        help="可重复：产出 jsonl / 目录（目录展开一层 *.jsonl，自动剔掉旁路日志）",
    )
    rt.add_argument("--limit", type=int, default=0, help="最多检验多少条；0 = 全检")
    rt.add_argument(
        "--out", default=None,
        help="逐条结果写到这个 jsonl；默认 <第一个输入>.roundtrip.jsonl",
    )
    rt.add_argument("--workers", type=int, default=1, help="并发线程数（本机端点会被收到 LOCAL_WORKER_CAP）")
    rt.add_argument("--seed", type=int, default=20261007, help="抽样种子（--limit 小于总条数时才用）")

    diff = sub.add_parser("difficulty", help="L5：有作答数据才做 Rasch 校准，没有就如实报未标定")
    diff.add_argument("--responses", required=True, help="作答 jsonl，每条 {person, item, correct}")
    diff.add_argument("--out", default=None, help="校准结果输出路径，默认 <responses>.difficulty.json")
    return parser


def _add_exclude_flag(parser: argparse.ArgumentParser) -> None:
    """输入：子命令解析器。输出：无（就地加参数）。
    预期：`--exclude` 在 `dry-run` 与 `run` 上共用同一份说明——开新一轮时
          不排除已有产出就会把上一轮同一批字再采一遍（实测 597 条产出只覆盖
          87 组（题型, 字））。`dry-run` 上有这个开关，就能不花一次 LLM 请求先验计划。
    """
    parser.add_argument(
        "--exclude", action="append", default=None,
        help="可重复：已有产出文件（或目录），其 (task_id, ziid) 主键一律跳过。"
             "换 --out 开新一轮时必须给，否则会把上一轮同一批字再采一遍",
    )


def _add_difficulty_flags(parser: argparse.ArgumentParser) -> None:
    """输入：子命令解析器。输出：无（就地加参数）。
    预期：难度旋钮必须能从命令行调——否则又变成「配置里有、实际调不动」的死开关。
    """
    parser.add_argument("--hops", type=int, default=1, choices=(1, 2, 3), help="T9/T12 的链长/编辑步数")
    parser.add_argument("--constraints", type=int, default=1, choices=(1, 2, 3), help="T8 同时施加的条件个数")
    parser.add_argument(
        "--distractor-depth", default="near", choices=DISTRACTOR_DEPTHS,
        help="干扰项距离档（near 最难）；mixed = 按 near 4 : mid 3 : far 2 抽样，让难度成分布"
    )
    parser.add_argument("--pool", type=int, default=0, help="索引只建在前 N 个字上；0 = 全池")
    parser.add_argument("--max-scan", type=int, default=20000, help="出题阶段最多扫描多少个字")
    parser.add_argument(
        "--workers", type=int, default=1, help="L3 措辞的并发线程数（远端网关接近线性；本机端点会被收到 LOCAL_WORKER_CAP=2）"
    )
    parser.add_argument(
        "--retries", type=int, default=1, help="L4 不过时换语体重试的次数（网络层的重试在 provider 里，是另一回事）"
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="一次请求处理几道题；0 = 跟随 secrets.toml 的 batch。调大前必须过 A/B 质量对比（小模型批量会掉质量）"
    )


def _exclude_paths(raw: Sequence[str] | None) -> tuple[Path, ...]:
    """输入：CLI 给的 --exclude 路径（文件或目录，可重复）。输出：展开后的文件路径元组。
    预期：目录展开一层 `*.jsonl` 并剔掉旁路日志（与 `audit` 同一套判据）；
          不存在的路径直接报错——打错一个字就静默失效，会让人以为已排除。
    """
    if not raw:
        return ()
    found: list[Path] = []
    for item in raw:
        path = Path(item)
        if not path.exists():
            raise SystemExit(f"--exclude 路径不存在：{path}")
        if path.is_dir():
            found.extend(sorted(p for p in path.glob("*.jsonl") if is_published_source(p)))
        else:
            found.append(path)
    return tuple(dict.fromkeys(found))


def _task_ids(raw: str | None) -> tuple[str, ...] | None:
    """输入：逗号分隔的题型串。输出：元组或 None。预期：未知题型立刻报错，不静默忽略。"""
    if not raw:
        return None
    ids = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [tid for tid in ids if tid not in CONSTRUCTORS]
    if unknown:
        raise SystemExit(f"未知题型：{unknown}；可用：{sorted(CONSTRUCTORS)}")
    return ids


def main(argv: list[str] | None = None) -> int:
    """输入：命令行参数。输出：进程退出码。预期：配置/数据错误打印人话并返回 2。"""
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        for task_id in sorted(CONSTRUCTORS):
            print(task_id)
        return 0

    if args.command == "coverage":
        try:
            report = coverage_report(
                Path(args.out),
                Path(args.data) if args.data else default_data_path(),
                limit=(args.pool or None),
            )
        except DataError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "reject-stats":
        try:
            report = summarize_rejects(Path(args.rejects))
        except DataError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "attempt-stats":
        try:
            report = summarize_attempts(Path(args.attempts))
        except DataError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dedup":
        path = Path(args.out)
        if not path.exists():
            print(f"错误：产出文件不存在：{path}", file=sys.stderr)
            return 2
        report, kept = dedup_file(
            path,
            key=args.key,
            ngram=DEFAULT_NGRAM if args.ngram is None else args.ngram,
            threshold=DEFAULT_THRESHOLD if args.threshold is None else args.threshold,
        )
        report_path = write_report(path, report)
        if args.write_kept:
            report["kept_path"] = str(write_kept(path, kept))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告：{report_path}", file=sys.stderr)
        return 0

    if args.command == "contamination":
        path = Path(args.out)
        report = contamination_file(
            path,
            Path(args.corpus) if args.corpus else None,
            ngram=13 if args.ngram is None else args.ngram,
        )
        report_path = write_report(path, report, suffix=".contamination.json")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"报告：{report_path}", file=sys.stderr)
        return 0 if report.get("checked") else 1

    if args.command == "difficulty":
        responses_path = Path(args.responses)
        if not responses_path.exists():
            print(f"错误：作答文件不存在：{responses_path}", file=sys.stderr)
            return 2
        result = estimate_irt_from_responses(load_responses(responses_path))
        target = Path(args.out) if args.out else responses_path.with_name(responses_path.name + ".difficulty.json")
        write_calibration(target, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("calibrated") else 1

    if args.command == "roundtrip":
        inputs = [Path(item) for item in (args.inputs or [str(DEFAULT_OUT.parent)])]
        out_path = Path(args.out) if args.out else inputs[0].with_name(inputs[0].name + ".roundtrip.jsonl")
        stats = run_roundtrip(
            inputs=inputs, out_path=out_path, limit=args.limit, workers=args.workers,
            seed=args.seed, progress=lambda msg: print(msg, flush=True),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"逐条结果：{out_path}", file=sys.stderr)
        return 0 if stats["failed"] == 0 else 1

    if args.command == "audit":
        inputs = [Path(item) for item in (args.inputs or [str(DEFAULT_OUT.parent)])]
        out_path = Path(args.out)
        # 本次的输出文件必须从输入里剔掉：否则第二次跑会把上一次的数据集当源记录吃进去
        sources = collect_sources(inputs, order=args.order, exclude=[out_path])
        if not sources:
            print("错误：没有找到任何已发布产出文件", file=sys.stderr)
            return 2
        kept, dropped, report = audit_sources(sources, unique_by=args.unique_by)
        report["dataset_path"] = str(out_path)
        report["kept_lines"] = write_jsonl(out_path, kept)
        report["dropped_path"] = str(out_path.with_suffix(".dropped.jsonl"))
        write_jsonl(out_path.with_suffix(".dropped.jsonl"), dropped)
        report_path = write_audit_report(out_path.with_suffix(".audit.json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"数据集：{out_path}\n报告：{report_path}", file=sys.stderr)
        return 0

    try:
        config = PipelineConfig(
            limit=args.limit,
            task_ids=_task_ids(args.tasks),
            out_path=Path(args.out),
            data_path=(Path(args.data) if getattr(args, "data", None) else None),
            offline=args.command == "dry-run",
            resume=not getattr(args, "overwrite", False),
            temperature=getattr(args, "temperature", 0.6),
            seed=getattr(args, "seed", 20261006),
            pool_size=getattr(args, "pool", 0),
            max_scan=getattr(args, "max_scan", 20000),
            workers=getattr(args, "workers", 1),
            retries=getattr(args, "retries", 1),
            batch=getattr(args, "batch", 0),
            judge=getattr(args, "judge", False),
            roundtrip=getattr(args, "roundtrip", False),
            hops=getattr(args, "hops", 1),
            constraints=getattr(args, "constraints", 1),
            distractor_depth=getattr(args, "distractor_depth", "near"),
            exclude_paths=_exclude_paths(getattr(args, "exclude", None)),
        )
        if getattr(args, "rejects", None):
            replay_stats = replay_rejects(config, Path(args.rejects), progress=lambda msg: print(msg, flush=True))
            print(
                json.dumps(
                    {k: v for k, v in replay_stats.items() if k not in {"started", "finished"}},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        stats = run_pipeline(config, progress=lambda msg: print(msg, flush=True))
    except (ConfigError, DataError, PipelineBusy) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(json.dumps({k: v for k, v in stats.items() if k not in {"started", "finished"}},
                     ensure_ascii=False, indent=2))
    target = getattr(args, "min_coverage", 0.0)
    coverage = stats.get("coverage", {})
    if target and coverage.get("ratio", 0.0) < target:
        print(
            f"覆盖率未达标：{coverage.get('ratio', 0.0):.4%} < {target:.4%}，还差 {coverage.get('remaining', 0)} 个字；"
            f"本轮产出 {stats['ok']} 条，加大 --limit 或再跑几轮（续跑只追加，不会覆盖）",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
