"""L4.6 往返检验的编排：读已发布产出 → 逐条交回模型答题 → 报出答不出答案的那些。

为什么把它做成**只读**命令：往返检验的价值是「对**已经发出去的**数据做一次独立体检」。
它不改文本、不改产出文件，只写一份逐条结果（`<in>.roundtrip.jsonl`）——这样它既能当发布前的
校验（退出码非 0 = 有记录答不出答案），也能当事后审计（换模型、换温度重跑一遍对比）。
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import ProviderConfig, load_providers
from ..llm.provider import ChatClient
from ..llm.roundtrip import RoundTrip, roundtrip_answer, subject_identified_by_gates
from ..llm.verbalizer import Verbalized
from .audit import collect_sources
from .dedup import load_jsonl
from .run import PipelineConfig, spec_from_record, with_local_worker_cap


def _verbalized_of(record: dict) -> Verbalized:
    """输入：一条产出记录。输出：只带读者可见两段文本的 Verbalized。
    预期：往返检验只该看到题干与解析，其余字段一个都不带——带了答案这段检验就自证了。
    """
    return Verbalized(
        ok=True,
        stem=str(record.get("stem", "")),
        explanation=str(record.get("explanation", "")),
        provider=str(record.get("provider", "")),
        model=str(record.get("model", "")),
    )


def run_roundtrip(
    *,
    inputs: Sequence[Path],
    out_path: Path,
    limit: int = 0,
    workers: int = 1,
    seed: int = 20261007,
    temperature: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """输入：输入路径 + 输出路径 + 上限 + 并发 + 种子。输出：统计字典。

    预期：逐条把（题干, 选项）交给模型答题，答案对不上提示内容里的定死答案就记一条失败。
          比对**先规则后仲裁**：子串比对过了就过，对不上再让同一个模型判「是不是同一个答案」
          （读者答出的是自然语言，只做子串比对会误杀「读作 xié」这种合法写法）。
          「读者自报缺条件」还要过一道旁证：写入文件记录里的 `subject_in_stem` / `options_listed`
          两条校验确认题面写全了识别条件时，按读者误报处理（`subject_identified_by_gates`）。
          提示内容不全（老记录缺校验字段）的条目**不猜**，单独计数并跳过——拿缺字段的记录复验
          只会静默降级，那比不查更坏。`--limit` 小于总条数时按种子抽样，可复现。
    """
    say = progress or (lambda _msg: None)
    sources = collect_sources(inputs, order="newest", exclude=[out_path])
    records: list[tuple[str, dict]] = []
    unverifiable = 0
    for path in sources:
        rows, _skipped = load_jsonl(path)
        for record in rows:
            try:
                spec_from_record(record)
            except Exception:
                unverifiable += 1
                continue
            records.append((path.name, record))
    total = len(records)
    if limit and limit < total:
        records = random.Random(seed).sample(records, limit)
    say(f"往返检验：源文件 {len(sources)} 个、可检验 {total} 条，本次检验 {len(records)} 条")

    providers: Sequence[ProviderConfig] = tuple(load_providers(role="verbalizer"))
    config, note = with_local_worker_cap(PipelineConfig(workers=workers), providers)
    if note:
        say(note)
    client = ChatClient(list(providers))
    workers = max(1, config.workers)

    def check(item: tuple[str, dict]) -> dict[str, Any]:
        """输入：(来源文件名, 记录)。输出：一行逐条结果。预期：请求挂了也如实记账，不吞。"""
        source, record = item
        spec = spec_from_record(record)
        result: RoundTrip = roundtrip_answer(
            client, spec, _verbalized_of(record), temperature=temperature,
            answer_judge=client,
            subject_identified=subject_identified_by_gates(record.get("verification")),
        )
        return {
            "task_id": spec.task_id,
            "ziid": str(record.get("ziid", "")),
            "source": source,
            "expected": spec.answer,
            "answer": result.answer,
            "ok": result.ok,
            "error": result.error,
            "provider": result.provider,
            "model": result.model,
        }

    if workers <= 1:
        rows = [check(item) for item in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(check, records))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed = [row for row in rows if not row["ok"] and not row["error"]]
    absent = [row for row in rows if row["error"]]
    by_task: dict[str, int] = {}
    for row in failed:
        by_task[row["task_id"]] = by_task.get(row["task_id"], 0) + 1
    return {
        "sources": [str(path) for path in sources],
        "unverifiable": unverifiable,
        "checked": len(rows),
        "ok": len(rows) - len(failed) - len(absent),
        "failed": len(failed),
        "absent": len(absent),
        "failed_by_task": dict(sorted(by_task.items())),
        "out_path": str(out_path),
        "results_path": str(out_path),
    }
