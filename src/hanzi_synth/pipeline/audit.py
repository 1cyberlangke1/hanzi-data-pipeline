"""L6 发布门禁：把已发布产出重过 L4，只留 A 级，组装成最终数据集。

输入：一批「已发布产出」jsonl（历轮实验的原始日志，每轮一个文件）。
输出：只含 A 级记录的最终数据集、被剔除记录的旁路文件、汇总报告。

预期：
- **只读原文件、一个字都不改写**：筛选结果写到 `--out`；本工具只判不改，不合格的记录原样进
  旁路文件，等主流程重新合成（历史上这里还有一层 L3.5 清洗，那一层已整体删除）；
- **判据复用**：`spec_from_record` 重建 TaskSpec，`verify` 与主流程同一份实现——
  在这里另写一套判据，两套口径必然漂移；
- **只发布「可全量复验」的记录**：闸门字段（`facts` + `visible_subject` / `must_contain` /
  `hidden_subject` / `stem_asks_any` / `structure_words` / **`roundtrip`**）缺一个，`spec_from_record`
  就**静默降级成空元组**，「题干有没有写出那个字」「题干问的是不是答案那件事」这些判据全部失效。
  实测旧 schema 的记录（16ab 及更早、以及早期云端那批）里就躺着「**这个字是什么？**」这种
  读者根本无从作答的废题——降级复验照样判 A 收进数据集。**查得没那么严 ≠ 查过了**，
  所以这类记录一律不发布（它们由当前管线重新合成即可，产出记录现在必然带全闸门字段）；
- `--unique-by task-zi` 才按 (题型, 字) 收唯一（默认全收——「合成足够多」优先，重复量在报告里
  如实报出来，由使用者决定要不要收唯一）；
- **自然语言那一层也在这道门里**：机械闸门只做子串 / 集合比对，判不了「这段话读者读得懂吗」。
  所以 `roundtrip`（把题干交回模型答题的判定）是发布必查项：往返否决的记录不发，往返**缺席**
  （请求挂了 / 输出坏了）的记录同样不发——缺席不算否决，也不算通过（见 `llm/roundtrip.py:38`）。

为什么需要它：产出目录里躺的是**历轮实验的原始日志**，每条记录是「当时那一版代码」判过的。
判据只增不减，旧文件里必然留着今天判不过的记录（实测 46 条，其中 34 条选择题的题干根本没列
候选——读者没得选）。「最后的产出」不能是这些文件的简单并集，必须过一道可复现的门禁。

数据集记录的字段：
- `stem` / `explanation`：与源记录逐字相同（本工具不改写文本）；
- `verification`：**机械闸门按本记录自己的文本重算**——源记录里的那一份是旧代码对旧文本的
  结论，留着会让「判定」与「文本」对不上；但**LLM 那一道认记录里的结论**（`roundtrip` / `judge`
  两个 dict，以及答案仲裁的痕迹 `semantic_answer`）：它们重跑不了，而重跑机械闸门会把「主流程
  靠 LLM 仲裁撤掉的那条 `answer_in_explanation`」重新判成失败——同一条数据两处结论相反，
  实测 93 条被剔记录里 62 条就是这么来的（解析用「左右结构，左边是亻」这种散文说出答案，
  子串比对判不出，LLM 仲裁判得出）；
- `audit`：`{"source": 源文件名, "recheck": 复验强度}`，说明这条从哪个文件来、查得严不严。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..llm.verbalizer import Verbalized
from ..verify.checks import verify
from .dedup import load_jsonl
from .run import _round_passed, _round_reasons, spec_from_record

# 旁路文件后缀：这些不是产出，是日志 / 报告 / 备份。
# `--in` 给目录时按名字剔掉；给具体文件时不剔——使用者显式点名了就照做。
SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".attempts.jsonl",
    ".rejects.jsonl",
    ".replayed.jsonl",
    ".dropped.jsonl",
    ".dedup.jsonl",
)

# 收唯一的两种口径：`none` 全收（默认），`task-zi` 按 (题型, 字) 收一条。
UNIQUE_MODES: tuple[str, ...] = ("none", "task-zi")

# 复验强度所依赖的闸门字段。`spec_from_record` 缺字段时会**静默降级成空元组**，
# 于是「题干必须问对方向」「必须写出读音」「不得泄露隐藏主语」「不得点出结构词」这些闸门
# 在复验里根本不生效——旧 schema 的产出记录（16ab 及更早、以及云端那批）就没有这几个字段。
# 所以复验必须分「full / partial」两档报出来，不能拿降级的结论当查过了；
# 而**发布**只认 full——partial 记录一律不进数据集（见模块 docstring）。
# `facts` 也在必查清单里：少了它，`numeric_fidelity`、`no_invented_hanzi` 这些
# 「正文里的数字 / 扩展区汉字必须来自 facts」的判据同样退化成空转。
# `roundtrip` 是**自然语言那一层**的判定（把题干交回模型，看它答不答得出定死的答案）。
# 它必须在清单里：机械闸门只会做子串 / 集合比对，判不了「这段话读者读得懂吗」，
# 少了它，数据集对可读性就没有任何证据——而「没查」不能当成「查过了」。
GATE_FIELDS: tuple[str, ...] = (
    "facts",
    "visible_subject",
    "must_contain",
    "hidden_subject",
    "stem_asks_any",
    "structure_words",
    "roundtrip",
)

# 缺 `roundtrip` 与缺机械闸门字段是两回事，旁路文件里要分开说清楚。
LLM_GATE_NOTE = "；roundtrip 缺失 = 自然语言那一层没被判过（机械闸门只比子串，判不了可读性）"

# 离线模板产出的 provenance 标记（只由 `render.generate` 写，措辞路径不带这个键）。
# 为什么它可以不要求 `roundtrip`：往返检验查的是**模型写的那段自然语言**读不读得懂
# （实测 2B 会把出题说明整段抄成题干）。离线产出的自然语言**不是模型写的**——是程序按
# 人手写的句式池 + 字典事实拼出来的，检验对象不存在；它的自然语言那一层由句式池本身 +
# 全量重渲染确定性保证（`rerender_offline.py`：705,175 条 0 条不一致）。
# 豁免**只认这个标记**，不认「roundtrip 恰好缺失」——否则任何模型产出只要漏掉这一键就能
# 免检发布，那正是本门禁要拦的事。
OFFLINE_MODE = "offline-template"


def is_offline_template(row: Mapping[str, Any]) -> bool:
    """输入：一条产出记录。输出：它是不是离线模板产出（自然语言由程序拼出）。
    预期：只认 `provenance.mode == "offline-template"`；`provenance` 缺失 / 不是映射 /
    `mode` 是别的值一律 False——豁免口径必须比「键在不在」严，不能靠缺字段蒙混。
    """
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        return False
    return str(provenance.get("mode", "")) == OFFLINE_MODE


def _missing_gates(row: Mapping[str, Any]) -> list[str]:
    """输入：一条产出记录。输出：缺掉的闸门字段名（按 `GATE_FIELDS` 顺序）。

    预期：`roundtrip` 这一项**必须是判定本身**——记录里写成 `null`（跑的时候关了往返检验）
          与键根本不在，都算没查过。其余字段只判「在不在」：`visible_subject` 为空串
          是合法值（选择题没有单一主语），不能拿真假值去判。
          离线模板产出（`is_offline_template`）**不要求 `roundtrip`**：见 `OFFLINE_MODE`
          那段——它的自然语言不是模型写的。其余机械闸门字段照样一个不能缺。
    """
    required = GATE_FIELDS if not is_offline_template(row) else tuple(
        field for field in GATE_FIELDS if field != "roundtrip"
    )
    missing = [field for field in required if field not in row]
    if "roundtrip" in required and not isinstance(row.get("roundtrip"), dict):
        missing.append("roundtrip")
    return missing


def recheck_strength(row: Mapping[str, Any]) -> str:
    """输入：一条产出记录。输出：`"full"` 或 `"partial"`。
    预期：`partial` = 记录缺闸门字段，复验只跑到「从记录本身能重建」的那些防线。
          这不是「这条有问题」，而是「这条查得没那么严」——报告要如实说清楚，
          **调用方据它决定不发**：查得没那么严的记录不许进数据集。
          离线模板产出只要机械闸门字段齐全就是 `full`：它的 `roundtrip` 本来就没有对象
          （`OFFLINE_MODE`），把「没有对象」当成「查得不严」会让整批离线数据一条都发不出去。
    """
    return "full" if not _missing_gates(row) else "partial"


def is_published_source(path: Path) -> bool:
    """输入：一个文件路径。输出：它看起来是不是「已发布产出」。
    预期：只按文件名判、不读内容，所以判据必须便宜且不会误判——
          旁路日志 / 报告 / 备份一律返回 False。
    """
    name = path.name
    if not name.endswith(".jsonl"):
        return False
    return not any(name.endswith(suffix) for suffix in SIDECAR_SUFFIXES)


def collect_sources(
    inputs: Iterable[Path], *, order: str = "newest", exclude: Iterable[Path] = ()
) -> list[Path]:
    """输入：CLI 给的路径（文件 / 目录）+ 排序口径 + 要排除的路径（本次的输出文件）。
    输出：去重后的产出文件列表。

    预期：目录展开成一层 `*.jsonl` 并按 `is_published_source` 过滤；
          `order="newest"` 按修改时间从新到旧（`--unique-by task-zi` 时新轮次优先），
          `order="path"` 按路径字典序（跨机器可复现）；
          同一文件被重复点名（既给目录又给文件）只留一份；
          **`exclude` 里的路径一定剔掉**——否则第二次跑会把上一次的数据集当输入吃进去。
    """
    banned = {str(path.resolve()) for path in exclude}
    found: list[Path] = []
    for raw in inputs:
        if raw.is_dir():
            found.extend(sorted(p for p in raw.glob("*.jsonl") if is_published_source(p)))
        elif raw.exists():
            found.append(raw)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path.resolve())
        if key in seen or key in banned:
            continue
        seen.add(key)
        unique.append(path)
    if order == "newest":
        unique.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    else:
        unique.sort(key=lambda p: str(p))
    return unique


def is_published_record(row: Mapping[str, Any]) -> bool:
    """输入：一行 jsonl 解析出的字典。输出：它是不是一条**已发布产出**。
    预期：按内容判，不按文件名判——产出目录里同时躺着 `cli dry-run` 的离线载荷
          （`_offline_record` 的产物：只有 instruction/facts/answer，没有 stem），
          那不是产出，拿它去跑 L4 只会得到一堆 `schema` 噪声。产出记录必然同时有
          `stem` 与 `explanation` 两个槽位（发布闸门 `schema` 就是这么判的）。
    """
    return "stem" in row and "explanation" in row


# 记录里「子串比对没过、但 LLM 仲裁判过」的痕迹前缀。主流程 `run._verified_with_semantic_answer`
# 撤掉那条失败时写的就是它（见 `verify/checks.Verification.semantic_answer`）。
_SEMANTIC_ARBITRATION_PREFIX = "LLM 仲裁"


def recorded_answer_arbitration(row: Mapping[str, Any]) -> str:
    """输入：一条产出记录。输出：记录里那次「子串比对没过、但 LLM 仲裁判过」的痕迹（没有则空串）。

    预期：只在记录**自己写着** `answer_in_explanation` 为真、且 `semantic_answer` 是仲裁结论时
          才算数——那正是主流程撤掉这条失败、于是把记录发布出去的那一刻。
          「解析说没说答案」是 **LLM 那一道**，与 `roundtrip` / `judge` 同类：本工具重跑不了，
          只认记录里的结论。不认它就会出现「主流程发布、门禁剔掉」这种自相矛盾。
    """
    recorded = row.get("verification")
    if not isinstance(recorded, Mapping) or not recorded.get("answer_in_explanation"):
        return ""
    note = str(recorded.get("semantic_answer") or "")
    return note if note.startswith(_SEMANTIC_ARBITRATION_PREFIX) else ""


def audit_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """输入：一条已发布产出记录。输出：判定结果。

    预期：判定口径与主流程**完全同一份实现**——`spec_from_record` 重建 TaskSpec、`verify` 跑同一
          份防线、`_round_passed` 当发布判据、`_round_reasons` 出失败原因。在这里另写一套必然漂移：
          实测旧版就漏了「模型侧检验**缺席**」这一档——缺席既不是否决也不是通过（见
          `llm/roundtrip.py:38`），没被检验过的记录同样不能进数据集。本工具**不改写文本**，
          所以只有一个 `keep` 结论。
          答案那一格还有一层：`answer_in_explanation` 是子串比对，主流程在它没过时还会问一次
          LLM 仲裁并撤掉这条失败；本工具重跑不了那一次，按记录里的痕迹认它
          （`recorded_answer_arbitration`），否则「主流程发布、门禁剔掉」会同时发生在一条数据上。
    """
    spec = spec_from_record(row)
    original = Verbalized(
        ok=True,
        stem=str(row.get("stem", "") or ""),
        explanation=str(row.get("explanation", "") or ""),
        provider=str(row.get("provider", "") or ""),
        model=str(row.get("model", "") or ""),
    )
    verification = verify(spec, original)
    arbitration = recorded_answer_arbitration(row)
    if arbitration and not verification.answer_in_explanation:
        # 机械复验只会做子串比对，而主流程在这一格上还问过 LLM 仲裁并撤掉了这条失败。
        # 按记录撤掉它：`failures` / `notes` 是同一批失败的两份说法，撤一条要两边一起撤。
        kept = [
            (failure, note)
            for failure, note in zip(verification.failures, verification.notes)
            if not failure.startswith("answer_in_explanation")
        ]
        verification = replace(
            verification,
            answer_in_explanation=True,
            semantic_answer=arbitration,
            failures=tuple(failure for failure, _ in kept),
            notes=tuple(note for _, note in kept),
        )
    judgement = row.get("judge")
    roundtrip = row.get("roundtrip")
    return {
        "tier": verification.tier,
        "failures": _round_reasons(verification.tier, verification.failures, judgement, roundtrip),
        "keep": _round_passed(verification.tier, judgement, roundtrip),
        "verification": verification,
    }


def _rule_name(failure: str) -> str:
    """输入：一条失败原因（`"防线名: 说明"`）。输出：防线名；没有冒号就整条返回。
    预期：汇总报告按防线名分组，所以只取冒号前那一段。
    """
    head = failure.split(":", 1)[0].strip()
    return head or failure


def _dataset_row(row: Mapping[str, Any], outcome: Mapping[str, Any], source: str) -> dict:
    """输入：源记录 + 判定结果 + 源文件名。输出：数据集记录。
    预期：只改 `verification`（按本记录自己的文本重算），其余字段原样带出；再加 `audit`
          溯源块（源文件 / 复验强度）。绝不在这里补造字段——数据集必须能逐字段追回源记录。
    """
    out = dict(row)
    out["verification"] = outcome["verification"].as_dict()
    out["audit"] = {"source": source, "recheck": recheck_strength(row)}
    return out


# 回炉载荷要带的字段：构造器 / 闸门侧的全部输入。`spec_from_record` 只认这些键，
# `stem` / `explanation` / `verification` 一律不带——回炉要**重写**文本，带上旧文本只会被照抄。
REPLAY_FIELDS: tuple[str, ...] = (
    "instruction",
    "facts",
    "answer",
    "accepted",
    "options",
    "visible_subject",
    "must_contain",
    "hidden_subject",
    "stem_asks_any",
    "structure_words",
    "payload_facts",
    "proof",
    "distractors",
    "provenance",
)


def _replay_payload(row: Mapping[str, Any]) -> dict:
    """输入：产出记录。输出：可直接喂 `cli replay` 的回炉载荷。
    预期：形状与 `run.py` 写 rejects 时的 `spec` 载荷一致（`_offline_record` 的产物），
          键按 `REPLAY_FIELDS` 挑；缺字段就不带（`spec_from_record` 有安全默认值）。
          「不进数据集」不等于「可以丢」——没有这份载荷，旁路文件只剩定位信息，
          回炉时要人工回源文件里找，等于事实上的丢弃。
    """
    payload: dict[str, Any] = {"task_id": str(row.get("task_id", "")), "ziid": str(row.get("ziid", ""))}
    for field in REPLAY_FIELDS:
        if field in row:
            payload[field] = row[field]
    return payload


def _dropped_row(row: Mapping[str, Any], outcome: Mapping[str, Any], source: str) -> dict:
    """输入：源记录 + 判定结果 + 源文件名。输出：被剔除记录的旁路行。
    预期：定位信息 + 原因 + **回炉载荷**（`spec`）——不带整条正文（stem / explanation 不进旁路），
          但必须能重建 TaskSpec，否则回炉无从下手（见 `_replay_payload`）。
    """
    return {
        "task_id": row.get("task_id", ""),
        "ziid": row.get("ziid", ""),
        "source": source,
        "tier": outcome["tier"],
        "failures": list(outcome["failures"]),
        "spec": _replay_payload(row),
    }


def _unverifiable_row(row: Mapping[str, Any], source: str) -> dict:
    """输入：缺闸门字段的产出记录 + 源文件名。输出：旁路行。
    预期：与 `_dropped_row` 同一形状（含回炉载荷），但 `tier` 记 `unverifiable`、`failures`
          点明缺了什么——这不是「内容不合格」，而是「这条查不严，所以不发布」，
          两种原因不能在旁路文件里混成一种。
    """
    missing = _missing_gates(row)
    return {
        "task_id": row.get("task_id", ""),
        "ziid": row.get("ziid", ""),
        "source": source,
        "tier": "unverifiable",
        "failures": [
            f"unverifiable: 记录缺闸门字段 {'/'.join(missing)}——复验会静默降级，"
            "「题干有没有写出那个字」这类判据全部失效，所以不发布（当前管线重新合成即可）"
            + (LLM_GATE_NOTE if "roundtrip" in missing else "")
        ],
        "spec": _replay_payload(row),
    }


def audit_sources(
    sources: Sequence[Path], *, unique_by: str = "none"
) -> tuple[list[dict], list[dict], dict]:
    """输入：产出文件列表 + 收唯一口径。输出：(数据集, 被剔除, 报告)。

    预期：**每一条记录都过门禁**——重复与否不影响「它本身合不合格」的判定，所以先判后收；
          `unique_by="task-zi"` 时按 (题型, 字) 收一条，先到先得（`collect_sources` 已把
          新轮次排在前面）；`unique_by="none"` 时全收，重复量只如实计数。
          文件不存在 / 有坏行都如实计数，不抛异常——旁路工具不该因为一个文件坏掉就整批失败。
    """
    if unique_by not in UNIQUE_MODES:
        raise ValueError(f"unique_by 必须是 {UNIQUE_MODES} 之一，收到 {unique_by!r}")
    kept: list[dict] = []
    dropped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    stems: set[str] = set()
    per_source: list[dict] = []
    failure_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    fact_duplicates = 0
    stem_duplicates = 0
    skipped_lines = 0
    non_records = 0
    non_record_sources: Counter[str] = Counter()
    partial_recheck = 0
    partial_dropped = 0
    partial_sources: Counter[str] = Counter()
    offline_records = 0
    already_assembled = 0
    assembled_sources: Counter[str] = Counter()

    for path in sources:
        rows, skipped = load_jsonl(path)
        skipped_lines += skipped
        source_kept = 0
        source_dropped = 0
        source_non_records = 0
        for row in rows:
            # 带 `audit` 块的是**本工具上一次组装出来的数据集行**，不是产出源。
            # 不拦掉它，第二次跑（尤其换过 --out 路径时）就会把上一轮的数据集吃回来，
            # 数据集一轮一轮地自我增殖。
            if "audit" in row:
                already_assembled += 1
                assembled_sources[path.name] += 1
                continue
            if not is_published_record(row):
                non_records += 1
                non_record_sources[path.name] += 1
                source_non_records += 1
                continue
            if is_offline_template(row):
                # 只计数：这类记录不要求 roundtrip（见 `OFFLINE_MODE`），报告里要如实报出来，
                # 免得「数据集里有多少条没被读者读过」变成一个看不见的数。
                offline_records += 1
            partial = recheck_strength(row) == "partial"
            if partial:
                # 缺闸门字段 = 复验降级 = 没查严。**不发布**，也不浪费一次降级复验：
                # 降级结论对「这条合不合格」没有证明力（见模块 docstring 的「这个字是什么？」）。
                partial_recheck += 1
                partial_sources[path.name] += 1
                dropped.append(_unverifiable_row(row, path.name))
                source_dropped += 1
                partial_dropped += 1
                task_counts[str(row.get("task_id", ""))] += 1
                continue
            outcome = audit_record(row)
            key = (str(row.get("task_id", "")), str(row.get("ziid", "")))
            if key in seen:
                fact_duplicates += 1
            if not outcome["keep"]:
                dropped.append(_dropped_row(row, outcome, path.name))
                source_dropped += 1
                for failure in outcome["failures"]:
                    failure_counts[_rule_name(failure)] += 1
                task_counts[str(row.get("task_id", ""))] += 1
                continue
            stem = str(row.get("stem", ""))
            if stem in stems:
                stem_duplicates += 1
            if key in seen and unique_by == "task-zi":
                continue
            seen.add(key)
            stems.add(stem)
            kept.append(_dataset_row(row, outcome, path.name))
            source_kept += 1
        per_source.append(
            {
                "path": str(path),
                "records": len(rows),
                "kept": source_kept,
                "dropped": source_dropped,
                "non_records": source_non_records,
                "skipped_lines": skipped,
            }
        )

    report = {
        "sources": per_source,
        "source_files": len(sources),
        "total_lines": sum(item["records"] for item in per_source),
        "non_records": non_records,
        "non_records_by_source": dict(non_record_sources.most_common()),
        "already_assembled": already_assembled,
        "already_assembled_by_source": dict(assembled_sources.most_common()),
        "total_records": sum(item["records"] - item["non_records"] for item in per_source),
        "kept": len(kept),
        "dropped": len(dropped),
        "unique_by": unique_by,
        "unique_facts": len(seen),
        "fact_duplicates": fact_duplicates,
        "stem_duplicates": stem_duplicates,
        "recheck_partial": partial_recheck,
        "recheck_partial_dropped": partial_dropped,
        "recheck_partial_by_source": dict(partial_sources.most_common()),
        "offline_records": offline_records,
        "offline_roundtrip_exempt": offline_records,
        "skipped_lines": skipped_lines,
        "dropped_by_defense": dict(failure_counts.most_common()),
        "dropped_by_task": dict(task_counts.most_common()),
    }
    return kept, dropped, report


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """输入：目标路径 + 记录序列。输出：写入条数。
    预期：`ensure_ascii=False` 保住汉字可读；每条一行、UTF-8 无 BOM、LF 行尾；目录不存在就建。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_audit_report(path: Path, report: Mapping[str, Any]) -> Path:
    """输入：报告路径 + 报告字典。输出：实际写出的路径。
    预期：与 dedup 的报告同一套写法，但**函数名必须区分**——`cli.py` 里已经导入了
          dedup 的 `write_report`（带 `suffix` 参数），同名会把它顶掉。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return path