"""离线渲染出题：不调任何模型，直接按句式池把 TaskSpec 渲染成数据。

输入：字数上限、题型白名单、输出路径、种子。
输出：写入文件 JSONL + 统计 dict。
预期：句式是程序自己写死的模板，措辞正确性由模板本身保证；但「模板是程序写的」不等于「一定对」，
      所以每条仍然逐条过同一套机械校验（`verify`），过不了的**不写入文件**并计入 `unverified`——
      写入文件的每一条都带 `verification` 自证。
      事实与答案只有 TaskSpec 一个来源，所以答案不可能被措辞写错。
      唯一需要换句式的情况：非选择题的答案恰好出现在题干里（泄题），见 `_render_one`。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from ..facts.index import CharacterIndex
from ..facts.loader import load_characters
from ..llm.verbalizer import Verbalized
from ..pipeline.difficulty import Difficulty, difficulty_note
from ..tasks.base import MIXED_DEPTH, MIXED_LEVEL
from ..tasks.registry import CONSTRUCTORS, build_all
from ..verify.checks import verify
from . import offline as O


def _start_variant(seed: int, task_id: str, ziid: str, space: int) -> int:
    """输入：种子 + 题目标识 + 句式空间大小。输出：起始 variant 序号。
    预期：同一条题在同一个种子下永远从同一个句式起步（确定性、可复现）；
          换种子就换一批句式，同一批数据里不会全是同一个说法。
    """
    if space <= 0:
        return 0
    digest = hashlib.sha256(f"{seed}|{task_id}|{ziid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % space


def spec_rng(seed: int, task_id: str, ziid: str) -> random.Random:
    """输入：种子 + 题型 + 字标识。输出：这一 (题型, 字) **专属**的随机源。
    预期：随机源只由自己的键决定，与「字库里有哪几型」「前面的字抽了几次」全都无关。
          原先用的是一个跨 (字, 题型) 共享的 `random.Random`，于是删掉一型就把后面每一型的
          随机流整体挪位——实测删掉 T1 之后 T10 的候选 0/343 相同、T11 0/240、T16 0/354
          （`synthesis/out/probe_coupling_before.txt`），写入文件数据的内容取决于当时字库里有哪几型，
          换一次题型清单就复现不出同一份数据集。`random.Random(str)` 内部走 sha512，
          跨进程、跨平台稳定（不受 PYTHONHASHSEED 影响）。
    """
    return random.Random(f"{seed}|{task_id}|{ziid}")


def _render_one(spec, seed: int) -> tuple[str, str, int] | None:
    """输入：TaskSpec + 种子。输出：(stem, explanation, variant)，渲染不出时 None。
    预期：模板是程序自己写的，不过任何校验；从 `_start_variant` 起步环形找一个能渲染的
          句式。`render` 只对一种变体返回 None——非选择题的答案恰好落在题干里（泄题），
          这时顺着环形换下一个句式，不是「校验失败」。
          走位按**题干空间**封顶：泄题与否只由题干决定，一段解析里的题干全试过了还没得挑，
          换下一段解析也是同一批题干（白走）。
    """
    space = O.variant_space(spec)
    if space <= 0:
        return None
    start = _start_variant(seed, spec.task_id, str(spec.provenance.get("ziid", "")), space)
    for step in range(O.stem_space(spec)):
        variant = (start + step) % space
        rendered = O.render(spec, variant)
        if rendered is None:
            continue
        stem, explanation = rendered
        return stem, explanation, variant
    return None


def render_difficulty() -> Difficulty:
    """输入：无。输出：离线渲染的**默认**难度（三个旋钮都走分布档 `mixed`）。
    预期：让离线数据集里真的有多跳 T9/T12、多约束 T8，以及 near/mid/far 成分布的干扰项——
          否则整份数据的 difficulty 注记全是 hops=1 / near，等于难度旋钮白装了。
          `render-stats` 也用同一个函数，保证「量到的句式空间」就是生产会渲染的那一批。
    """
    return Difficulty(hops=MIXED_LEVEL, constraints=MIXED_LEVEL, distractor_depth=MIXED_DEPTH)


def generate(
    *,
    limit: int,
    out_path: Path,
    dropped_path: Path | None = None,
    data_path: Path | None = None,
    task_ids: Sequence[str] = (),
    seed: int = 20261006,
    min_per_type: int = 0,
    difficulty: Difficulty | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """输入：上限 + 输出路径 + 落选文件路径 + 数据路径 + 题型白名单 + 种子 + 每型下限 + 难度。输出：统计 dict。
    预期：逐字逐题型构造 TaskSpec → 渲染 → 过校验 → 写入文件；同一 (题型, 字) 只出一条。
          达不到 `min_per_type` 时如实报告，不假装完成。
          **过不了校验 / 渲染不出来的那几条一条都不丢**：写进 `<out>.dropped.jsonl`，带 `drop_reason`
          与校验给出的 `failures`（默认落在 `out_path` 旁边，可用 `dropped_path` 改）。
          「不落进主数据集」与「丢了」是两件事——主数据集要干净，落选的样本要能被人翻出来查。
          出题一律走 `registry.build_all`（唯一入口）——难度阶梯与 `resolved_*` 落定只写在一处。
    """
    started = time.perf_counter()
    characters = load_characters(data_path)
    index = CharacterIndex(characters)
    difficulty = render_difficulty() if difficulty is None else difficulty
    dropped_path = dropped_path or out_path.with_suffix(".dropped.jsonl")
    # 释义类题（T13 靠释义认字 / T17 问字义）走 LLM 路径，模板池里没有它们（见
    # `offline.LLM_ONLY_TASK_IDS`）。这里**整型排除**，不让它们落进 `unrenderable`——
    # 那是「模板渲染失败」，这两型是「按纪律不该走这条路」，两件事不能混成同一个计数。
    requested = [tid for tid in CONSTRUCTORS if (not task_ids or tid in task_ids)]
    wanted = [tid for tid in requested if O.has_templates(tid)]
    skipped_llm_only = [tid for tid in requested if not O.has_templates(tid)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "mode": "offline-render",
        "planned": limit,
        "written": 0,
        "unrenderable": 0,
        "unverified": 0,
        "scanned": 0,
        "by_task": {},
        "skipped_llm_only": skipped_llm_only,
        "out_path": str(out_path),
        "dropped_path": str(dropped_path),
        "seed": seed,
    }
    by_task: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as handle, dropped_path.open(
        "w", encoding="utf-8"
    ) as dropped:
        for character in characters:
            if stats["written"] >= limit:
                break
            stats["scanned"] += 1
            for task_id in wanted:
                if stats["written"] >= limit:
                    break
                if min_per_type and by_task.get(task_id, 0) >= min_per_type:
                    continue
                ziid = getattr(character, "ziid", "")
                try:
                    built = build_all(
                        character,
                        rng=spec_rng(seed, task_id, ziid),
                        difficulty=difficulty,
                        index=index,
                        only=(task_id,),
                    )
                except Exception:
                    built = []
                if not built:
                    continue
                spec = built[0]
                spec = _with_ziid(spec, character)
                rendered = _render_one(spec, seed)
                if rendered is None:
                    stats["unrenderable"] += 1
                    dropped.write(
                        json.dumps(
                            _record(
                                spec, character, seed, difficulty, None, "", "", None,
                                drop_reason="unrenderable",
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                stem, explanation, variant = rendered
                # 「模板是程序写的」不等于「一定对」：逐条过同一套机械校验。
                # 过不了不进主数据集（如实计入 unverified，并整条写进落选文件）；写入文件的每一条都带
                # verification 自证。
                verification = verify(spec, Verbalized(ok=True, stem=stem, explanation=explanation))
                if verification.tier != "A":
                    stats["unverified"] += 1
                    dropped.write(
                        json.dumps(
                            _record(
                                spec, character, seed, difficulty, variant, stem, explanation,
                                verification, drop_reason="unverified",
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                handle.write(
                    json.dumps(
                        _record(
                            spec, character, seed, difficulty, variant, stem, explanation,
                            verification,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stats["written"] += 1
                by_task[task_id] = by_task.get(task_id, 0) + 1
                progress(f"[{stats['written']}/{limit}] {task_id} {getattr(character, 'ziid', '')} OK")

    stats["by_task"] = dict(sorted(by_task.items()))
    stats["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return stats


def _record(
    spec,
    character,
    seed: int,
    difficulty: Difficulty,
    variant: int | None,
    stem: str,
    explanation: str,
    verification,
    *,
    drop_reason: str = "",
) -> dict[str, Any]:
    """输入：一条 spec + 字记录 + 种子 + 难度 + 句式序号 + 产出文本 + 校验结果 + 落选原因。
    输出：一条待写入文件的记录（主数据集与落选文件**同一个形状**）。
    预期：两处共用这一个构造器，落选样本才带得齐「它本来要写成什么样、为什么没过」；
          `drop_reason` 空串表示这是进主数据集的正式产出，非空表示它落选了。
    """
    record: dict[str, Any] = {
        "task_id": spec.task_id,
        "ziid": getattr(character, "ziid", ""),
        "instruction": spec.instruction,
        "facts": dict(spec.facts),
        "answer": spec.answer,
        "accepted": list(spec.accepted),
        "options": list(spec.options),
        "visible_subject": spec.visible_subject,
        "must_contain": list(spec.must_contain),
        "hidden_subject": spec.hidden_subject,
        "stem_asks_any": list(spec.stem_asks_any),
        "structure_words": list(spec.structure_words),
        "difficulty": difficulty_note(
            spec.task_id,
            difficulty,
            resolved_depth=spec.resolved_depth,
            resolved_hops=spec.resolved_hops,
            resolved_constraints=spec.resolved_constraints,
        ),
        "provenance": {
            "mode": "offline-template",
            "variant": variant,
            "seed": seed,
            "ziid": getattr(character, "ziid", ""),
            "constructor_id": spec.task_id,
            "constructor_version": str((spec.provenance or {}).get("constructor_version", "")),
        },
        "stem": stem,
        "explanation": explanation,
        "provider": "offline-template",
        "model": "template-pool-v1",
        "verification": verification.as_dict() if verification is not None else None,
    }
    if drop_reason:
        record["drop_reason"] = drop_reason
    return record


def _with_ziid(spec, character):
    """输入：TaskSpec + 字记录。输出：provenance 里补上 `ziid` 的 TaskSpec。
    预期：`_start_variant` 要拿 ziid 算起始句式，而构造器不保证把它写进 provenance。
    """
    from dataclasses import replace

    provenance = dict(spec.provenance or {})
    provenance.setdefault("ziid", getattr(character, "ziid", ""))
    return replace(spec, provenance=provenance)
