"""离线模板池的句式统计：把「每个题型到底能渲染出多少种不同句式」量出来。

输入：chars.jsonl + 题型白名单 + 每型取样几个字 + 种子。
输出：{题型: {samples, space, stems, verified, explanations, skeletons, skeleton_tails, seconds}}。
预期：**穷举整段句式空间**，不是抽样——「每型 ≥500 种句式」这条指标只有穷举才说得准，
      抽样只能给下界。取样按字库顺序取每种题型的头 N 个字（确定性、可复现）；
      同一题型换一个字就换一批事实，所以句式集合要取并集。

「不同句式」不是一个数，是三层，所以三个都报，免得把最外层误读成最里层：
  - `skeletons`：**问法骨架**条数（`offline._CORES` 里被用到的条数），这是真正的「问法」；
  - `skeleton_tails`：骨架 × 收尾语（「请写出答案。」这类）；
  - `stems`：真的渲染出来的不同题干（再把 4 种引号写法算进去）——指标「≥500」量的是这一层。
统计走 `offline.axes`：与 `render` 共用同一份进制拆法，所以数出来的下标就是渲染真正用的下标。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..facts.index import CharacterIndex
from ..facts.loader import load_characters
from ..llm.verbalizer import Verbalized
from ..tasks.base import TaskSpec
from ..tasks.registry import CONSTRUCTORS, build_all
from ..verify.checks import verify
from . import offline as O
from .generate import render_difficulty, spec_rng


@dataclass(frozen=True)
class Census:
    """输入：无。输出：一条题的句式普查结果。预期：三层句式数各留一个字段，一起报。"""

    stems: set[str] = field(default_factory=set)
    verified: set[str] = field(default_factory=set)
    explanations: set[str] = field(default_factory=set)
    skeletons: set[int] = field(default_factory=set)
    skeleton_tails: set[tuple[int, int]] = field(default_factory=set)
    space: int = 0


def census(spec: TaskSpec) -> Census:
    """输入：一条 TaskSpec。输出：`Census`（见上面的字段说明）。
    预期：把这道题的**整个**句式空间枚举一遍（题干 × 解析写法，两者都是独立轴）。渲染不出来的
          变体（非选择题的答案落在题干里，`render` 返回 None）按设计直接跳过，不进任何集合。
          `verified` 是 `stems` 的子集，两者相等就说明模板池渲染出的每一句都过得了 L4 校验。
          校验要连解析一起看（`answer_in_explanation` / `numbers_from_facts` 都读解析），
          所以每一对 (题干, 解析) 都真过一遍，不拿「解析不影响校验」当理由跳验。
    """
    space = O.variant_space(spec)
    result = Census(space=space)
    for variant in range(space):
        rendered = O.render(spec, variant)
        if rendered is None:
            continue
        stem, explanation = rendered
        result.stems.add(stem)
        result.explanations.add(explanation)
        picked = O.axes(spec.task_id, variant, has_options=bool(spec.options))
        if picked is not None:
            result.skeletons.add(picked.core)
            result.skeleton_tails.add((picked.core, picked.tail))
        if verify(spec, Verbalized(ok=True, stem=stem, explanation=explanation)).tier == "A":
            result.verified.add(stem)
    return result


def measure(
    *,
    data_path: Path | str | None = None,
    task_ids: Sequence[str] = (),
    per_type: int = 1,
    seed: int = 20261006,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """输入：数据路径 + 题型白名单 + 每型取样数 + 种子 + 进度回调。输出：{题型: 统计}。
    预期：每个题型取头 `per_type` 个字做样本；取不满就按实际条数算并如实记进 `samples`，
          不假装取满。题型在字库里一个字都出不来时给全 0，不静默消失。
    """
    say = progress or (lambda _msg: None)
    characters = load_characters(data_path)
    index = CharacterIndex(characters)
    # 释义类题（T13 / T17）的句式由模型写，模板池里没有它们（见 `offline.LLM_ONLY_TASK_IDS`）：
    # 不排除就会给出一行「句式空间 0」，看着像模板池漏了，其实是按纪律不走这条路。
    wanted = [
        tid for tid in CONSTRUCTORS if (not task_ids or tid in task_ids) and O.has_templates(tid)
    ]
    picked: dict[str, list[TaskSpec]] = {tid: [] for tid in wanted}
    for character in characters:
        if all(len(picked[tid]) >= per_type for tid in wanted):
            break
        for task_id in wanted:
            if len(picked[task_id]) >= per_type:
                continue
            # 随机源按 (题型, 字) 独立（同 `generate`）：不然「取样了哪几型」会挪动这一型的随机流，
            # 同一道题的候选 / 句式随白名单变化，量出来的句式数也就跟着抖。
            rng = spec_rng(seed, task_id, getattr(character, "ziid", ""))
            try:
                # 走 `build_all`（唯一入口）+ 生产默认难度：量到的句式空间就是生产会渲染的那一批
                # （含多跳 T9/T12、多约束 T8）。
                built = build_all(
                    character, rng=rng, difficulty=render_difficulty(), index=index, only=(task_id,),
                )
            except Exception:  # noqa: BLE001 - 单个字构造不出来就跳过，不带走整轮统计
                built = []
            if built:
                picked[task_id].append(built[0])

    report: dict[str, dict[str, Any]] = {}
    for task_id in wanted:
        started = time.perf_counter()
        stems: set[str] = set()
        verified: set[str] = set()
        explanations: set[str] = set()
        skeletons: set[int] = set()
        skeleton_tails: set[tuple[int, int]] = set()
        space = 0
        for spec in picked[task_id]:
            one = census(spec)
            stems |= one.stems
            verified |= one.verified
            explanations |= one.explanations
            skeletons |= one.skeletons
            skeleton_tails |= one.skeleton_tails
            space += one.space
        report[task_id] = {
            "samples": len(picked[task_id]),
            "space": space,
            "stems": len(stems),
            "verified": len(verified),
            "explanations": len(explanations),
            "skeletons": len(skeletons),
            "skeleton_tails": len(skeleton_tails),
            "seconds": round(time.perf_counter() - started, 2),
        }
        say(
            f"{task_id}: 句式空间 {space}，不同句式 {len(stems)}"
            f"（问法骨架 {len(skeletons)}，骨架×收尾 {len(skeleton_tails)}），"
            f"其中过校验 {len(verified)}，不同解析 {len(explanations)}"
            f"（{report[task_id]['seconds']}s）"
        )
    return report
