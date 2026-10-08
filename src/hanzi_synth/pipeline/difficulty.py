"""L5 难度标定：给「难度旋钮」一个可解释的期望通过率，并在**真有作答数据**时做 Rasch 校准。

诚实原则（主人明确要求「不许假装标定过」）：
- `target_facility` 是**解析式先验**，不需要数据，只用于给每条产出打一个可比较的难度标尺，
  它**不是实测值**，字段名如实叫 `kind: "analytic-prior"`；
- `estimate_irt_from_responses` 才是真正的标定：没有作答数据时如实返回 `calibrated: False`，
  绝不编一组参数出来。

本模块不做网络 IO；标定结果写旁路 JSON，不改产出。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..tasks.base import CONSTRAINT_TASKS, HOP_TASKS, Difficulty
from .dedup import load_jsonl

# 默认等价分支数：一条题面把答案锁定在「1 个正确 + K 个等价干扰」里
DEFAULT_BRANCHES = 4
# `HOP_TASKS` / `CONSTRAINT_TASKS` 定义在 tasks/base.py——难度阶梯要在 `build_all` 里展开，
# 而 tasks 层不能反向依赖 pipeline 层。这里只做再导出，判据只有一份。


def branches_for(task_id: str, constraints: int) -> int:
    """输入：题型 ID + 该题型的**已落定**约束个数。输出：该题型的等价分支数 K。
    预期：T8 的 K = 约束个数（见 `target_facility` 对方向的解释），其余题型用默认值。
          只认已知题型，未知题型按默认值处理，不抛错。
    """
    if task_id in CONSTRAINT_TASKS:
        return max(1, int(constraints))
    return DEFAULT_BRANCHES


def target_facility(*, hops: int = 1, branches: int = DEFAULT_BRANCHES) -> float:
    """输入：跳数 + 等价分支数 K。输出：期望通过率 0..1（解析式先验，非实测）。
    预期：单跳通过率 p = K/(K+1)，`hops` 跳独立串联则 facility = 1 - (1 - p) ** hops。

    方向说明：K 是**独立校验机会数**。约束越多，解空间被压得越小、唯一解越确定，
    因此先验上更容易命中——这与「约束越多越难」的直觉相反，但 T8 只在答案唯一时才出题，
    多的那条约束是「帮你排除」而不是「害你猜」。方向和假设都写在这里，不藏在代码里。
    """
    if hops < 1:
        raise ValueError(f"hops 必须 >= 1，收到 {hops}")
    if branches < 1:
        raise ValueError(f"branches 必须 >= 1，收到 {branches}")
    single_hop = branches / (branches + 1)
    return round(1.0 - (1.0 - single_hop) ** hops, 6)


def difficulty_note(
    task_id: str,
    difficulty: Difficulty,
    *,
    resolved_depth: str = "",
    resolved_hops: int | None = None,
    resolved_constraints: int | None = None,
) -> dict:
    """输入：题型 ID + 难度设定 + 本条实际用到的干扰项档位 / 跳数 / 约束个数（都可空）。
    输出：写进产出的难度注记。
    预期：把「哪些旋钮真的作用于这个题型」和解析式期望通过率一起记下来，
          让每条产出自带可比较的标尺；不生效的旋钮写 None，不假装它起作用。
          `distractor_depth` / `hops` / `constraints` 写**真实用到的值**（`mixed` 抽成具体档、
          难度阶梯退档之后的结果），对应的 `*_mode` 保留配置档——这样既看得出这条题多难，
          也看得出它是抽样 / 退档来的。
    """
    hops = resolved_hops if resolved_hops is not None else difficulty.hops
    hops = int(hops) if task_id in HOP_TASKS else 1
    constraints = resolved_constraints if resolved_constraints is not None else difficulty.constraints
    constraints = int(constraints) if task_id in CONSTRAINT_TASKS else None
    branches = branches_for(task_id, constraints if constraints is not None else DEFAULT_BRANCHES)
    return {
        "kind": "analytic-prior",
        "hops": hops,
        "hops_mode": difficulty.hops if task_id in HOP_TASKS else None,
        "constraints": constraints,
        "constraints_mode": difficulty.constraints if task_id in CONSTRAINT_TASKS else None,
        "distractor_depth": resolved_depth or difficulty.distractor_depth,
        "depth_mode": difficulty.distractor_depth,
        "branches": branches,
        "target_facility": target_facility(hops=hops, branches=branches),
    }


def _sigmoid(value: float) -> float:
    """输入：实数。输出：logistic 值 0..1。预期：极值下不溢出（分段计算）。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _as_bool(value: Any) -> bool | None:
    """输入：任意取值。输出：True / False / None（无法判定）。预期：兼容 0/1 与常见中英文字面。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "对", "正确", "是"):
            return True
        if text in ("0", "false", "no", "n", "错", "错误", "否"):
            return False
    return None


def load_responses(path: Path) -> list[dict]:
    """输入：作答 jsonl 路径。输出：记录列表（每条含 person / item / correct）。
    预期：坏行跳过；缺字段的行保留，由 `estimate_irt_from_responses` 统一过滤并计数。
    """
    rows, _skipped = load_jsonl(path)
    return rows


def _log_likelihood(pairs: Mapping[tuple[str, str], bool], theta: Mapping[str, float], beta: Mapping[str, float]) -> float:
    """输入：作答对 + 能力 + 难度。输出：对数似然。预期：用于报告收敛质量。"""
    total = 0.0
    for (person, item), correct in pairs.items():
        probability = _sigmoid(theta[person] - beta[item])
        probability = min(max(probability, 1e-12), 1.0 - 1e-12)
        total += math.log(probability) if correct else math.log(1.0 - probability)
    return total


def estimate_irt_from_responses(
    responses: Sequence[Mapping[str, Any]],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    boundary_eps: float = 1e-9,
) -> dict:
    """输入：作答记录序列，每条 {"person": str, "item": str, "correct": bool|0|1}。
    输出：Rasch(1PL) 联合极大似然（JMLE）校准结果。
    预期：无可用作答 -> `{"calibrated": False, "reason": ...}`，**绝不编参数**；
          有数据则用 Newton 迭代同时估能力 θ 与难度 b，并以 mean(b)=0 做识别约束；
          某人全对/全错、某题全对/全错时参数无界，如实标 `boundary: True`，不假装估出来了。
    """
    pairs: dict[tuple[str, str], bool] = {}
    for row in responses:
        if not isinstance(row, Mapping):
            continue
        person = str(row.get("person", "")).strip()
        item = str(row.get("item", "")).strip()
        correct = _as_bool(row.get("correct"))
        if not person or not item or correct is None:
            continue
        pairs[(person, item)] = correct
    if not pairs:
        return {
            "calibrated": False,
            "reason": "没有可用的作答记录（每条需要 person / item / correct 三个字段）",
        }

    persons = sorted({person for person, _item in pairs})
    items = sorted({item for _person, item in pairs})
    theta = {person: 0.0 for person in persons}
    beta = {item: 0.0 for item in items}

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        max_delta = 0.0
        for person in persons:  # 能力：Δθ = Σ(x - p) / Σp(1-p)
            info = residual = 0.0
            for item in items:
                key = (person, item)
                if key not in pairs:
                    continue
                probability = _sigmoid(theta[person] - beta[item])
                info += probability * (1.0 - probability)
                residual += (1.0 if pairs[key] else 0.0) - probability
            if info > boundary_eps:
                delta = residual / info
                theta[person] += delta
                max_delta = max(max_delta, abs(delta))
        for item in items:  # 难度：Δb = -Σ(x - p) / Σp(1-p)
            info = residual = 0.0
            for person in persons:
                key = (person, item)
                if key not in pairs:
                    continue
                probability = _sigmoid(theta[person] - beta[item])
                info += probability * (1.0 - probability)
                residual += (1.0 if pairs[key] else 0.0) - probability
            if info > boundary_eps:
                delta = residual / info
                beta[item] -= delta
                max_delta = max(max_delta, abs(delta))
        # 识别约束：似然对 θ->θ+c、b->b+c 不变，把 b 平移到均值 0 并同步平移 θ
        shift = sum(beta.values()) / len(beta)
        for item in items:
            beta[item] -= shift
        for person in persons:
            theta[person] += shift
        if max_delta < tol:
            converged = True
            break

    item_parameters: dict[str, dict] = {}
    for item in items:
        answers = [pairs[(person, item)] for person in persons if (person, item) in pairs]
        item_parameters[item] = {
            "difficulty": round(beta[item], 6),
            "responses": len(answers),
            "boundary": all(answers) or not any(answers),
        }
    person_parameters: dict[str, dict] = {}
    for person in persons:
        answers = [pairs[(person, item)] for item in items if (person, item) in pairs]
        person_parameters[person] = {
            "ability": round(theta[person], 6),
            "responses": len(answers),
            "boundary": all(answers) or not any(answers),
        }
    return {
        "calibrated": True,
        "model": "rasch-1pl-jmle",
        "iterations": iterations,
        "converged": converged,
        "persons": len(persons),
        "items": len(items),
        "responses": len(pairs),
        "log_likelihood": round(_log_likelihood(pairs, theta, beta), 4),
        "item_parameters": item_parameters,
        "person_parameters": person_parameters,
    }


def write_calibration(path: Path, result: Mapping[str, Any]) -> Path:
    """输入：目标路径 + 校准结果。输出：实际写入的路径。预期：UTF-8 JSON，父目录自动创建。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
