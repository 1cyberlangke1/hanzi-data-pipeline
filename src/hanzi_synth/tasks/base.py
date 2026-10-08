"""L2 题目构造器的公共契约。

核心纪律：
- `answer` 与 `proof` 完全由**程序**产生，LLM 只做措辞；
- 答案必须是**字段的直接值**，不是字段间推导（「假设字典一定对」）；
- 不能验证的题型宁可不做。

L2 输出的是「已确定的事实 + 答案 + 推理链 + 干扰项」。其中 `proof` 与 `distractors[].why_wrong`
是**给 L3 的措辞素材**（只作输入，不进产出记录）；产出记录里的自然语言（`stem` / `explanation`）
一律由 L3（llm.verbalizer）调用 LLM 写，绝不接受程序拼接的句子。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..facts.records import Character


@dataclass(frozen=True)
class Difficulty:
    """输入：三个**已被真实使用**的难度旋钮。输出：不可变难度设定。
    预期：难度只由结构决定，不由措辞长度决定（明确排除伪难度）。

    三个旋钮都另有**分布档** `mixed`：按权重抽样落成具体值，让难度在**数据集层面**成分布，
    而不是全表一个值。`resolved()` 负责把 mixed 落定，`build_all` 每个字调一次。

    旋钮的消费方（改这里必须同步改构造器，否则又会变成死代码）：
    - `hops`             → T9 部件链的链长、T12 结构变换的编辑步数；
    - `constraints`      → T8 的约束档位（1 = 部首+总笔画，2 = 再加顶层结构，
                           3 = 再加部件个数）。只给部首时桶里永远有几十上百个字，
                           判不出唯一解，所以最低档就是两条条件。
    - `distractor_depth` → T1/T3/T4/T5/T10/T11/T12 的干扰项选取范围（near 同部首近笔画、
                           mid 同笔画异部首、far 全池随机）。

    原设计里的第四个旋钮 `frequency_tier`（常用/生僻）**已删除**：`chars.jsonl` 的
    `property` 里根本没有任何字频字段（只有 font_variant/head*/tail*/stroke_*/structure/
    standard_form/zi/ziid 与 refs），没有数据源就只能靠猜，猜出来的难度标签是假的。
    """

    hops: int | str = 1
    constraints: int | str = 1
    distractor_depth: str = "near"  # near / mid / far / mixed

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期：越界值当场报错，而不是静默产出降级题目。"""
        if not _is_level(self.hops):
            raise ValueError(f"hops 必须是 1..3 或 {MIXED_LEVEL!r}，收到 {self.hops!r}")
        if not _is_level(self.constraints):
            raise ValueError(f"constraints 必须是 1..3 或 {MIXED_LEVEL!r}，收到 {self.constraints!r}")
        if self.distractor_depth not in DISTRACTOR_DEPTHS:
            raise ValueError(f"distractor_depth 必须是 {DISTRACTOR_DEPTHS} 之一，收到 {self.distractor_depth!r}")

    def resolved(self, rng: random.Random) -> "Difficulty":
        """输入：随机源。输出：把三个 `mixed` 都落成具体值后的 Difficulty。
        预期：`build_all` 每个字调一次——同一个字的跳数 / 约束数 / 干扰项档位一致，
              难度注记里写的是**真实用到的值**，不是配置里的 `mixed`。
              一个 mixed 都没有时原样返回自己（不新建对象、不消耗随机数）。
        """
        if (
            self.hops != MIXED_LEVEL
            and self.constraints != MIXED_LEVEL
            and self.distractor_depth != MIXED_DEPTH
        ):
            return self
        return Difficulty(
            hops=resolve_hops(self.hops, rng),
            constraints=resolve_constraints(self.constraints, rng),
            distractor_depth=resolve_depth(self.distractor_depth, rng),
        )


# 干扰项距离档位：near 最像（难）、far 最不像（易）
DISTRACTOR_DEPTHS: tuple[str, ...] = ("near", "mid", "far", "mixed")

# 难度旋钮的分布档：与 `distractor_depth` 的 `mixed` 同一思路，让难度在数据集层面成分布。
MIXED_LEVEL = "mixed"
# 权重是「难 : 中 : 易」。1 跳 / 1 条约束是覆盖保底，权重最高——抽到高档位却凑不出题时，
# `build_all` 会沿阶梯退到最低档（见 `tasks.registry._difficulty_ladder`）。
HOPS_MIX_WEIGHTS: tuple[tuple[int, int], ...] = ((3, 3), (2, 3), (1, 4))
CONSTRAINT_MIX_WEIGHTS: tuple[tuple[int, int], ...] = ((3, 3), (2, 3), (1, 4))

# 真正消费 hops / constraints 的题型。放在 tasks 层而不是 pipeline/difficulty.py：
# 难度阶梯要在 build_all 里展开，而 tasks 层不能反向依赖 pipeline 层（会成环）。
HOP_TASKS: tuple[str, ...] = ("T9_component_chain", "T12_ids_transform")
CONSTRAINT_TASKS: tuple[str, ...] = ("T8_constraint",)

# `mixed` 是一个**分布档**而不是第四个距离：按权重抽样，让难度在数据集层面成分布
MIXED_DEPTH = "mixed"
MIXED_DEPTH_WEIGHTS: tuple[tuple[str, int], ...] = (("near", 4), ("mid", 3), ("far", 2))


def resolve_depth(depth: str, rng: random.Random) -> str:
    """输入：难度档（可为 `mixed`）与随机源。输出：一个具体档位（near / mid / far）。
    预期：`mixed` 时按 4:3:2 抽样（near 最像、最能考辨别力，所以给最高权重）；
          其余档位原样返回。抽样用传入的 rng，同种子可复现。
    """
    if depth != MIXED_DEPTH:
        return depth
    return _sample_weighted(MIXED_DEPTH_WEIGHTS, rng)


def _is_level(value: Any) -> bool:
    """输入：hops / constraints 的取值。输出：它是不是合法档位（1..3 或 `mixed`）。
    预期：给 `Difficulty.__post_init__` 用；`True` / `False` 不算数字档位（bool 是 int 的子类）。
    """
    if value == MIXED_LEVEL:
        return True
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 3


def _sample_weighted(weights: tuple[tuple[Any, int], ...], rng: random.Random) -> Any:
    """输入：[(值, 权重)] + 随机源。输出：按权重抽出的一个值。
    预期：权重是正整数，抽一次 randrange（同种子可复现）；权重之和非正时当场报错。
    """
    total = sum(weight for _value, weight in weights)
    if total <= 0:
        raise ValueError(f"权重之和必须为正，收到 {weights!r}")
    roll = rng.randrange(total)
    for value, weight in weights:
        if roll < weight:
            return value
        roll -= weight
    return weights[-1][0]


def resolve_hops(value: int | str, rng: random.Random) -> int:
    """输入：hops 档（1..3 或 `mixed`）+ 随机源。输出：具体跳数。
    预期：`mixed` 时按 3:3:4 抽样——多跳更接近真实难度，但 1 跳是覆盖保底，权重最高；
          非 mixed 原样返回且**不消耗随机数**（否则固定档位的产出会被无谓扰动）。
    """
    if value != MIXED_LEVEL:
        return value
    return _sample_weighted(HOPS_MIX_WEIGHTS, rng)


def resolve_constraints(value: int | str, rng: random.Random) -> int:
    """输入：constraints 档（1..3 或 `mixed`）+ 随机源。输出：具体约束个数。
    预期：与 `resolve_hops` 同一条规矩（同一套权重、非 mixed 不消耗随机数）。
    """
    if value != MIXED_LEVEL:
        return value
    return _sample_weighted(CONSTRAINT_MIX_WEIGHTS, rng)


@dataclass(frozen=True)
class Distractor:
    """输入：一个错误选项 + 它错在哪。输出：不可变干扰项。
    预期：`why_wrong` 由程序写，天然正确，可直接当解析素材。
    """

    text: str
    why_wrong: str


@dataclass(frozen=True)
class TaskSpec:
    """输入：构造器产出的全部要素。输出：一道题的「程序侧真相」。
    预期：`facts` 是唯一允许进入 L3 的事实来源；`answer` 与 `answer_aliases` 构成判分集合。
    """

    task_id: str
    # 给 L3 的**机器面**出题说明：一句中文的「要问什么」，写成**名词性短语**（「「㐀」的总笔画数」）。
    # 名词性短语不是问句——小模型就算原样照抄也落不成一道题，往返检验的 `is_question` 当场判否，
    # 所以不必再靠英文切断照抄路径（实测换成名词性短语后照抄从 9/12 降到 0/12）。
    # 它说的是「这道题要问什么」，不是最终题干——题干必须由 L3 用中文自己写。
    instruction: str
    facts: Mapping[str, Any]
    answer: str
    answer_aliases: tuple[str, ...] = ()
    surface_forms: tuple[str, ...] = ()
    # 选择题的候选列表：**必须独立成字段**（程序侧真相，供裁判与消费方读），
    # 同时**题干必须逐个原样列出这些候选**——只写「从下列选项中选」读者根本没得选（真机实测踩过）。
    # 因此选择题不做 `no_answer_leak`（答案就是候选之一，必然出现在题干里），
    # 改由 `options_ok` 的「每个候选只出现一次 + 不得点出答案在第几个」兜住。
    options: tuple[str, ...] = ()
    # 题干必须**出现**的字符串（空 = 无要求）：问「这个字」的题必须让读者看得见是哪个字，
    # 否则产出就是「这个字的 Unicode 码位是多少？」这种谁也答不了的残废题（真机实测踩过）。
    visible_subject: str = ""
    # 题型声明的、**题干里必须出现**的其它串（`visible_subject` 之外的）。反向题靠它要求写出读音——
    # 没有读音，读者就认不出是哪个字，题干也就不自足了；写进这里才是校验，不是写作要求。
    must_contain: tuple[str, ...] = ()
    proof: tuple[str, ...] = ()
    distractors: tuple[Distractor, ...] = ()
    slots: tuple[str, ...] = ("stem", "explanation")
    provenance: Mapping[str, Any] = field(default_factory=dict)
    # 本次实际使用的干扰项档位（`mixed` 解析后的结果）；空串 = 构造器未声明。
    # 有了它，产出的 difficulty 注记才写得出「这条到底是 near 还是 far」。
    resolved_depth: str = ""
    # 本次实际使用的跳数 / 约束个数（`mixed` 解析 + 难度阶梯回退之后的结果）。
    # 不消费该旋钮的题型恒为 1 / 1，与 `difficulty_note` 的标准一致。
    resolved_hops: int = 1
    resolved_constraints: int = 1

    # 题干**至少要命中其中一个**说法（空 = 不检查）：这是「题干问的必须是答案那件事」的校验。
    # 实测 T13 的题干写「请问这个字是什么？」——问的是那个字本身，可答案是它的 IDS 拆解，
    # 读者照着题干答必然答错。声明后，题干里一个都不出现即判不合格，走重试换一条措辞路径。
    stem_asks_any: tuple[str, ...] = ()
    # 本题答案的**字形结构位置词**（构造器声明；空 = 本题与字形结构无关，两条校验都不启用）。
    # 实测 T13 的小模型产出：题干写「字形结构属于左上包围，上部部件为𠂆……」——把答案那棵 IDS 树
    # 用散文复述了一遍；解析又写「按照左右结构与上下结构组合而成」，而答案 ⿸𠂆⿱𫜹𠃌 里根本
    # 没有左右结构。声明后：
    #   1. 题干里一个位置词都不许出现（候选之外的写法都算泄题）；
    #   2. 解析里出现的位置词必须来自本清单（不许编出答案里没有的结构）。
    structure_words: tuple[str, ...] = ()

    # 本题的**隐藏主语**（空 = 无）：题干必须靠别的线索（释义、读音）让读者认出这个字，
    # 绝不能直接把它写出来。T13 反向题的风险点——题干若写「「㐆」的 IDS 拆解是什么」，
    # 读者不用看释义，查一下就行，反向题退化成 T3 正向题。
    # 构造器保证 `meaning_source` 里本来就不含这个字（`character.zi not in cleaned` 才选用），
    # 所以「题干里出现它」只可能是模型自己写出来的——判据是**可靠的子串判据**，不会误杀。
    hidden_subject: str = ""

    # 措辞层（L3）**真正需要**的事实键：`llm.verbalizer.payload_for` 只把这些键发给模型。
    # 空 = 发全部（只给手工构造的 spec 用）；注册题型一律显式声明，`test_llm` 有单测钉住。
    # 为什么要分开：`facts` 是**给校验和离线渲染用的全集**（判分、干扰项、模板都要读它），
    # 而措辞模型只需要「写这道题要引用的那几个值」。两者混在一起时，模型会拿别的题的材料
    # 当素材——实测 T16 的 `codepoint_forms` 里混着 UTF-8 字节序列，产出写成
    # 「给出码位 E3 90 8B，让读者从下列选项中选出对应的字」（那条已在构造器侧收窄）。
    # 示范（`FEW_SHOT_TURNS`）
    # 的 facts 形状就是「只发写这道题需要的键」，这里让真实提示内容与示范同形。
    payload_facts: tuple[str, ...] = ()

    @property
    def accepted(self) -> tuple[str, ...]:
        """输入：无。输出：判分时接受的答案集合（主答案 + 别名，去重保序）。
        预期：多解 IDS 一律算对——「一个字当然有多种写法」。
        """
        seen: list[str] = []
        for item in (self.answer, *self.answer_aliases, *self.surface_forms):
            if item and item not in seen:
                seen.append(item)
        return tuple(seen)


def make_rng(seed: int) -> random.Random:
    """输入：整数种子。输出：独立的随机数发生器。
    预期：同一 seed 必须复现同一批题（README 铁律 6：全链路可追溯）。
    """
    return random.Random(seed)


def provenance(character: Character, constructor_id: str, version: str, rng_seed: int) -> dict:
    """输入：字记录 + 构造器标识 + 版本 + 随机种子。输出：溯源字典。
    预期：写进每条产出，用于回查「这条数据是怎么来的」。
    """
    return {
        "seed_ziid": character.ziid,
        "constructor_id": constructor_id,
        "constructor_version": version,
        "rng_seed": rng_seed,
    }
