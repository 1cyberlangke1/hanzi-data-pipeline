"""推理类题目构造器：T8 约束满足 / T9 部件链多跳 / T10 反例否定 / T11 组合排序 / T12 结构变换。

五个题型共用的三条纪律：
1. **答案仍是字段直接值**——T8 的答案是那个字本身，T9 的答案是 IDS 树里的一个叶子部件，
   T10/T11 的答案是选项里的某个字，T12 的答案是「真实收录」的某个字；全部可由程序从
   `chars.jsonl` 复算，不需要任何外部知识；
2. **解必须唯一**——T8 用索引判「约束桶里只剩本字」，T9 用位置标签锁定唯一部件，
   T12 用挖空索引保证「替换后确实收录」；做不到唯一就不出题；
3. **选择题的选项独立成 `TaskSpec.options`**，题干只负责把候选**逐个原样列出**（读者要有
   得选）；答案就是候选之一，所以选择题不做 `no_answer_leak`，改由 `options_ok` 兜住。

难度旋钮在这里才真正生效（此前 `Difficulty` 是死代码）：
- `hops`             → T9 的链长、T12 的编辑步数；
- `constraints`      → T8 同时施加的条件个数（部首 / 总笔画 / 顶层结构）；
- `distractor_depth` → 各题干扰项的距离档。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..facts.ids import IdsSyntaxError, leaves, parse_tree, render, replace_leaf, top_operator
from ..facts.index import CharacterIndex
from ..facts.records import Character
from ..facts.surface import IDC_LABELS
from .base import Difficulty, Distractor, TaskSpec, provenance as make_provenance, resolve_depth

CONSTRUCTOR_VERSION = "0.1.0"

# 干扰项候选池的抽样上界：避免对全池（7.9 万条）做整表 shuffle
_SAMPLE_CAP = 200

# 位置标签：下标与 IDS 描述符的书写顺序一致（第 0 个就是最左边/最上面那个部件）
_POSITION_LABELS: dict[str, tuple[str, ...]] = {
    "\u2ff0": ("左边", "右边"),
    "\u2ff1": ("上边", "下边"),
    "\u2ff2": ("左边", "中间", "右边"),
    "\u2ff3": ("上边", "中间", "下边"),
    "\u2ff4": ("外框", "内部"),
    "\u2ff5": ("外框", "内部"),
    "\u2ff6": ("外框", "内部"),
    "\u2ff7": ("外框", "内部"),
    "\u2ff8": ("包围部件", "被包围部件"),
    "\u2ff9": ("包围部件", "被包围部件"),
    "\u2ffa": ("包围部件", "被包围部件"),
    "\u2ffb": ("第一个部件", "第二个部件"),
    "\u2ffc": ("右包围部件", "被包围部件"),
    "\u2ffd": ("左下包围部件", "被包围部件"),
    "\u31ef": ("第一个部件", "第二个部件"),
}


def _position_labels(operator: str, arity: int) -> tuple[str, ...]:
    """输入：IDS 描述符与子节点个数。输出：按书写顺序的位置名。
    预期：表里有就用人话名，没有就退化成「第 N 个部件」——绝不编造方位。
    """
    labels = _POSITION_LABELS.get(operator)
    if labels and len(labels) == arity:
        return labels
    return tuple(f"第 {index + 1} 个部件" for index in range(arity))


def _leaf_at(ids: str, position: int) -> str | None:
    """输入：IDS 串与叶子下标。输出：该位置的叶子部件字符；越界或非法时 None。"""
    try:
        parts = leaves(parse_tree(ids))
    except IdsSyntaxError:
        return None
    return parts[position] if 0 <= position < len(parts) else None


def _pick(
    tiers: Sequence[Iterable[Character]],
    banned: Sequence[str],
    rng: random.Random,
    why_wrong: Callable[[Character], str],
    count: int = 3,
) -> tuple[Distractor, ...]:
    """输入：按距离从近到远的候选池、禁用文本、随机源、错误说明模板、条数。输出：干扰项。
    预期：逐档取样，绝不重复、绝不落在 banned 里；凑不满就返回现有的，不硬凑。
    """
    seen = {text for text in banned if text}
    picked: list[Distractor] = []
    for tier in tiers:
        pool = [c for c in tier if c.zi]
        # 全池 7.9 万条 shuffle 一次要几百毫秒，这里只随机抽一个上界固定的子集，
        # 抽样本身就是均匀的，效果等价但成本从 O(池) 降到 O(200)
        if len(pool) > _SAMPLE_CAP:
            pool = rng.sample(pool, _SAMPLE_CAP)
        rng.shuffle(pool)
        for candidate in pool:
            if candidate.zi in seen:
                continue
            seen.add(candidate.zi)
            picked.append(Distractor(text=candidate.zi, why_wrong=why_wrong(candidate)))
            if len(picked) >= count:
                return tuple(picked)
    return tuple(picked)


# ---------------------------------------------------------------- T8 约束满足


def _constraint_pool(character: Character, index: CharacterIndex, level: int) -> tuple[list[Character], str, dict]:
    """输入：字记录、索引、约束档位（1..3）。输出：(满足约束的字, 人话描述, facts 片段)。
    预期：档位对应**实际可判唯一**的条件组合（实测：只给部首时桶里永远有几十上百个字，
          根本判不出唯一解，所以最低档就是「部首 + 总笔画」）：
          - 1 -> 部首 + 总笔画；
          - 2 -> 再加顶层结构；
          - 3 -> 再加部件个数（从 IDS 直接数出来）。
    """
    radical = character.radical or ""
    stroke = character.stroke_total
    operator = top_operator(character.ids_candidates[0])
    pool = list(index.by_radical.get(radical, ()))
    parts = [f"部首是「{radical}」", f"总笔画是 {stroke} 画"]
    facts: dict = {"radical": radical, "stroke_total": stroke}
    pool = [c for c in pool if c.stroke_total == stroke]
    if level >= 2:
        structure = IDC_LABELS.get(operator or "", operator or "")
        pool = [c for c in pool if c.ids_candidates and top_operator(c.ids_candidates[0]) == operator]
        parts.append(f"顶层结构是「{structure}」")
        facts["structure"] = structure
    if level >= 3:
        count = len(leaves(parse_tree(character.ids_candidates[0])))
        pool = [c for c in pool if c.ids_candidates and _leaf_total(c) == count]
        parts.append(f"一共由 {count} 个部件组成")
        facts["component_count"] = count
    return pool, "，".join(parts), facts


def _leaf_total(character: Character) -> int:
    """输入：字记录。输出：它 IDS 里的部件个数；解析失败返回 -1（永不等于任何真实个数）。"""
    try:
        return len(leaves(parse_tree(character.ids_candidates[0])))
    except IdsSyntaxError:
        return -1


# **题干校验**（判据在 L4 的 `stem_asks`）：校验空着 = 不设防，实测第十轮真机因此放出了
# 「题干问读音、答案是笔画数」的废题。以下四条按「答案是什么」列同义说法。
# 代价（549 条已发布产出逐条量）：T8 要补「一个字 / 一个汉字」（实测有产出写「请写出一个汉字，
# 其部首是「宀」」）、T10 要补「选出 / 有别」、T12 要补「哪个」；补完只有 1 条 T10 被没通过——
# 那条的题干写「请判断下列四个汉字「緻」「㐀」「𲷳」「𫄩」的部首关系。」，压根没说要从四个里挑出
# 哪一个（读者不知道该答什么），是**真废题**。
_CONSTRAINT_ASK_MARKERS = ("什么字", "哪个字", "哪一个字", "哪一个", "一个汉字", "一个字", "满足", "符合", "是什么")
_ODD_ASK_MARKERS = ("哪一个", "哪个", "不同", "不一样", "异类", "有别", "选出")
_EXTREME_ASK_MARKERS = ("笔画", "画数", "几画", "几笔", "多少画", "多少笔")
_TRANSFORM_ASK_MARKERS = ("哪个字", "什么字", "哪个", "哪一个", "得到", "结果", "是什么")
# T9 的校验（答案是一个**部件**）：原先内联在构造器里，提出来是因为
# `registry.TASK_STEM_ASKS` 要按题型拿到它——旧产出记录的 `stem_asks_any` 是空的，
# 复验时得用**当前**题型声明的校验补上（见 `pipeline.run.spec_from_record`）。
_CHAIN_ASK_MARKERS = ("部件", "哪个字", "什么字", "哪一个字", "何字")


def build_constraint(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T8：给若干条约束，问满足条件的字是哪个。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；缺字段、答案会出现在题干里、或约束桶不是唯一解时返回 None。
    预期：唯一性用索引实判（桶里去重后只剩本字），绝不靠「看起来挺独特」。
    """
    if not character.zi or not character.radical or character.stroke_total <= 0 or not character.ids_candidates:
        return None
    if character.zi == character.radical:
        return None  # 题干的部首描述里会直接出现答案
    level = difficulty.constraints
    pool, description, facts = _constraint_pool(character, index, level)
    if {c.zi for c in pool if c.zi} != {character.zi}:
        return None  # 不是唯一解 -> 歧义题，丢弃

    def satisfies(candidate: Character) -> bool:
        if candidate.radical != character.radical:
            return False
        if candidate.stroke_total != character.stroke_total:
            return False
        if level >= 2:
            if not candidate.ids_candidates:
                return False
            if top_operator(candidate.ids_candidates[0]) != top_operator(character.ids_candidates[0]):
                return False
        if level >= 3:
            if not candidate.ids_candidates:
                return False
            if _leaf_total(candidate) != _leaf_total(character):
                return False
        return True

    near = [c for c in index.by_radical.get(character.radical, ()) if not satisfies(c)]
    mid = [c for c in index.by_stroke.get(character.stroke_total, ()) if not satisfies(c)]
    far = [c for c in index.characters if not satisfies(c)]
    depth = resolve_depth(difficulty.distractor_depth, rng)
    tiers = {"near": (near, mid, far), "mid": (mid, near, far), "far": (far, mid, near)}
    distractors = _pick(
        tiers.get(depth, (near, mid, far)),
        [character.zi],
        rng,
        lambda c: f"「{c.zi}」不满足题干条件（{description}）",
    )
    return TaskSpec(
        task_id="T8_constraint",
        payload_facts=("conditions",),
        instruction=(
            "同时满足列出的全部条件的那个字"
        ),
        facts={**facts, "conditions": description, "answer_zi": character.zi},
        answer=character.zi,
        stem_asks_any=_CONSTRAINT_ASK_MARKERS,
        distractors=distractors,
        proof=(f"满足「{description}」的字是「{character.zi}」",),
        provenance=make_provenance(character, "T8_constraint", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


# ---------------------------------------------------------------- T9 部件链多跳


@dataclass(frozen=True)
class _Step:
    """输入：一次「按位置取部件」的解析结果。输出：不可变步骤。预期：标签唯一锁定该部件。"""

    label: str
    component: str


def _component_steps(character: Character) -> list[_Step]:
    """输入：字记录。输出：它的全部「位置 -> 单字部件」步骤（子树部件不算）。
    预期：只保留**直接是单字**的子节点——子树（如 ⿰七七）没法当字来指称。
    """
    try:
        tree = parse_tree(character.ids_candidates[0])
    except IdsSyntaxError:
        return []
    if tree.is_leaf:
        return []
    labels = _position_labels(tree.op or "", len(tree.children))
    return [
        _Step(label=labels[index], component=child.char or "")
        for index, child in enumerate(tree.children)
        if child.is_leaf and child.char
    ]


def build_component_chain(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T9：沿着 IDS 树走 hops 跳，问终点位置的部件是哪个字。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；链子接不起来（中间件不在池里 / 部件不是单字）时返回 None。
    预期：**中间件一律不点名**，只用位置标签描述——否则题目会退化成单跳查表。
          答案不得与题干里出现过的任何字重复，撞车就丢弃。
    """
    path: list[str] = []
    named: list[str] = [character.zi]
    cursor = character
    for _ in range(max(difficulty.hops - 1, 0)):
        steps = [s for s in _component_steps(cursor) if index.lookup_char(s.component)]
        if not steps:
            return None
        step = steps[rng.randrange(len(steps))]
        nxt = index.lookup_char(step.component)
        if nxt is None or not nxt.ids_candidates:
            return None
        path.append(step.label)
        named.append(step.component)
        cursor = nxt
    final_steps = _component_steps(cursor)
    if not final_steps:
        return None
    final = final_steps[rng.randrange(len(final_steps))]
    # 泄露判据必须按**子串**比，不能按相等比：题干为了写出路径，必然原样带上每个位置词
    # （「右边」「中」这类），而答案可能是位置词的子串。实测有 (路径含「右边」, 答案「右」)
    # 与 (路径含「中」, 答案「中华字海」) 两种组合——无论模型怎么写，题干都会同时带上
    # 路径词与答案，被判泄露后反复重跑，永远出不来。
    tokens = [*named, *path, final.label]
    if not final.component or any(
        final.component in token or token in final.component for token in tokens if token
    ):
        return None  # 答案与题干里出现过的字/位置词撞车 = 泄露
    path.append(final.label)

    return TaskSpec(
        task_id="T9_component_chain",
        payload_facts=("start_zi", "path"),
        instruction=(
            # 位置链是题干**必须出现**的串（`must_contain`）：只写「沿位置链」小模型就不写具体
            # 位置词，读者无从判断走哪一层，这道题不可作答（实测 0/2 → 2/2，probe_weave.jsonl）。
            f"从「{character.zi}」出发，沿 {'、'.join(path)} 这条位置链走到最里层的那个部件"
        ),
        facts={"start_zi": character.zi, "path": path, "depth": difficulty.hops},
        answer=final.component,
        visible_subject=character.zi,
        # 路径描述是**必须出现的串**，不只是写作要求：实测 16r 的 T9 题干写「起始汉字为㐑，
        # 按照描述逐层拆解，最内层的部件是哪个字？」——描述一个都没写，读者无从判断走哪一层，
        # 这道题不可作答。少了任何一个位置词都会被 `subject_visible` 拦下并重跑。
        must_contain=tuple(path),
        # 问的必须是答案那件事：答案是一个**部件**，问「这个字的结构是什么」读者必然答错
        # （实测 16p 的 T9 就是这么写的）
        stem_asks_any=_CHAIN_ASK_MARKERS,
        distractors=_pick(
            [
                list(index.same_radical(character)),
                list(index.by_stroke.get(character.stroke_total, ())),
                list(index.characters),
            ],
            [*named, final.component],
            rng,
            lambda c: f"「{c.zi}」不在题干描述的那个位置上",
        ),
        # proof 会被模型当素材抄进解析：里面绝不能出现「题干 / 题目」这类内部指代，
        # 否则解析必然带着它一起进产出（实测 16p 的 T9 解析写「按题目逐层拆解」）
        proof=(f"按位置描述逐层拆下去，最内层的部件是「{final.component}」",),
        provenance=make_provenance(character, "T9_component_chain", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


# ---------------------------------------------------------------- T10 反例否定


def build_odd_one_out(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T10：给一组选项，问哪一个的部首**不是**某个部首。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；找不到 3 个同部首同伴、或答案会出现在题干里时返回 None。
    预期：答案 = 本字（它的部首不是 facts.radical），干扰项 = 3 个**确实**属于该部首的字；
          选项走 `options` 字段，并要求题干把候选**逐个列出来**——读者看得见候选才答得了
          （只写「下列选项」的产出是废题）；答案只是候选之一，不构成泄露，
          因此选择题不做 `no_answer_leak`，由 `options_ok` 的「每个候选只出现一次」兜住。
    """
    if not character.zi or not character.radical:
        return None
    radicals = [r for r, members in index.by_radical.items() if r and r != character.radical and len(members) >= 3]
    if not radicals:
        return None
    radical = radicals[rng.randrange(len(radicals))]
    if character.zi == radical:
        return None  # 题干的部首名里会直接出现答案
    members = [c for c in index.by_radical[radical] if c.zi and c.zi != character.zi]
    if len(members) < 3:
        return None
    picked = rng.sample(members, 3)
    options = [character.zi, *(c.zi for c in picked)]
    rng.shuffle(options)
    return TaskSpec(
        task_id="T10_odd_one_out",
        payload_facts=("other_options_radical",),
        instruction=(
            # 候选**写进说明**：校验要求题干逐个列出候选（读者要有得选），而只写「候选里」
            # 小模型一个都不列（实测 0/2 → 2/2，probe_weave.jsonl）。生僻字由 `payload_for`
            # 统一换成占位符，模型不必自己敲。
            f"{'、'.join(options)} 里部首与其它几个不同的那个字"
        ),
        # 键名必须自解释：`radical` 是**其它三个候选共有**的部首，答案字的部首与它不同。
        # 旧键名 `radical` 有歧义，实测小模型据此写出自相矛盾的解析
        # （「「㐀」的部首是「子」，而其他三个字的部首也都是「子」，因此答案是「㐀」」）。
        facts={"other_options_radical": radical, "options": options, "answer_zi": character.zi},
        answer=character.zi,
        options=tuple(options),
        # 「部首」是本题的**比较对象**：题干不写它，读者不知道该按什么挑（实测 16z 的产出
        # 「请从下列选项中选出哪一个：㐀、…」）。写进 `must_contain` 才真的接上校验。
        must_contain=("部首",),
        stem_asks_any=_ODD_ASK_MARKERS,
        distractors=tuple(
            Distractor(text=c.zi, why_wrong=f"「{c.zi}」的部首确实是「{radical}」，所以它不是要找的那个")
            for c in picked
        ),
        proof=(f"选项里只有「{character.zi}」的部首不是「{radical}」",),
        provenance=make_provenance(character, "T10_odd_one_out", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


# ---------------------------------------------------------------- T11 组合排序


def build_stroke_extreme(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T11：给一组选项，问其中总笔画最多（或最少）的是哪个。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；凑不齐 3 个笔画互异且严格小于/大于本字的同伴时返回 None。
    预期：答案 = 本字；干扰项笔画两两互异且全部严格在同一侧，保证极值唯一；
          选项走 `options` 字段，并要求题干把候选**逐个列出来**——读者看得见候选才答得了
          （只写「下列选项」的产出是废题）；答案只是候选之一，不构成泄露，
          因此选择题不做 `no_answer_leak`，由 `options_ok` 的「每个候选只出现一次」兜住。
    """
    if not character.zi or character.stroke_total <= 0:
        return None
    direction = "most" if rng.random() < 0.5 else "fewest"
    if direction == "most":
        keys = [s for s in index.by_stroke if s < character.stroke_total]
    else:
        keys = [s for s in index.by_stroke if s > character.stroke_total]
    if len(keys) < 3:
        return None
    rng.shuffle(keys)
    others: list[Character] = []
    for stroke in keys[:3]:
        pool = [c for c in index.by_stroke[stroke] if c.zi and c.zi != character.zi]
        if not pool:
            return None
        others.append(pool[rng.randrange(len(pool))])
    options = [character.zi, *(c.zi for c in others)]
    rng.shuffle(options)
    comparator = "最多" if direction == "most" else "最少"
    return TaskSpec(
        task_id="T11_stroke_extreme",
        payload_facts=("strokes",),
        instruction=(
            # 候选写进说明（同 T10 的理由）：只写「候选里」小模型就不列候选，题干没有可选项。
            f"{'、'.join(options)} 里总笔画数{comparator}的那个字"
        ),
        facts={
            "direction": direction,
            "options": options,
            "answer_zi": character.zi,
            "strokes": {character.zi: character.stroke_total, **{c.zi: c.stroke_total for c in others}},
        },
        answer=character.zi,
        options=tuple(options),
        # 「笔画」是本题的**比较对象**：与 T10 同因——不声明就接不上 `subject_visible` 校验。
        must_contain=("笔画",),
        stem_asks_any=_EXTREME_ASK_MARKERS,
        distractors=tuple(
            Distractor(text=c.zi, why_wrong=f"「{c.zi}」是 {c.stroke_total} 画，不是选项里笔画{comparator}的")
            for c in others
        ),
        proof=(f"选项里「{character.zi}」是 {character.stroke_total} 画，笔画{comparator}",),
        provenance=make_provenance(character, "T11_stroke_extreme", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


# ---------------------------------------------------------------- T12 结构变换


@dataclass(frozen=True)
class _Edit:
    """输入：一次「换掉某个位置的单字部件」的结果。输出：不可变编辑记录。

    `ids` 是这次替换后**读者重建出来的那个 IDS 串**（= 命中的候选串），不是结果字的
    `ids_candidates[0]`——结果字可能有多个候选，候选[0] 未必是上一步渲染出的那串；
    下一步必须接着 `ids` 走，否则题干里的「旧部件」和读者重建出来的字对不上。
    """

    label: str
    old: str
    new: str
    result: Character
    ids: str


def _one_edit(source: Character, ids: str, index: CharacterIndex, rng: random.Random) -> _Edit | None:
    """输入：源字、**当前要拆的 IDS 串**、索引、随机源。输出：一次可验证的部件替换，找不到时 None。
    预期：只替换**顶层直接是单字**的部件（这样位置标签才精确）；
          索引给的候选必须与现场重算的 IDS 完全一致，双保险；
          返回的 `_Edit.ids` 是替换后读者会重建出来的那串（= 命中的候选串），供下一步接着拆。
    """
    if not ids:
        return None
    try:
        tree = parse_tree(ids)
    except IdsSyntaxError:
        return None
    if tree.is_leaf:
        return None
    labels = _position_labels(tree.op or "", len(tree.children))

    positions: list[tuple[int, int]] = []  # (顶层子下标, 该子节点对应的叶子下标)
    offset = 0
    for child_index, child in enumerate(tree.children):
        if child.is_leaf:
            positions.append((child_index, offset))
        offset += len(leaves(child))
    rng.shuffle(positions)

    for child_index, leaf_position in positions:
        old = _leaf_at(ids, leaf_position)
        if not old:
            continue
        mates = list(index.blank_mates(ids, leaf_position, source.zi))
        rng.shuffle(mates)
        for other, mate_position, candidate in mates:
            if not other.zi:
                continue
            new = _leaf_at(candidate, mate_position)
            if not new or new == old:
                continue
            if render(replace_leaf(tree, leaf_position, new)) != candidate:
                continue  # 索引与实算不一致就跳过，绝不产出对不上的编辑
            return _Edit(label=labels[child_index], old=old, new=new, result=other, ids=candidate)
    return None


def build_ids_transform(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T12：把某几个部件换掉，问得到的字是哪个。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；做不出 `difficulty.hops` 步可验证的替换、或答案会泄露时返回 None。
    预期：每一步都必须是**真实收录**的字（挖空索引保证）；中间结果不点名，只描述编辑动作。
    """
    if not character.zi or not character.ids_candidates:
        return None
    cursor = character
    current_ids = character.ids_candidates[0]
    described: list[str] = []
    forbidden = [character.zi]
    for _ in range(difficulty.hops):
        edit = _one_edit(cursor, current_ids, index, rng)
        if edit is None:
            return None
        described.append(f"把{edit.label}的部件「{edit.old}」换成「{edit.new}」")
        forbidden.extend([edit.old, edit.new])
        cursor = edit.result
        current_ids = edit.ids
    answer = cursor.zi
    if not answer or answer in forbidden:
        return None  # 答案与题干里出现过的字撞车 = 泄露

    return TaskSpec(
        task_id="T12_ids_transform",
        payload_facts=("start_zi", "edits"),
        instruction=(
            f"从「{character.zi}」出发按列出的替换步骤依次执行后得到的那个字"
        ),
        facts={
            "start_zi": character.zi,
            "edits": described,
            "steps": difficulty.hops,
            "answer_zi": answer,
        },
        answer=answer,
        visible_subject=character.zi,
        # 题干校验：答案是一个字，题干必须问「得到的字是哪个」（同 T1 的理由——校验空着等于不设防）
        stem_asks_any=_TRANSFORM_ASK_MARKERS,
        distractors=_pick(
            [
                list(index.same_radical(cursor)),
                list(index.by_stroke.get(cursor.stroke_total, ())),
                list(index.characters),
            ],
            [*forbidden, answer],
            rng,
            lambda c: f"「{c.zi}」不是按题干替换后得到的字",
        ),
        # 同上：proof 里不许出现「题干」
        proof=(f"按替换动作依次执行，得到的是「{answer}」",),
        provenance=make_provenance(character, "T12_ids_transform", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )
