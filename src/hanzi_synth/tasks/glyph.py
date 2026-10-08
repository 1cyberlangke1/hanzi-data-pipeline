"""字形类题目构造器：T1 笔画 / T3 IDS 正向 / T13 释义反向造字形。

答案全部是**字段直接值**：
- 笔画数 / 部首 ← 直接读字段；
- 字形写法 ← 直接读 `ids` 候选集合，多解全算对（「一个字当然有多种写法」）。

干扰项由 `distractors.char_distractors` 按 `Difficulty.distractor_depth` 取，
near/mid/far 三档真实生效（同部首近笔画 / 同笔画异部首 / 全池随机）。

构造器**不写自然语言**，只产出事实、答案、推理链、干扰项；措辞交给 L3。
"""

from __future__ import annotations

import random
import weakref
from collections import Counter

from ..facts.ids import IdsSyntaxError, Node, parse_tree
from ..facts.index import CharacterIndex
from ..facts.gloss import gloss_phrase, plain_senses
from ..facts.records import Character, clean_meaning_block
from ..facts.surface import IDC_LABELS, ids_surfaces, reading_surfaces
from .base import Difficulty, Distractor, TaskSpec, provenance as make_provenance
from .distractors import char_distractors, numeric_distractors
from .reading import extract_readings

CONSTRUCTOR_VERSION = "0.2.0"

# 问 IDS 拆写的题（T3 正向 / T13 反向）题干至少要命中其中一个说法：这是「问的必须是
# 答案那件事」的校验。答案是 IDS 拆解，题干写「请问这个字是什么？」就是在问那个字本身
# （实测 16j 漏网）；T3 还漏过「问这个字一共有几画」这种换了问题的产出。
# 「结构式 / 部件结构 / 部件分解」是实测补进来的：说明改成名词性短语后，小模型组句时写的是
# 「「㐀」拆到部件一级的结构式是什么？」，只认「拆解」会把同一件事的另一种说法判成「问错了方向」
# ——那是校验的假阴性，不是产出的毛病（探针 v11 里 5 条 T3 全栽在这一条）。
_ASK_MARKERS = ("拆解", "拆写", "结构拆解", "结构式", "部件结构", "部件分解", "表意文字描述序列")

# T1 的题干校验：答案是**笔画数**，题干至少要命中一个「问笔画」的说法。实测第十轮真机的
# T1 U0340A 题干写「请写出它的正确读音。这个字的ID是 ⿱士乙。请写出这个字的拼音是什么？
# 请写出这个字的部首与其他几个相同吗？」——问的是读音与部首，答案是 4 画，读者照着题干答
# 必然答错。原先 T1 没声明 `stem_asks_any`，`stem_asks` 那一关**整条跳过**（校验空着 =
# 不设防），这道废题因此发布了出去。代价：549 条已发布产出里 **0 条**被它没通过。
_STROKE_ASK_MARKERS = ("笔画", "几画", "多少画", "画数", "几笔", "多少笔", "笔数")

# 释义块截断长度：够 LLM 重述即可，过长只会抬高调用成本
_MEANING_MAX_CHARS = 220

# 释义清洗后至少要有这么多汉字才够当识别条件：「好」这种单字释义重述不出题。
_MIN_MEANING_CHARS = 6

# 反向题（T13）的识别条件 → 池里有几个字。题干就是「读作 X、释义是「Y」」拼出来的，所以
# **池里出现第二个字给出同一组识别条件** = 这道题有两个都算对的答案（实测全池 17798 个可出
# 反向题的字里 6619 个命中——「女子人名用字」「人名也」这种通用释义被好几个字共享）。
# 表按**索引对象**缓存（WeakKeyDictionary，索引回收即消失）：建表是 O(池大小)，而计划阶段每个字
# 都要调一次构造器，不缓存就是 O(池大小²)。
_IDENTIFICATION_CACHE: "weakref.WeakKeyDictionary[CharacterIndex, dict[tuple, int]]" = (
    weakref.WeakKeyDictionary()
)


def _identification_material(character: Character) -> tuple[tuple[str, ...], str, str] | None:
    """输入：字记录。输出：(读音展示写法元组, 识别条件短语, 完整清洗释义)；凑不出时 None。

    预期：读音与释义取自**同一个释义块**——读音是该块开头的标注，换块就对不上；释义用
          `facts.gloss.gloss_phrase` 洗成纯中文义项，`～` / 括注 / 注音 / 互见这类排版残片
          一个都进不来。实测旧写法（清洗后原文直接当题面）会把「鸡～」「【儒】的民国一简」
          原样写进题干，194 字的题面读者读不懂、模型也只能照抄——于是被 `no_source_copy`
          反复打回（语义池 rejects 里 52% 是这一条）。
          释义太短、释义里写着本字本身、洗不出义项、抽不到读音时返回 None：这些字不出反向题，
          不拿一道废题去占位。
    """
    for block in character.meaning_blocks():
        cleaned = clean_meaning_block(block)[:_MEANING_MAX_CHARS].strip()
        # 释义里出现本字本身也不能用：反向题要读者靠读音与释义认出是哪个字，
        # 题干里写着它就没题可做了（实测 㐂 的释义就是「【喜】的草体写成「㐂」三个七……」）。
        if len(cleaned) < _MIN_MEANING_CHARS or character.zi in cleaned:
            continue
        phrase = gloss_phrase(block)
        if not phrase or character.zi in phrase:
            continue
        # `facts.meaning_source` 给的是**整块清洗释义**（不是那两三字条件）：校验
        # `no_source_copy` 拿它比题干，比的是「有没有把字典原文整段搬过来」。给短条件的话
        # 这条校验就永远不可能触发，等于白留一道校验规则。
        full = "、".join(plain_senses(block))
        readings = extract_readings(block.strip()[:_MEANING_MAX_CHARS])
        display = tuple(
            forms[-1] if len(forms) > 1 else forms[0]
            for forms in (reading_surfaces(reading) for reading in readings)
            if forms
        )
        if display:
            return display, phrase, full
    return None


def _identification_key(character: Character) -> tuple | None:
    """输入：字记录。输出：反向题的识别条件 `(展示读音元组, 释义短语)`；凑不出时 None。

    预期：键与 `build_ids_reverse` 拼题干用的是**同一份材料**（都走 `_identification_material`），
          所以「键相同」等价于「题干相同」。凑不出材料的字本来也出不了反向题，不参与计数。
    """
    material = _identification_material(character)
    return None if material is None else (material[0], material[1])


def _identification_counts(index: CharacterIndex) -> dict[tuple, int]:
    """输入：索引。输出：{识别条件: 池里有几个字}。预期：按索引缓存，同一池只算一次。"""
    counts = _IDENTIFICATION_CACHE.get(index)
    if counts is None:
        counts = Counter(
            key for key in (_identification_key(character) for character in index.characters) if key
        )
        _IDENTIFICATION_CACHE[index] = counts
    return counts



def _shown_ids(character: Character, rng: random.Random) -> str:
    """输入：字记录与随机源。输出：本次展示用的 IDS surface 形式。
    预期：在 Unicode IDC 形式与描述式之间**成分布**地选，保证数据集层面两种写法都出现。
    """
    variants = ids_surfaces(character.ids_candidates[0])
    return variants[rng.randrange(len(variants))]


def display_text(character: Character) -> str:
    """输入：字记录。输出：可写进题干的字符；无码位字（Z/P 组）返回空串。
    预期：调用方据空串判定「该字无法显示，不能出看字形的题」。
    """
    return character.zi


def _ids_distractors(
    index: CharacterIndex,
    character: Character,
    rng: random.Random,
    difficulty: Difficulty,
    why_wrong,
) -> tuple[Distractor, ...]:
    """输入：索引、目标字、随机源、难度、错误说明模板。输出：IDS 文本干扰项。
    预期：禁用集合 = 目标字的全部 IDS 候选（含描述式），保证干扰项不会「其实也对」。
    """
    banned: list[str] = [character.zi, *character.ids_candidates]
    for candidate in character.ids_candidates:
        banned.extend(ids_surfaces(candidate))
    return char_distractors(
        index,
        character,
        banned,
        rng,
        depth=difficulty.distractor_depth,
        why_wrong=why_wrong,
        text_of=lambda other: other.ids_candidates[0],
    )


def build_stroke_total(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T1：问这个字一共有几画。
    输入：字记录 + 随机源 + 难度 + 索引（未用）。
    输出：TaskSpec；无码位或笔画为 0 时返回 None。
    预期：答案 = `stroke_total` 字段直接值，不做任何换算；干扰项按 distractor_depth 取邻近数值。
    """
    if not display_text(character) or character.stroke_total <= 0:
        return None
    answer = character.stroke_total
    return TaskSpec(
        task_id="T1_stroke_total",
        payload_facts=("zi", "stroke_total"),
        instruction=(
            f"「{character.zi}」的总笔画数"
        ),
        facts={
            "zi": character.zi,
            "radical": character.radical,
            "stroke_total": answer,
            "ids": list(character.ids_candidates),
        },
        answer=str(answer),
        visible_subject=character.zi,
        stem_asks_any=_STROKE_ASK_MARKERS,
        distractors=numeric_distractors(
            answer, rng, depth=difficulty.distractor_depth, why=lambda v: f"该字总笔画是 {answer}，不是 {v}"
        ),
        proof=(f"这个字的总笔画是 {answer} 画",),
        provenance=make_provenance(character, "T1_stroke_total", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def build_ids_forward(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T3：问这个字的 IDS 写法（正向）。
    输入：字记录 + 随机源 + 难度 + 索引（用于造干扰项）。
    输出：TaskSpec；无码位时返回 None。
    预期：答案集合 = `ids` 全部候选（多解全算对）；干扰项取其它字的 IDS，档位随 distractor_depth。
    """
    if not display_text(character) or not character.ids_candidates:
        return None
    distractors = _ids_distractors(
        index, character, rng, difficulty, lambda other: f"这是「{other.zi}」的拆解，不是本字"
    )
    canonical = character.ids_candidates[0]
    variants = ids_surfaces(canonical)
    shown = variants[rng.randrange(len(variants))]
    return TaskSpec(
        task_id="T3_ids_forward",
        payload_facts=("zi", "ids"),
        instruction=(
            f"「{character.zi}」拆到部件一级的结构式"
        ),
        facts={"zi": character.zi, "ids": list(character.ids_candidates), "stroke_total": character.stroke_total},
        answer=canonical,
        answer_aliases=character.ids_candidates[1:],
        surface_forms=variants[1:],
        visible_subject=character.zi,
        stem_asks_any=_ASK_MARKERS,
        distractors=distractors,
        proof=(f"这个字的结构拆解写作 {shown}",),
        provenance=make_provenance(character, "T3_ids_forward", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )


def _walk_operators(node: Node) -> list[str]:
    """输入：IDS 语法树节点。输出：这棵树里出现的全部 IDC 描述符（含嵌套）。
    预期：叶子返回空表；只收集描述符本身，调用方拿去查位置词表，顺序无关。
    """
    if node.is_leaf:
        return []
    found = [node.op or ""]
    for child in node.children:
        found.extend(_walk_operators(child))
    return found


def structure_words(character: Character) -> tuple[str, ...]:
    """输入：字记录。输出：它**全部** IDS 候选里真实出现过的位置关系词（去重保序）。
    预期：多解字（如 ⿸… 与 ⿱… 两个候选）要把两个候选的算子都算上——模型照候选[1] 描述时，
          不能因为「候选[0] 里没这个词」被误判成编造结构。解析不了的候选跳过，不因此丢字。
    """
    words: list[str] = []
    for candidate in character.ids_candidates:
        try:
            tree = parse_tree(candidate)
        except IdsSyntaxError:
            continue
        for operator in _walk_operators(tree):
            label = IDC_LABELS.get(operator)
            if label and label not in words:
                words.append(label)
    return tuple(words)


def build_ids_reverse(
    character: Character, *, rng: random.Random, difficulty: Difficulty, index: CharacterIndex
) -> TaskSpec | None:
    """T13：根据释义反向造字形（给意思，问怎么写）。
    输入：字记录 + 随机源 + 难度 + 索引。
    输出：TaskSpec；无码位或无释义时返回 None（题干既显示不了字、又没材料）。
    预期：facts 里给出**释义短语**（纯中文义项，见 `facts.gloss.gloss_phrase`）供 L3 受约束重述，
          释义里能抽出读音时一并给出读音——
          反向题按设计**不写出那个字**，读者要靠「读音 + 释义」认出是哪个字；
          实测只给释义碎片时裁判（正确地）判「未指定具体字、认不出是哪个字」，
          缺的就是识别条件。抽不出读音就不加这条要求，不因此砍掉这批字。
          答案仍是 `ids` 候选集合；题干必须重述释义（L4 校验与原文的连续字面重叠），不得照抄，
          也不得照搬互见条目「同【某字】」——那会把答案那个字直接写出来。
          **识别条件在池里必须唯一**：同一组「读音 + 释义」对应两个以上的字时，这道题有几个都算对的
          答案，读者答哪个都能被说成对，模型学到的就是噪声——这类字不出反向题（它们仍然出 T17）。
          另外声明 `structure_words`：实测小模型会在题干里写「字形结构属于左上包围，上部部件为
          𠂆……」把答案那棵树用散文说一遍，所以题干里一个位置词都不许出现；解析里只许出现
          答案真实用到的位置词（同一条产出写「按照左右结构与上下结构组合而成」，而答案里
          根本没有左右结构——那是编造的断言）。
    """
    if not display_text(character):
        return None
    material = _identification_material(character)
    if material is None:
        return None
    # 读音的**展示写法**取带调符号那一种：`yǐn` 是读者眼里的正常写法，`yin3` 只是程序记法。
    # 实测钉 ascii 时产出写成「读音为 yi1 和 yin3」——那是数据在说程序话，不是中文。
    # facts 里也只给这一种：给了 ascii 那份，小模型会挑它抄（挑哪份由它掷骰子，不如不给）。
    display, phrase, source = material
    reading_display = list(display)
    # 题干自足性校验：识别条件（读音 + 释义）在池里必须**唯一**。不唯一时同一个题干对应多个都算对的
    # 答案（实测全池 6619/17798 命中），那种题不是难题、是错题——读者答哪个都"对"，模型学不到东西。
    # 这些字仍然会出 T17（题干里写着那个字，没有歧义），只是不出反向题。
    if _identification_counts(index).get((tuple(reading_display), phrase), 0) > 1:
        return None

    distractors = _ids_distractors(
        index, character, rng, difficulty, lambda other: f"这是「{other.zi}」的写法，不符合题干释义"
    )
    shown = _shown_ids(character, rng)


    return TaskSpec(
        task_id="T13_ids_reverse",
        payload_facts=("reading_display", "meaning_source", "ids"),
        instruction=(
            # 说明一律中文：英文说明多一道翻译，小模型会译错——真机 T13 把 `reading` 译成「阅读」，
            # 产出写「请根据阅读和释义，拆解并写出该汉字的部件结构」，读者根本不知道要给的是读音。
            # 识别条件（读音 + 释义）**写进说明**：它们本来就是题干必须出现的串（读音在
            # `must_contain`，释义是认字条件），只写「靠读音与释义认出」小模型一个都不写——
            # 真机 2/18 栽在读音、3 条样本 0 条写出释义，题干只剩「这个字…」，读者无从作答。
            # 写进说明后：读音 3/3、释义 3/3、tier 全 A（probe_t13.jsonl）。
            f"读作 {'、'.join(reading_display)}、释义是「{phrase}」的那个字的部件结构式"
        ),
        facts={
            "meaning_source": source,
            "ids": list(character.ids_candidates),
            "radical": character.radical,
            "stroke_total": character.stroke_total,
            "reading_display": reading_display,
        },
        answer=character.ids_candidates[0],
        answer_aliases=character.ids_candidates[1:],
        surface_forms=ids_surfaces(character.ids_candidates[0])[1:],
        # 读音是反向题的识别条件，不是可选的写作要求：抽得出读音就必须写进题干（程序侧校验）
        must_contain=tuple(reading_display),
        hidden_subject=character.zi,
        stem_asks_any=_ASK_MARKERS,
        structure_words=structure_words(character),
        distractors=distractors,
        slots=("stem", "explanation"),
        proof=(f"这个字的结构拆解写作 {shown}",),
        provenance=make_provenance(character, "T13_ids_reverse", CONSTRUCTOR_VERSION, rng.randrange(1 << 30)),
    )
