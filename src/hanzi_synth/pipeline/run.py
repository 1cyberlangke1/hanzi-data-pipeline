"""管线编排：把 L0 事实底座 → L2 构造 → L3 措辞 → L4 验证 串成一次可重复的运行。

设计：
- 每个字先由 L2 产出 TaskSpec（答案与推理链此时已确定），再交给 L3 措辞；
- 只有 L4 判 `tier == 'A'` 的记录才写进产出文件；
- 失败的记录写进旁路日志（`*.rejects.jsonl`），保留失败原因，便于回炉；
- **不遍历全池出题**：每桶攒够 `bucket_cap` 条就收手。构造器要为干扰项查索引，
  7.9 万字 × 13 题型全量出题既慢又没必要，抽样出题在统计上等价；
- **并发只在 L3**：L2 与 L4 仍是单线程、按序执行，产出顺序与串行时逐条一致（可复现）。
  并发收益**取决于端点在哪**：远端网关是纯网络等待，`workers > 1` 接近线性；
  本机端点则是同一张显卡在吐字，多开只会互相抢显存带宽。同一条生产路径实测
  （`--limit 4 --exclude synthesis/out`）：1 路 96.4 s/请求、2 路 10.5 s/请求；
  而同一台机器另有东西吃显卡时，1 路能慢到 233 s/请求——波动来自显卡争用，不是并发度。
  所以本机端点按 `LOCAL_WORKER_CAP` 收口；
- **产出默认续跑（append + 去重）**：以 (task_id, ziid) 为唯一键跳过已有的，
  新的追加写。跑一半被中断、或分多次补量，都不会冲掉已有产物；
  要重来一份干净的就显式 `resume=False`（CLI `--overwrite`，覆盖前自动备份）。
- **不丢数据的五道闸门**：①产出路径与输入数据是同一个文件时直接拒绝，原始数据不可能被覆盖；
  ②续跑只追加不截断；③上轮被掐断留下的半行先补换行再追加，避免新记录粘在残行尾部
  把两行一起毁掉；④每条落盘即 flush + fsync；⑤同一产出文件加独占锁，两批并发不会互相踩。
- **缺席 ≠ 通过**：读者 / 裁判这类模型侧检验跑挂（请求失败 / 只吐思考段 / 输出不是 JSON）时先
  **重问同一个问题**（`CHECK_ATTEMPTS` 次）；重问完仍缺席就**不发布**，题进 rejects 回炉——
  没被检验过的题不能进数据集，但题也不会丢；
- **覆盖率旁路文件**：每次运行写 `<out>.coverage.json`，报「池子多大 / 覆盖了多少 /
  还差多少 / 按本轮实测速度还要多久」，进度不靠人工抽检。
"""

from __future__ import annotations

import json
import os
import hashlib
from collections import Counter
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..config import REPO_ROOT, ConfigError, ProviderConfig, default_data_path, load_providers
from ..facts.index import CharacterIndex
from ..facts.loader import DataError, iter_characters
from ..llm.judge import judge_record
from ..llm.provider import ChatClient
from ..llm.verbalizer import Verbalized, verbalize, verbalize_batch
from ..tasks.base import Difficulty, Distractor, TaskSpec
from ..tasks.registry import CONSTRUCTORS, TASK_STEM_ASKS, build_all
from ..llm.roundtrip import explanation_states_answer, roundtrip_answer, subject_identified_by_gates
from ..verify.checks import JUDGE_FEEDBACK_KEY, ROUNDTRIP_FEEDBACK_KEY, verify
from .dedup import load_jsonl
from .difficulty import difficulty_note

DEFAULT_OUT = REPO_ROOT / "synthesis" / "out" / "batch.jsonl"

# 产出唯一键：一个字在一个题型下只会产出一条，所以 (题型, 字) 就是主键
RecordKey = tuple[str, str]

# 产出记录里**不落**的字段：都是程序拼的中文（措辞素材 / 面向写作者的要求），
# 只给 L3 当素材，进了产出记录就等于数据集里混进「程序写的中文」（铁律 2）。
# 闸门字段不在其中——它们必须落盘，否则 `spec_from_record` 复验时会静默降级。
WORDING_ONLY_KEYS = ("proof", "distractors")


# 本机端点的并发上限。4060 Laptop 8GB：显存与显存带宽就是瓶颈，多开只会互相抢。
# 上限 4 是端点自身的能力边界——本机服务端最多同时处理 4 路请求，再开就是排队。
# 它是**封顶**不是「开满更快」：短输出探针实测 4 路并行墙钟 5.93 s，
# 而 4 路串行合计只要 4.79 s，多开在这台机器上没有正收益。
# 同一条生产路径（`--limit 4 --exclude synthesis/out`）实测：1 路 96.4 s/请求、2 路 10.5 s/请求。
LOCAL_WORKER_CAP = 4


def is_local_endpoint(base_url: str) -> bool:
    """输入：provider 的 base_url。输出：是否指向本机。
    预期：只认主机名——127.0.0.1 / localhost / ::1；端口、路径、协议都不参与判断。
          本机端点写成局域网 IP（192.168.x.x）时这里判不出来，那种情况由主人自己收并发。
    """
    host = urlparse(base_url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


def with_local_worker_cap(
    config: PipelineConfig, providers: Sequence[ProviderConfig]
) -> tuple[PipelineConfig, str]:
    """输入：运行配置 + 本轮实际会用的 provider。输出：(收口后的配置, 给用户看的说明或空串)。
    预期：只要有一个 provider 指向本机、且并发超过 LOCAL_WORKER_CAP，就收到上限并给出说明。
          这是**上限**不是定值——主人要开 1 路照样是 1 路；远端网关不受影响（靠并发压墙钟）。
          静默改配置会让人以为参数没生效，所以收口必须带一句人话说明。
    """
    if config.workers <= LOCAL_WORKER_CAP:
        return config, ""
    if not any(is_local_endpoint(p.base_url) for p in providers):
        return config, ""
    note = (
        f"本机端点：--workers {config.workers} 已收到上限 {LOCAL_WORKER_CAP}"
        f"（同一张显卡多开只会互相抢显存带宽，长输出实测无正收益）"
    )
    return replace(config, workers=LOCAL_WORKER_CAP), note


@dataclass(frozen=True)
class PipelineConfig:
    """输入：一次运行的全部参数。输出：不可变配置。
    预期：`offline=True` 时完全不联网，只跑 L0/L2（自检用）；
          `pool_size <= 0` 表示索引建在全池上（约束题的唯一性判定需要全池）；
          `resume=True` 表示续跑：跳过已有产出、追加写，绝不截断旧文件。
    """

    limit: int = 40
    task_ids: tuple[str, ...] | None = None
    out_path: Path = DEFAULT_OUT
    data_path: Path | None = None
    seed: int = 20261006
    temperature: float = 0.6
    offline: bool = False
    pool_size: int = 0
    max_scan: int = 20000
    hops: int = 1
    constraints: int = 1
    distractor_depth: str = "near"
    workers: int = 1
    retries: int = 1
    # 一次请求处理几道题。0 = 跟随 provider 配置里的 `batch`（secrets.toml），1 = 单题模式。
    # 默认 0 是为了让 secrets.toml 的 `batch` 真正生效——小模型批量会掉质量，
    # 所以这个值必须能在配置里改、也必须能过 A/B 质量对比，不能写死在代码里。
    batch: int = 0
    # L4.5 LLM 裁判：默认关。README §8.2 要求先用金标标定（κ ≥ 0.6）才采用，所以它是开关
    # 而不是默认行为；打开后，每道**过了程序防线**的题会多花一次请求。
    judge: bool = False
    # L4.6 往返检验：把（题干, 解析, 选项）交回模型答题，答案对不上载荷里定死的答案就否决重试。
    # 它比裁判便宜（只让模型答一句话）、判据是闭集，弱模型也够用。
    # 这里默认**关**：单元测试用假客户端、拿不到「读者」这一步的回复，默认开会让无关测试全炸；
    # **生产入口 CLI `run` 默认开**（`--no-roundtrip` 关）。所以这条开关必须按生产默认参数复验。
    roundtrip: bool = False
    resume: bool = True
    # 开新一轮时要把**已有产出**的主键也排除掉：池子是从头扫的，不排除就会把上一轮
    # 同一批字再采一遍。实测 597 条产出只覆盖 87 组 (题型, 字)——重复 508 条，覆盖没往前走。
    exclude_paths: tuple[Path, ...] = ()
    providers: Sequence[ProviderConfig] = field(default_factory=tuple)

    def difficulty(self) -> Difficulty:
        """输入：无。输出：由配置构造的 Difficulty。预期：非法值在这里就抛错，不带到出题阶段。"""
        return Difficulty(hops=self.hops, constraints=self.constraints, distractor_depth=self.distractor_depth)


def _spec_to_record(
    spec: TaskSpec, verbalized: Any, verification: Any, character_id: str, difficulty: Difficulty,
    judgement: Any = None, roundtrip: Any = None,
) -> dict:
    """输入：TaskSpec + L3 产出 + L4 结论 + 字 ID + 难度设定。输出：可落盘的产出记录。
    预期：产出里的自然语言只有 `stem` / `explanation` 两处，**都必须来自 LLM**；
          L2 拼出来的 `distractors[].why_wrong` 与 `proof` 一样只是给 L3 的措辞素材，
          不进产出记录——否则数据集里会混进「程序拼的中文」，与措辞必须走模型的铁律冲突。
          错项本身可由 `options` 减 `answer` 推出，不落盘也不丢信息。
          difficulty 是解析式先验（非实测），字段里如实标注。
          `judge` 如实落盘：None = 这次没开裁判，{"ok": false, "error": ...} = 裁判缺席（自己跑挂了），
          {"ok": true/false, ...} = 裁判的判定——留痕才能事后标定它。
          `roundtrip` 是**自然语言那一层**的判定（把题干交回模型，看它答不答得出定死的答案），
          同样如实落盘：机械闸门只会做子串 / 集合比对，判不了「这段话读者读得懂吗」，
          少了这一份，数据集对可读性就没有任何证据，事后也分不清「没查」与「查过了」。

          载荷直接复用 `_offline_record` 再剔掉 `WORDING_ONLY_KEYS`：**闸门字段必须落盘**。
          原先只落 `facts` 是不够的——`visible_subject` / `must_contain` / `stem_asks_any` /
          `structure_words` / `hidden_subject` / `instruction` 一个都不在，`spec_from_record`
          就会静默降级成空值，于是「题干有没有写出那个字」「题干问的是不是答案那件事」这些
          判据在事后复验里**全部失效**（实测：16q 的 T4 在复验里被清掉了主语却仍判 A）。
          一处定义两处用，也避免两条序列化路径各写一份而漂移。
    """
    record = {
        key: value
        for key, value in _offline_record(character_id, spec, difficulty).items()
        if key not in WORDING_ONLY_KEYS
    }
    record.update(
        {
            "stem": verbalized.stem,
            "explanation": verbalized.explanation,
            "provider": verbalized.provider,
            "model": verbalized.model,
            "verification": verification.as_dict(),
            # judgement 有两种形态：直连调用给 Judgement 对象，主流程 / 回炉从 history 取到的
            # 已是 as_dict() 后的 dict——在这里统一归一，避免两条路径各转一次转出 AttributeError
            "judge": judgement.as_dict() if hasattr(judgement, "as_dict") else judgement,
            "roundtrip": roundtrip.as_dict() if hasattr(roundtrip, "as_dict") else roundtrip,
        }
    )
    return record


def _offline_record(ziid: str, spec: TaskSpec, difficulty: Difficulty) -> dict:
    """输入：字 ID + TaskSpec + 难度设定。输出：不含自然语言的产出记录。
    预期：dry-run 自检用；同时是 rejects 里的回炉载荷，`spec_from_record` 与之互逆——
          没有这份载荷，rejects 就只剩「题干 + 失败原因」，回炉时无从重建题目。
          只收 ziid 而不是整个 Character，是因为回炉时手上没有字记录，只有载荷本身。
    """
    return {
        "task_id": spec.task_id,
        "ziid": ziid,
        "instruction": spec.instruction,
        "facts": dict(spec.facts),
        "answer": spec.answer,
        "accepted": list(spec.accepted),
        "options": list(spec.options),
        "visible_subject": spec.visible_subject,
        "must_contain": list(spec.must_contain),
        "hidden_subject": spec.hidden_subject,
        # 这两条闸门的判据也必须落盘：丢了它们，**回炉路径**会比实时路径松——
        # 实时拦下的「题干问错方向」「题干泄结构」在 replay 里就会直接发布出去。
        "stem_asks_any": list(spec.stem_asks_any),
        "structure_words": list(spec.structure_words),
        # 载荷白名单也要落盘：它是**措辞时发给模型的事实键**，丢了它回炉就会退回「整份 facts 全发」——
        # 实测 T17 因此把 `source`（康熙字典）又发了过去，回炉那一条写成「根据《康熙字典》记载，…」，
        # 而实时路径已经不发这个字段了。回炉路径不许比实时路径松（同 `stem_asks_any` 那条）。
        "payload_facts": list(spec.payload_facts),
        "proof": list(spec.proof),
        "distractors": [{"text": d.text, "why_wrong": d.why_wrong} for d in spec.distractors],
        "difficulty": difficulty_note(
            spec.task_id,
            difficulty,
            resolved_depth=spec.resolved_depth,
            resolved_hops=spec.resolved_hops,
            resolved_constraints=spec.resolved_constraints,
        ),
        "provenance": dict(spec.provenance),
    }


def spec_from_record(row: Mapping[str, Any]) -> TaskSpec:
    """输入：rejects 里存的 spec 载荷（`_offline_record` 的产物）。输出：可重新措辞的 TaskSpec。
    预期：与 `_offline_record` 严格互逆；缺字段用安全默认值，绝不因为一个字段缺失就崩掉整批回炉。
          `stem_asks_any` 为**空**时用该题型当前声明的闸门补上（`TASK_STEM_ASKS`）。
          理由：该字段是**生产时**落盘的，旧构造器根本没声明闸门的题型其历史记录就是空元组，
          而 `stem_asks` 那一关在空元组时整条跳过——于是复验会拿一个「没有闸门」的 spec
          去查，把「题干问读音、答案是笔画数」这种废题判成 tier A。这正是 `_offline_record`
          里「回炉路径不能比实时路径松」那句话的具体做法。该表是题型级常量，不随字变。
    """
    task_id = str(row.get("task_id", ""))
    answer = str(row.get("answer", ""))
    accepted = tuple(str(item) for item in row.get("accepted", ()) if item)
    stem_asks_any = tuple(str(item) for item in row.get("stem_asks_any", ()) if item)
    if not stem_asks_any:
        stem_asks_any = TASK_STEM_ASKS.get(task_id, ())
    return TaskSpec(
        task_id=task_id,
        instruction=str(row.get("instruction", "")),
        facts=dict(row.get("facts") or {}),
        answer=answer,
        answer_aliases=tuple(item for item in accepted if item != answer),
        options=tuple(str(item) for item in row.get("options", ())),
        visible_subject=str(row.get("visible_subject", "")),
        must_contain=tuple(str(item) for item in row.get("must_contain", ()) if item),
        hidden_subject=str(row.get("hidden_subject", "")),
        stem_asks_any=stem_asks_any,
        structure_words=tuple(str(item) for item in row.get("structure_words", ()) if item),
        payload_facts=tuple(str(item) for item in row.get("payload_facts", ()) if item),
        proof=tuple(str(item) for item in row.get("proof", ())),
        distractors=tuple(
            Distractor(text=str(item.get("text", "")), why_wrong=str(item.get("why_wrong", "")))
            for item in row.get("distractors", ())
            if isinstance(item, Mapping)
        ),
        provenance=dict(row.get("provenance") or {}),
    )


AVOID_TAIL = 6

# 模型侧检验（读者 / 裁判）的缺席重问次数。缺席 = 请求挂了 / 模型只吐思考段没吐 JSON /
# 输出不是合法 JSON——那是**工具没干成活**，不是「这条数据不合格」，所以同一个问题重问，
# 直到真的读出一个判定为止。重问完仍缺席就是**没过**（见 `_round_passed`）：
# 没被检验过的题不发布，进 rejects 回炉，题不丢。
CHECK_ATTEMPTS = 3

# 缺席在 rejects / attempt-stats 里单列一个桶：它不是「被哪条防线否决」，
# 而是「这条检验根本没跑成」——混进「往返检验否决」桶里会让人以为读者读了并说不行。
ROUNDTRIP_ABSENT_KEY = "读者缺席"
JUDGE_ABSENT_KEY = "裁判缺席"


def _judge_vetoed(judgement: Any) -> bool:
    """输入：history 里落盘的 judge 判定（dict 或 None）。输出：是否被裁判真正否决。
    预期：None / 非 dict = 没开裁判；只有 `ok` 为 False 且没有 error 才是真否决。
          `error` 非空 = 裁判自己跑挂了（缺席）——那**既不是否决也不是通过**：调用方先重问
          （`_judge_with_retries`），重问完仍缺席就不发布（`_round_passed`），题进 rejects 回炉。
    """
    if not isinstance(judgement, dict):
        return False
    return judgement.get("ok") is False and not judgement.get("error")


def _roundtrip_vetoed(roundtrip: Any) -> bool:
    """输入：history 里落盘的往返检验结果（dict 或 None）。输出：是否被往返检验真正否决。

    预期：只有「跑成了、且答出的答案与载荷里的定死答案对不上」才否决。
          `error` 非空是**缺席**（请求挂了 / 只吐思考段 / 输出不是 JSON）——那既不是否决也不是
          通过：调用方先重问（`_roundtrip_with_retries`），重问完仍缺席就不发布（`_round_passed`）。
    """
    if not isinstance(roundtrip, dict):
        return False
    return roundtrip.get("ok") is False and not roundtrip.get("error")


def _roundtrip_absent(roundtrip: Any) -> bool:
    """输入：history 里落盘的往返检验结果（dict 或 None）。输出：这次检验是不是**没跑成**。

    预期：`None` = 这一轮根本没要求做这条检验（tier 不是 A，或 `--no-roundtrip`），不算缺席；
          dict 且 `error` 非空 = 读者没读成。缺席**不是通过**——`_round_passed` 按没过处理。
    """
    if not isinstance(roundtrip, dict):
        return False
    return bool(roundtrip.get("error"))


def _judge_absent(judgement: Any) -> bool:
    """输入：history 里落盘的 judge 判定（dict 或 None）。输出：这次裁判是不是**没跑成**。

    预期：与 `_roundtrip_absent` 同一条规矩——`None` 是没开裁判，dict 且 `error` 非空是缺席。
    """
    if not isinstance(judgement, dict):
        return False
    return bool(judgement.get("error"))


def _roundtrip_with_retries(
    client: ChatClient,
    spec: TaskSpec,
    verbalized: Verbalized,
    *,
    answer_judge: ChatClient | None = None,
    subject_identified: bool | None = None,
    attempts: int = CHECK_ATTEMPTS,
) -> Any:
    """输入：读者客户端 + TaskSpec + L3 产出 + 仲裁客户端 + 识别条件旁证 + 重问次数。输出：RoundTrip。

    预期：**只对缺席重问**，重问的是同一道题、同一段文字——读者没读成是工具问题，不是这段文字
          有问题，重新措辞只会白花配额。读者读出了判定（ok True/False）立刻返回；
          重问次数用尽仍缺席时把最后一次（带 error）原样返回，由 `_round_passed` 判成没过。
    """
    result = roundtrip_answer(
        client, spec, verbalized, answer_judge=answer_judge, subject_identified=subject_identified,
    )
    for _ in range(max(1, attempts) - 1):
        if not result.absent:
            return result
        result = roundtrip_answer(
            client, spec, verbalized, answer_judge=answer_judge, subject_identified=subject_identified,
        )
    return result


def _judge_with_retries(
    judge_client: ChatClient, spec: TaskSpec, verbalized: Verbalized, *, attempts: int = CHECK_ATTEMPTS,
) -> Any:
    """输入：裁判客户端 + TaskSpec + L3 产出 + 重问次数。输出：Judgement。

    预期：与 `_roundtrip_with_retries` 同一条规矩——裁判缺席（网络错 / 输出不合法）就重问同一道题，
          只有真的判出一个结论（放行 / 否决）才停；重问完仍缺席就不发布（见 `_round_passed`）。
    """
    result = judge_record(judge_client, spec, verbalized)
    for _ in range(max(1, attempts) - 1):
        if not result.absent:
            return result
        result = judge_record(judge_client, spec, verbalized)
    return result


def _round_passed(tier: Any, judgement: Any, roundtrip: Any = None) -> bool:
    """输入：程序防线 tier + 裁判判定 + 读者检验结果。输出：这一轮算不算通过。
    预期：**唯一的通过判据**——发布（`_publishable`）与事后汇总（`summarize_attempts`）共用，
          否则会出现「线上没发布、attempt-stats 却记成一次过」这种两套口径。
          缺席（裁判 / 读者没跑成）**既不是否决也不是通过**：没被检验过的题不能发布，
          所以缺席按没过算——它进 rejects 回炉，题不丢，也不会带着「没读过的题」进数据集。
    """
    if tier != "A":
        return False
    if _judge_vetoed(judgement) or _roundtrip_vetoed(roundtrip):
        return False
    return not _judge_absent(judgement) and not _roundtrip_absent(roundtrip)


def _publishable(verification: Any, judgement: Any, roundtrip: Any = None) -> bool:
    """输入：程序防线结论 + 裁判判定。输出：这道题能不能发布。
    预期：**唯一的发布闸门**——主流程与回炉共用，避免两处各写一份判据而漂移。
          `tier == 'A'` 只管程序防线；开了裁判时还要裁判没否决（README §8.2 第 7 道防线）。
    """
    return _round_passed(verification.tier, judgement, roundtrip)


def _roundtrip_reason(roundtrip: Any) -> str:
    """输入：落盘的读者检验结果（dict）。输出：一条**中文**失败原因（进 rejects / attempt-stats）。

    预期：把「读者到底读出了什么」写清楚——只记一句「往返检验否决」的话，rejects 里看不出
          这道题是被四条判据里的哪一条拦下的。给措辞模型的那份见 `_roundtrip_feedback`。
    """
    note = str(roundtrip.get("note") or "")
    if roundtrip.get("is_question") is False:
        return f"{ROUNDTRIP_FEEDBACK_KEY}: 题干没有向读者提问（{note or '读者读不出作答要求'}）"
    if roundtrip.get("identifies_subject") is False:
        return f"{ROUNDTRIP_FEEDBACK_KEY}: 题干没说清问的是哪个字（{note or '读者认不出题目针对哪个字'}）"
    phrases = roundtrip.get("writer_phrases") or []
    if phrases:
        return f"{ROUNDTRIP_FEEDBACK_KEY}: 题干或解析里夹着写给写题人的话 {list(phrases)}"
    answer = str(roundtrip.get("answer") or "（答不出）")
    return f"{ROUNDTRIP_FEEDBACK_KEY}: 只凭题干答出的是「{answer}」，与标准答案不符"


def _roundtrip_feedback(spec: TaskSpec, roundtrip: Any) -> str:
    """输入：TaskSpec + 这一轮的读者检验结论（RoundTrip）。输出：回灌给措辞模型的反馈。

    预期：**中文**，且**不带闸门名**——回灌的是「读者读出了什么」，闸门名（「往返检验否决」）是
          给 rejects 看的行话，塞进模型上下文只会被抄进产出。给人看的那份在 `_roundtrip_reason`
          （带闸门名、进 rejects），两份用途不同、各写各的。反馈刻意用闸门会拦的词（题干 / 读者）
          ——模型把反馈抄进产出即被 `no_internal_terms` 拦下。
    """
    if not roundtrip.is_question:
        return (
            f"读者在题干里读不出问题（{roundtrip.note or '看不出要读者答什么'}）。"
            "题干要向读者提问——写成一句问话，把作答要求交给读者，不要写成陈述句，"
            "也不要把出题说明复述一遍。"
        )
    if not roundtrip.identifies_subject:
        return (
            f"读者看不出题干问的是哪个字（{roundtrip.note or '那个字根本没在题干里出现'}）。"
            "把那个字本身（或整份候选名单）写进题干。"
        )
    if roundtrip.writer_phrases:
        return (
            f"题干里夹着写给写题人的话 {list(roundtrip.writer_phrases)}。删掉——"
            "题干里只留问题本身。"
        )
    return (
        f"读者只凭题干答出的是「{roundtrip.answer or '什么也没答出'}」，"
        f"而这道题的答案是「{spec.answer}」。把读者答出这个答案需要的条件补进题干。"
    )


def _round_reasons(tier: Any, failures: Any, judgement: Any, roundtrip: Any = None) -> list[str]:
    """输入：程序防线 tier + 该轮的 failures + 裁判判定。输出：该轮的失败原因列表。
    预期：程序防线有自己的 failures 就用它；「程序全过但裁判否决」时 failures 是空的，
          这时补一条 `裁判否决: 理由`——否则 rejects 行与 attempt-stats 里都看不出这条
          为什么被拒（实测首版就是这样：failures 为空、理由只躺在 judge 字段里）。
    """
    reasons = list(failures or [])
    if not reasons and tier == "A" and _roundtrip_vetoed(roundtrip):
        reasons.append(_roundtrip_reason(roundtrip or {}))
    if not reasons and tier == "A" and _judge_vetoed(judgement):
        reason = str((judgement or {}).get("reason") or "未给理由")
        reasons.append(f"{JUDGE_FEEDBACK_KEY}: {reason}")
    if not reasons and tier == "A" and _roundtrip_absent(roundtrip):
        reasons.append(f"{ROUNDTRIP_ABSENT_KEY}: 读者没读成（{roundtrip.get('error')}）")
    if not reasons and tier == "A" and _judge_absent(judgement):
        reasons.append(f"{JUDGE_ABSENT_KEY}: 裁判没跑成（{judgement.get('error')}）")
    return reasons


def _better_round(history: list[dict], best: int, index: int) -> int:
    """输入：逐轮判定记录 + 当前最好一轮的下标 + 新一轮的下标。输出：新的最好轮下标。

    预期：只比**判定**，不比文本——tier A 优先，其次「往返检验 / 裁判没有否决」，最后 failures
          少的优先；并列保留**更早**的一轮（先到的更稳）。重试用尽时返回这一轮，否则第二轮退化
          （实测 T10 / T5_variant 第二轮整段回显载荷或输出散文）会把第一轮的合格产出顶掉。
    """
    def score(at: int) -> tuple[int, int, int]:
        row = history[at]
        # 「缺席」与「否决」在这条排序里同权：两者都意味着这一轮**没通过**，没通过的轮次
        # 不该被当成最好的一轮——否则缺席的那轮会被选中，发布判据再从它身上取 roundtrip。
        blocked = not _round_passed(row.get("tier"), row.get("judge"), row.get("roundtrip"))
        return (
            1 if row.get("tier") == "A" else 0,
            0 if blocked else 1,
            -len(list(row.get("failures") or [])),
        )

    return index if score(index) > score(best) else best


def _chosen_row(history: list[dict]) -> dict:
    """输入：逐轮判定记录。输出：**被选中那一轮**的记录（空 history 返回空字典）。

    预期：`_verbalize_with_retries` 在重试用尽时返回最好的一轮，并在那一轮打上 `chosen`；
          发布判据必须取那一轮，不能取 `history[-1]`——否则「最后一轮退化」会把好轮次的判定
          记到数据上。没打标记时退回最后一行（兼容旧记录与手工构造的 history）。
    """
    for row in reversed(history):
        if row.get("chosen"):
            return row
    return history[-1] if history else {}


def _verified_with_semantic_answer(
    verdict: Any, spec: TaskSpec, verbalized: Verbalized, answer_judge: ChatClient | None,
) -> Any:
    """输入：机械闸门结论 + TaskSpec + L3 产出 + 答案仲裁客户端。输出：可能被修正过的 Verification。

    预期：`answer_in_explanation` 是**子串比对**——它判得了「解析里有没有那串字」，判不了「解析说的
          是不是那个答案」（「这个字的读音不是 xié」「除了 qiu1 之外都对」都带着答案串，却把答案
          说反了）。所以它只做**快速通过**：过了不花配额；没过才让 LLM 读一遍解析，判「解析陈述出来
          的答案是不是标准答案」，判过就把那条失败撤掉（痕迹留在 `semantic_answer`）。
          仲裁缺席（跑挂了 / 输出不合法）维持机械结论——工具问题不判数据死刑，也不放行。
          failures 与 notes 是同一批失败的两份说法，撤一条要两边一起撤。
    """
    if verdict.answer_in_explanation or answer_judge is None:
        return verdict
    if any(not item.startswith("answer_in_explanation") for item in verdict.failures):
        # 别的闸门也挂了：这一条救回来照样发布不了，不花这次配额。
        return verdict
    stated, reason, error = explanation_states_answer(answer_judge, spec, verbalized)
    if stated is None:
        return replace(verdict, semantic_answer=f"仲裁缺席（{error}）")
    if not stated:
        return replace(verdict, semantic_answer=f"LLM 仲裁：{reason or '解析没说清答案'}")
    kept = [
        (failure, note)
        for failure, note in zip(verdict.failures, verdict.notes)
        if not failure.startswith("answer_in_explanation")
    ]
    return replace(
        verdict,
        answer_in_explanation=True,
        semantic_answer=f"LLM 仲裁：{reason or '解析说的就是标准答案'}",
        failures=tuple(failure for failure, _ in kept),
        notes=tuple(note for _, note in kept),
    )


def _verbalize_with_retries(
    client: ChatClient, spec: TaskSpec, *, retries: int, temperature: float,
    avoid: Sequence[str] = (), judge_client: ChatClient | None = None, do_roundtrip: bool = False,
    answer_judge: ChatClient | None = None,
) -> tuple[Verbalized, Any, int, list[dict]]:
    """输入：客户端 + TaskSpec + 重试次数 + 温度 + 已用过的开头 + 可选的裁判客户端。
    输出：(L3 产出, L4 结论, 实际尝试次数, 逐轮判定记录)。

    预期：一次就过就立即返回；不过就**回灌上一轮的拒绝原因**再试（最多 retries 次）。
          多样性不靠语体表，靠三件事：温度采样、把**已用过的题干开头**当负例喂回去（`avoid`）、
          以及 L6 去重闸门。回灌原因（reflexion）是为了不在同一处再摔一次。
          次数用尽仍不过，返回最后一次结果由调用方判进 rejects。
          主流程与回炉（`replay_rejects`）共用这一份实现，避免两条路径的重试语义漂移。
          每轮的 tier / 失败原因都记进 history 返回——只留一个 `retried` 计数的话，
          「首轮为什么被拒」就只活在内存里，跑完即失。
          `judge_client` 非空时**只在程序防线全过后**才叫裁判（README §8.2 第 7 道防线）：
          裁判缺席（自己跑挂了）不算否决；裁判否决则把它的理由回灌给下一轮。
          注意：裁判否决**不改** `verification.tier`（那是程序防线的结论），发布与否统一由
          `_publishable` 判——否则「程序全过 + 裁判否决」会被当成 tier A 发出去。
          `answer_judge` 是**答案仲裁**客户端，与裁判是两件事：往返检验里读者答出的是一句自然
          语言，子串比对只是快速通道，对不上时用它判「是不是同一个答案」（传 None 就只有快速通道）。
    """
    feedback: tuple[str, ...] = ()
    results: list[tuple[Verbalized, Any]] = []
    history: list[dict] = []
    best = 0
    for round_no in range(retries + 1):
        verbalized = verbalize(client, spec, temperature=temperature, feedback=feedback, avoid=avoid)
        # 机械闸门 + 解析的**语义核验**：`answer_in_explanation` 是子串比对，过不了才问 LLM。
        verdict = _verified_with_semantic_answer(verify(spec, verbalized), spec, verbalized, answer_judge)
        judgement = None
        if judge_client is not None and verdict.tier == "A":
            judgement = _judge_with_retries(judge_client, spec, verbalized)
        # L4.6 往返检验：把（题干, 解析, 选项）交回模型答题，答案对不上载荷里定死的答案就否决。
        # 它只判**闭集**（答案由代码定死），所以弱模型也够用；它测的是「读者能不能读懂并答对」，
        # 那正是枚举措辞形状永远追不完的那件事。
        roundtrip = None
        if do_roundtrip and verdict.tier == "A":
            roundtrip = _roundtrip_with_retries(
                client, spec, verbalized, answer_judge=answer_judge,
                subject_identified=subject_identified_by_gates(verdict),
            )
        judge_dict = judgement.as_dict() if judgement is not None else None
        roundtrip_dict = roundtrip.as_dict() if roundtrip is not None else None
        results.append((verbalized, verdict))
        history.append(
            {
                "round": round_no + 1,
                "tier": verdict.tier,
                "failures": list(verdict.failures),
                "error": verbalized.error or "",
                "judge": judge_dict,
                "roundtrip": roundtrip_dict,
            }
        )
        best = _better_round(history, best, round_no)
        if _round_passed(verdict.tier, judge_dict, roundtrip_dict):
            history[round_no]["chosen"] = True
            return verbalized, verdict, round_no + 1, history
        if verdict.failures:
            # 回灌 `notes`（模型侧说法）而不是 `failures`（给人看的行话）：两份同序同长，`notes`
            # 只讲「要改什么」，且刻意用 `no_internal_terms` 会拦的词——模型抄进产出即被拦下。
            # `notes` 缺席（手工构造的 Verification）时才退回 `failures`。
            feedback = tuple(verdict.notes) or tuple(verdict.failures)
        elif _roundtrip_vetoed(roundtrip_dict):
            feedback = (_roundtrip_feedback(spec, roundtrip),)
        elif judgement is not None:
            # 同样不带闸门名：模型只需要「哪里不对」，闸门名是给 rejects 看的行话。
            feedback = (judgement.reason or "裁判说这道题不合格，但没给理由。按上面的要求重写一遍。",)
        else:
            feedback = ()
    history[best]["chosen"] = True
    return results[best][0], results[best][1], retries + 1, history


def _verbalize_batch_with_retries(
    client: ChatClient, specs: Sequence[TaskSpec], *, retries: int, temperature: float,
    do_roundtrip: bool = False, answer_judge: ChatClient | None = None,
    ) -> list[tuple[Verbalized, Any, int, list[dict]]]:
    """输入：客户端 + 一批 TaskSpec + 重试次数 + 温度 + 要不要跑往返检验。
    输出：与 `specs` 等长、顺序一致的 (L3 产出, L4 结论, 实际轮数, 逐轮判定记录) 列表。
    预期：整批一次请求；**只把没过的那几道放进下一轮**（过了的不再重复花配额），
          并把按题分组的失败原因回灌（reflexion）。次数用尽仍不过的题返回最后一次结果。
          与单题路径共用 `verify`，两条路径的判据完全一致；清洗同样在这里做，
          否则批量模式会变成「另一套标准」。
          往返检验也在这里接上：**批量省的是措辞请求，省不掉这道检验**——它按题独立
          （每道题要各自被读一次，合不进一次批量请求），所以是每道过了程序防线的题
          各花一次请求。少接这一条的话，`batch > 1` 就成了「往返检验静默失效」的旁路。
          `answer_judge` 与单题路径同一件事：读者答出的自然语言与定死答案对不上时，
          交给 LLM 判「是不是同一个答案」；`subject_identified` 旁证同样在这里接上。
    """
    rounds = [0] * len(specs)
    results: list[list[tuple[Verbalized, Any]]] = [[] for _ in specs]
    histories: list[list[dict]] = [[] for _ in specs]
    best = [0] * len(specs)
    pending = list(range(len(specs)))
    feedback: dict[str, Sequence[str]] = {}
    for round_no in range(retries + 1):
        if not pending:
            break
        verbalized = verbalize_batch(
            client, [specs[index] for index in pending],
            temperature=temperature, feedback=feedback,
        )
        next_pending: list[int] = []
        feedback = {}
        for position, index in enumerate(pending):
            item = verbalized[position]
            verdict = _verified_with_semantic_answer(verify(specs[index], item), specs[index], item, answer_judge)
            roundtrip = (
                _roundtrip_with_retries(
                    client, specs[index], item, answer_judge=answer_judge,
                    subject_identified=subject_identified_by_gates(verdict),
                )
                if do_roundtrip and verdict.tier == "A"
                else None
            )
            roundtrip_dict = roundtrip.as_dict() if roundtrip is not None else None
            rounds[index] = round_no + 1
            results[index].append((item, verdict))
            histories[index].append(
                {
                    "round": round_no + 1,
                    "tier": verdict.tier,
                    "failures": list(verdict.failures),
                    "error": item.error or "",
                    "roundtrip": roundtrip_dict,
                }
            )
            best[index] = _better_round(histories[index], best[index], len(histories[index]) - 1)
            if not _round_passed(verdict.tier, None, roundtrip_dict):
                next_pending.append(index)
                feedback[f"第 {index + 1} 题（{specs[index].task_id}）"] = tuple(
                    verdict.notes or _round_reasons(verdict.tier, verdict.failures, None, roundtrip_dict)
                )
        pending = next_pending

    out: list[tuple[Verbalized, Any, int, list[dict]]] = []
    for index, entries in enumerate(results):
        if not entries:  # 只可能在 specs 为空时发生，真出现就当场炸，不静默补空
            raise RuntimeError(f"批量措辞没有覆盖第 {index + 1} 题")
        histories[index][best[index]]["chosen"] = True
        entry = entries[best[index]]
        out.append((entry[0], entry[1], rounds[index], histories[index]))
    return out


def load_produced_keys(path: Path) -> set[RecordKey]:
    """输入：已有产出文件路径。输出：里面出现过的 (题型, 字) 主键集合。
    预期：文件不存在或空行/坏行都跳过——续跑不该因为一行坏数据整批失败。
    """
    keys: set[RecordKey] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id, ziid = row.get("task_id"), row.get("ziid")
            if task_id and ziid:
                keys.add((str(task_id), str(ziid)))
    return keys


class PipelineBusy(RuntimeError):
    """输入：无。输出：可读异常。预期：同一产出文件已有活进程在写时抛出，避免两批互相踩。"""


def _pid_alive(pid: int) -> bool:
    """输入：进程号。输出：该进程是否仍活着。
    预期：Windows 上**绝不能**用 `os.kill(pid, 0)` 探测——CPython 在 Windows 下把它翻成
          TerminateProcess，等于把对方杀掉；这里改用 OpenProcess(SYNCHRONIZE) 探测。
          POSIX 下 `os.kill(pid, 0)` 是安全的零号信号，照常用。
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED：打不开 => 它活着
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_holder(lock_path: Path) -> int:
    """输入：锁文件路径。输出：持有者 pid（读不出来返回 0）。预期：坏锁文件按「无主」处理。"""
    try:
        return int(lock_path.read_text(encoding="utf-8").splitlines()[0].strip())
    except (OSError, IndexError, ValueError):
        return 0


def acquire_output_lock(out_path: Path) -> Path:
    """输入：产出文件路径。输出：已独占的锁文件路径。
    预期：用 O_EXCL 原子建锁；锁被活进程持有则抛 PipelineBusy；持有者已死（上次崩了）就接管，
          否则一次崩溃会让后面所有运行都跑不起来。
    """
    lock_path = out_path.with_name(out_path.name + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = _lock_holder(lock_path)
            if holder and _pid_alive(holder):
                raise PipelineBusy(
                    f"产出文件正被进程 {holder} 写入（锁：{lock_path}）；等它跑完，或确认它已死后删掉锁文件"
                )
            lock_path.unlink(missing_ok=True)  # 残留锁：接管
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return lock_path


def release_output_lock(lock_path: Path) -> None:
    """输入：锁文件路径。输出：无。预期：尽力删除；删不掉也不该让整个运行以异常收场。"""
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


class OutputLock:
    """输入：产出路径。输出：上下文管理器。预期：进出自动加/放锁，异常路径也不留锁。"""

    def __init__(self, out_path: Path) -> None:
        self._out_path = out_path
        self._lock: Path | None = None

    def __enter__(self) -> Path:
        self._lock = acquire_output_lock(self._out_path)
        return self._lock

    def __exit__(self, *_exc: object) -> None:
        if self._lock is not None:
            release_output_lock(self._lock)


def ensure_trailing_newline(path: Path) -> bool:
    """输入：产出文件路径。输出：是否补了换行。
    预期：上轮被 Ctrl-C 掐断时最后一行可能只写了一半，直接 append 会把新记录粘在残行尾部，
          **同时毁掉残行和新行**；这里先补一个换行把它们隔开，残行留给后续排查。
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return False
    with path.open("ab") as handle:
        handle.write(b"\n")
    return True


def backup_output(path: Path) -> Path | None:
    """输入：产出文件路径。输出：备份路径；文件不存在或为空时返回 None。
    预期：显式覆盖（resume=False）前先复制一份 `<名>.bak-<时间戳>`；用复制而不是改名，
          这样万一在打开新文件前崩掉，原文件还在原地。
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.bak-{stamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}.bak-{stamp}-{suffix}")
        suffix += 1
    shutil.copy2(path, target)
    return target


def _count_by_task(path: Path) -> dict[str, int]:
    """输入：产出文件路径。输出：{题型: 记录数}，按**文件全文**统计。
    预期：空行 / 坏行跳过，文件不存在返回空字典——旁路报告要的是累计进度，不是本轮增量。
    """
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = row.get("task_id")
            if task_id:
                counts[str(task_id)] = counts.get(str(task_id), 0) + 1
    return dict(sorted(counts.items()))


def _pool_groups(characters: Sequence[Any]) -> tuple[dict[str, int], int]:
    """输入：进池的字序列。输出：(分池计数 {U/Z/P}, 无码位字个数)。
    预期：P/Z 组无码位字（`zi` 为空）只能出「不需要写出该字」的题型，
          把这两个数写进覆盖率旁路文件，好判断覆盖进度是不是卡在这部分长尾上。
    """
    groups: dict[str, int] = {}
    codepointless = 0
    for character in characters:
        groups[character.group] = groups.get(character.group, 0) + 1
        if not character.has_codepoint:
            codepointless += 1
    return dict(sorted(groups.items())), codepointless


def coverage_report(out_path: Path, data_path: Path, *, limit: int | None = None) -> dict:
    """输入：产出文件、数据文件、池上限（None = 全池）。输出：覆盖率统计。
    预期：**只读**，绝不改动产出文件；产出文件不存在时按「0 覆盖」报，而不是报错。
    """
    characters = list(iter_characters(path=data_path, limit=limit))
    pool = {character.ziid for character in characters}
    keys = load_produced_keys(out_path)
    covered = {ziid for _task_id, ziid in keys}
    hit = pool & covered
    groups, codepointless = _pool_groups(characters)
    return {
        "out_path": str(out_path),
        "pool": len(pool),
        "covered": len(hit),
        "ratio": round(len(hit) / len(pool), 6) if pool else 0.0,
        "remaining": len(pool) - len(hit),
        "pool_by_group": groups,
        "codepointless": codepointless,
        "records": len(keys),
        "outside_pool": len(covered - pool),
        "by_task": _count_by_task(out_path),
    }


def write_coverage(out_path: Path, stats: dict) -> Path:
    """输入：产出路径与运行统计。输出：覆盖率旁路文件路径。
    预期：把「池子多大 / 覆盖多少 / 还差多少 / 按本轮实测速度还要多久」写成 JSON，省得翻日志猜进度；
          速度只在真的调过模型时才有意义，离线跑不写 ETA；
          `by_task` / `records` 按文件全文统计（累计进度），本轮增量单独放在 `this_run` 里，两者不混。
    """
    coverage = dict(stats.get("coverage", {}))
    per_record = stats.get("seconds_per_record")
    if per_record is not None:  # 0.0 也是有效实测值（假客户端跑得飞快），别被真值判断吞掉
        coverage["measured_seconds_per_record"] = round(per_record, 2)
        hours = coverage.get("remaining", 0) * per_record / 3600
        coverage["eta_hours_serial"] = round(hours, 2)
        coverage["eta_hours_with_workers"] = round(hours / max(1, stats.get("workers", 1)), 2)
    coverage["by_task"] = _count_by_task(out_path)
    coverage["records"] = sum(coverage["by_task"].values())
    coverage["this_run"] = {
        "ok": stats.get("ok", 0),
        "rejected": stats.get("rejected", 0),
        "retried": stats.get("retried", 0),
        "first_pass": stats.get("first_pass", 0),
        "requests": stats.get("requests", 0),
        "resumed": stats.get("resumed", 0),
    }
    target = out_path.with_name(out_path.name + ".coverage.json")
    target.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _plan(
    config: PipelineConfig,
    index: CharacterIndex,
    rng: random.Random,
    exclude: Iterable[RecordKey] = (),
) -> tuple[list[tuple[Any, TaskSpec]], int, int]:
    """输入：配置、索引、随机源、要跳过的已有主键。
    输出：(轮转排好序的出题计划, 真正构造过题的字数, 因覆盖满而跳过的字数)。
    预期：每个题型最多攒 `bucket_cap` 条候选就停止扫描——够轮转取样即可，不做全池穷举；
          已在 `exclude` 里的 (题型, 字) 不再进桶，保证续跑产出的是**新增**记录。
          `max_scan` 只数**真正构造过题的字**：一个字能出的题型数 = `len(wanted)`，
          达到这个数就说明它已覆盖满，直接跳过、不占用扫描预算。
          否则续跑时前 N 个字一旦全被覆盖，就会「扫完 20000 个字、一条也产不出」——
          用户看到的是静默空跑，这是最难排查的一类假成功。
          轮转取样时**优先挑还没用过的字**：同一个字能出多个题型，若每个题型都取桶里第一条，
          小批量就会全落在池子最前面那几个字上（实测 `--limit 16` 只覆盖 4 个字）。
          优先取未用字后，题型覆盖与字覆盖同时铺开；为让位而跳过的候选**不丢**——它们进 `deferred`，
          第一轮轮转结束后按桶序补齐。桶里每一条都是一个真实可出题的 (题型, 字) 组合，
          静默丢掉就是数据集缺一块（实测 T17 因此少 10,314 条）。
    """
    cap = max(4, config.limit)
    skip = set(exclude)
    wanted = set(config.task_ids) if config.task_ids else set(CONSTRUCTORS)
    produced_by_ziid: dict[str, int] = {}
    for _task_id, ziid in skip:
        produced_by_ziid[ziid] = produced_by_ziid.get(ziid, 0) + 1
    buckets: dict[str, list[tuple[Any, TaskSpec]]] = {}
    scanned = 0
    skipped_covered = 0
    for character in index.characters:
        if produced_by_ziid.get(character.ziid, 0) >= len(wanted):
            skipped_covered += 1
            continue  # 这个字的题型都出满了，再构造一遍纯属浪费
        if scanned >= config.max_scan:
            break  # 稀有题型（如 T8 唯一解）可能永远凑不满，不能因此扫穿全池
        scanned += 1
        specs = build_all(character, rng=rng, index=index, difficulty=config.difficulty(), only=config.task_ids)
        for spec in specs:
            if (spec.task_id, character.ziid) in skip:
                continue
            bucket = buckets.setdefault(spec.task_id, [])
            if len(bucket) < cap:
                bucket.append((character, spec))
        if len(buckets) >= len(wanted) and all(len(b) >= cap for b in buckets.values()):
            break

    planned: list[tuple[Any, TaskSpec]] = []
    cursors = {task_id: 0 for task_id in buckets}
    used_ziids: set[str] = set()
    # 让位候选：这一轮因为「这个字已经用过了」被跳过、暂时不取的候选。它们**不能丢**——
    # 桶里每一条都是一个真实可出题的 (题型, 字) 组合。第一轮只负责「把字铺开」（小批量时
    # 这就够了），第二轮把让位的按桶序补齐；`limit` 只在真不够时才切。
    deferred: dict[str, list[tuple[Any, TaskSpec]]] = {task_id: [] for task_id in buckets}
    while len(planned) < config.limit:
        progressed = False
        for task_id in sorted(buckets):
            if len(planned) >= config.limit:
                break
            bucket = buckets[task_id]
            picked: tuple[Any, TaskSpec] | None = None
            while cursors[task_id] < len(bucket):
                candidate = bucket[cursors[task_id]]
                cursors[task_id] += 1
                if candidate[0].ziid not in used_ziids:
                    picked = candidate
                    break
                deferred[task_id].append(candidate)  # 让位，但不丢
            if picked is None:
                continue
            planned.append(picked)
            used_ziids.add(picked[0].ziid)
            progressed = True
        if not progressed:
            break
    # 第二轮：补齐让位过的候选。字重复优于丢数据。
    for task_id in sorted(deferred):
        if len(planned) >= config.limit:
            break
        for candidate in deferred[task_id]:
            if len(planned) >= config.limit:
                break
            planned.append(candidate)
    return planned, scanned, skipped_covered


def run_pipeline(
    config: PipelineConfig,
    *,
    client: ChatClient | None = None,
    judge_client: ChatClient | None = None,
    answer_judge_client: ChatClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """输入：PipelineConfig（可注入 client 便于测试）+ 进度回调。输出：运行统计 dict。
    预期：产出写 `config.out_path`，失败写同名 `*.rejects.jsonl`；返回统计供 CLI 打印。
          `resume=True` 时**追加写且跳过已有主键**，绝不截断既有产物。
          `config.judge=True` 时启用 L4.5 LLM 裁判；`judge_client` 可注入（测试用），
          不注入就优先用 secrets.toml 里 role="judge" 的 provider，没有则复用措辞 provider。
          `answer_judge_client` 是**答案仲裁**客户端（测试用注入）：它与裁判不是一回事，
          非 offline 一律接上——两处比对都靠它：① 解析里说的是不是定死的答案；② 读者答出的
          自然语言与定死的答案是不是同一个。子串比对只是快速通道，判定权在它。
    """
    say = progress or (lambda _msg: None)
    rng = random.Random(config.seed)
    started_at = time.perf_counter()

    data_target = Path(config.data_path) if config.data_path else default_data_path()
    limit = config.pool_size if config.pool_size > 0 else None
    characters = list(iter_characters(path=data_target, limit=limit))
    index = CharacterIndex(characters)

    stats: dict[str, Any] = {
        "pool": len(characters),
        "planned": 0,
        "ok": 0,
        "rejected": 0,
        "retried": 0,
        "first_pass": 0,
        "requests": 0,
        "resumed": 0,
        # L3.5 清洗留痕：多少条产出被清洗过、各条规则各命中多少次（绝不静默清洗）
        "by_task": {},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = Path(config.out_path)
    if out_path.resolve() == data_target.resolve():
        raise DataError(f"产出路径与输入数据是同一个文件（{out_path}）：这会覆盖原始数据。请换 --out。")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path = out_path.with_suffix(".rejects.jsonl")

    produced: set[RecordKey] = set()
    covered: set[str] = set()
    mode = "w"
    if config.resume:
        produced = load_produced_keys(out_path)
        covered = {ziid for _task_id, ziid in produced}
        mode = "a" if out_path.exists() else "w"  # 续跑模式永不截断已有文件
        if mode == "a" and ensure_trailing_newline(out_path):
            say("上轮产出最后一行被截断：已补一个换行隔开，避免新记录粘在残行尾部")
    else:
        backup = backup_output(out_path)
        if backup is not None:
            say(f"覆盖模式：旧产出已备份到 {backup.name}")
        backup_output(reject_path)
    stats["resumed"] = len(produced)
    stats["workers"] = max(1, config.workers)
    if produced:
        say(f"续跑：已有 {len(produced)} 条产出，将跳过这些主键")

    # `--exclude`（可重复）：把**别的**产出文件里的 (题型, 字) 主键也并进跳过集。
    # 换 `--out` 开新一轮时池子是从头扫的，不排除已有产出就会把上一轮同一批字再采一遍
    # ——实测 597 条产出只覆盖 87 组 (题型, 字)。排除后桶里装的是**下一批**字，
    # 覆盖才真的往前走（`_plan` 把跳过集里的 (题型, 字) 直接挡在桶外）。
    excluded = 0
    for raw in config.exclude_paths:
        extra = load_produced_keys(Path(raw))
        excluded += len(extra - produced)
        produced |= extra
    if excluded:
        covered = {ziid for _task_id, ziid in produced}
        say(f"--exclude：额外跳过 {excluded} 个主键（累计覆盖 {len(covered)} 个字）")
    stats["excluded"] = excluded

    active_providers: Sequence[ProviderConfig] = list(config.providers)
    if not config.offline:
        if client is None:
            active_providers = active_providers or load_providers(role="verbalizer")
            client = ChatClient(active_providers)
            say(f"使用 provider：{[p.name for p in active_providers]}")
        else:
            say("使用注入的 client（跳过凭据加载）")

    config, capped_note = with_local_worker_cap(config, active_providers)
    if capped_note:
        say(capped_note)
        stats["workers"] = config.workers

    if config.judge and not config.offline and judge_client is None:
        try:
            dedicated = tuple(load_providers(role="judge"))
        except ConfigError:
            dedicated = ()
        if dedicated:
            judge_client = ChatClient(list(dedicated))
            say(f"裁判使用独立 provider：{[p.name for p in dedicated]}")
        else:
            judge_client = client
            say('裁判复用措辞 provider（secrets.toml 里没配 role="judge" 的块）')

    # 答案仲裁客户端：**不受 `--judge` 开关控制**——两处比对都是「快速通过 → LLM 判定」：
    # ① 解析里说的是不是定死的答案（`answer_in_explanation` 是子串比对，过不了才问）；
    # ② 读者答出的自然语言与定死的答案是不是同一个（「读作 xié」与 `xie2`）。
    # 规则仍是快速通道：子串比对过了就不发这次请求，所以常态下不额外花配额。
    answer_judge = None
    if not config.offline:
        answer_judge = answer_judge_client or client

    # 批量大小：CLI 显式给了就用它，否则跟随 provider 配置（secrets.toml 的 `batch`）。
    # 这样 secrets.toml 里的 `batch` 不再是没人读的死配置，而调大它必须过 A/B 质量对比。
    batch_size = config.batch if config.batch > 0 else max(1, active_providers[0].batch if active_providers else 1)
    stats["batch"] = batch_size
    if batch_size > 1:
        say(f"批量模式：一次请求处理 {batch_size} 道题（质量对比见 synthesis/README.md 附二）")
    if config.judge and batch_size > 1:
        raise ConfigError(
            "批量模式暂不支持 LLM 裁判：一次请求里多道题共享一个响应，裁判无法逐题给判定。"
            "请把 --batch 设回 1，或关掉 --judge。"
        )

    planned, scanned, skipped_covered = _plan(config, index, rng, exclude=produced)
    stats["planned"] = len(planned)
    if planned and len(planned) < config.limit:
        say(
            f"计划只有 {len(planned)} 条（< --limit {config.limit}）：扫了 {scanned} 个字"
            f"（--max-scan {config.max_scan}）仍凑不满候选，稀有题型（如 T8 唯一解）会先见底；"
            "调大 --max-scan / --pool，或减少 --tasks。"
        )
    # 这两个数是**实测**的，不是「池子大小减去什么」推出来的：推出来的数会在
    # 「覆盖满跳过 + max_scan 截断」同时发生时对不上，让排查静默空跑的人更迷惑。
    stats["scanned"] = scanned
    stats["skipped_covered"] = skipped_covered
    difficulty = config.difficulty()
    if not planned:
        if produced:
            say("本次没有可出题的新字：已产出的主键全被跳过。调大 --max-scan / --pool，或换 --tasks。")
        else:
            say("本次没有产出任何题：检查 --tasks / --max-scan，以及数据文件是否可读。")

    # 负例开头：同题型**已经发布过**的题干，喂回去让模型别写重样。
    # 这是负例（已经用过的），不是范例（应该用的）——限制重复，不限制表达。
    avoid_by_task: dict[str, list[str]] = {}

    def avoid_for(spec: TaskSpec) -> tuple[str, ...]:
        """输入：TaskSpec。输出：同题型最近用过的题干（最多 AVOID_TAIL 条）。预期：负例，不是范例。"""
        return tuple(avoid_by_task.get(spec.task_id, ())[-AVOID_TAIL:])

    attempt_path = out_path.with_suffix(".attempts.jsonl")
    with OutputLock(out_path), out_path.open(mode, encoding="utf-8") as out_handle, reject_path.open(
        "a" if config.resume else "w", encoding="utf-8"
    ) as reject_handle, attempt_path.open("a" if config.resume else "w", encoding="utf-8") as attempt_handle:

        def handle(no: int, character: Any, spec: TaskSpec, verbalized: Verbalized, verification: Any,
                   history: list[dict]) -> bool:
            """输入：序号、字、TaskSpec、L3 产出、L4 结论、逐轮判定记录。
            输出：这道题最终有没有发布（bool），供调用方累计 first_pass。
            预期：可发布的题进主产出，否则进 rejects（裁判否决也算不发布）；
                  rejects 里同时留 stem 与 explanation 便于回查。
                  无论最终过没过，每一轮判定都先追加进 `<out>.attempts.jsonl`——这样「首轮被拒的原因」
                  与「被救回的题首轮错在哪」都留得下，不再只剩一个 retried 计数。
            """
            chosen = _chosen_row(history)
            judgement = chosen.get("judge")
            roundtrip = chosen.get("roundtrip")
            published = _publishable(verification, judgement, roundtrip)
            for row in history:
                attempt_handle.write(
                    json.dumps(
                        {"task_id": spec.task_id, "ziid": character.ziid, **row}, ensure_ascii=False
                    )
                    + "\n"
                )
            if history:
                attempt_handle.flush()
            if published:
                out_handle.write(
                    json.dumps(
                        _spec_to_record(
                            spec, verbalized, verification, character.ziid, difficulty,
                            judgement, roundtrip,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out_handle.flush()
                os.fsync(out_handle.fileno())  # 崩溃/断电也不丢已完成的记录
                covered.add(character.ziid)
                stats["ok"] += 1
                stats["by_task"][spec.task_id] = stats["by_task"].get(spec.task_id, 0) + 1
                avoid_by_task.setdefault(spec.task_id, []).append(verbalized.stem)
            else:
                reject_handle.write(
                    json.dumps(
                        {
                            "task_id": spec.task_id,
                            "ziid": character.ziid,
                            "stem": verbalized.stem,
                            "explanation": verbalized.explanation,
                            "failures": _round_reasons(
                                verification.tier, verification.failures, judgement, roundtrip
                            ),
                            "error": verbalized.error,
                            # 裁判判定一并留痕：裁判否决的题也进 rejects，理由不能丢
                            "judge": judgement,
                            # 回炉载荷：没有它，rejects 只是「一道死掉的题」，无法重建 TaskSpec 重试
                            "spec": _offline_record(character.ziid, spec, difficulty),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                reject_handle.flush()
                stats["rejected"] += 1
            say(f"[{no}/{len(planned)}] {spec.task_id} {character.ziid} "
                f"{'OK' if published else 'REJECT'}")
            return published

        if config.offline:
            for character, spec in planned:
                out_handle.write(
                    json.dumps(_offline_record(character.ziid, spec, difficulty), ensure_ascii=False) + "\n"
                )
                out_handle.flush()
                covered.add(character.ziid)
                stats["ok"] += 1
                stats["by_task"][spec.task_id] = stats["by_task"].get(spec.task_id, 0) + 1
        else:
            assert client is not None

            def attempt(no: int, character: Any, spec: TaskSpec) -> tuple[Verbalized, Any, int, list[dict]]:
                """输入：序号、字、TaskSpec。输出：(L3 产出, L4 结论, 实际尝试次数, 逐轮判定记录)。
                预期：负例（同题型最近发布过的题干，见 `avoid_for`）从发布进度现取，与这道题在计划
                      里的位置无关——位置是固定的，拿它当起点会让同一题型永远看到同一批负例。
                """
                return _verbalize_with_retries(
                    client, spec, retries=config.retries, temperature=config.temperature,
                    avoid=avoid_for(spec),
                    judge_client=judge_client, do_roundtrip=config.roundtrip,
                    answer_judge=answer_judge,
                )

            def record(no: int, character: Any, spec: TaskSpec, verbalized: Verbalized,
                       verification: Any, rounds: int, history: list[dict]) -> None:
                """输入：序号、字、TaskSpec、L3 产出、L4 结论、实际轮数、逐轮判定记录。
                输出：无（累计统计 + 落盘）。
                预期：单题路径与批量路径共用这一份计数，两条路径的 retried / first_pass 语义必须一致。
                """
                stats["retried"] += 1 if rounds > 1 else 0
                published = handle(no, character, spec, verbalized, verification, history)
                stats["first_pass"] += 1 if rounds == 1 and published else 0

            def record_unit(unit: list[tuple[int, Any, TaskSpec]],
                            outcomes: list[tuple[Verbalized, Any, int, list[dict]]]) -> None:
                """输入：一个派发单元 + 它的结果。输出：无（累计请求数 + 逐条落盘）。
                预期：`requests` 按**单元轮数**记——批量模式下同一单元的题共享一次请求，
                      若按单题累加会把请求数算成 batch 倍，看不出批量到底省了多少；
                      裁判与往返检验都是**额外的** provider 调用，开了就必须把它们算进去，
                      否则配额账目会漏（实测 4 条漏 5 次）。批量模式下裁判恒为 0（未接），
                      往返检验则按题计——它不合批。
                """
                stats["requests"] += max(
                    rounds for _verbalized, _verification, rounds, _history in outcomes
                )
                stats["requests"] += sum(
                    1
                    for _v, _ver, _rounds, history in outcomes
                    for row in history
                    if row.get("judge") is not None
                )
                stats["requests"] += sum(
                    1
                    for _v, _ver, _rounds, history in outcomes
                    for row in history
                    if row.get("roundtrip") is not None
                )
                for (no, character, spec), (verbalized, verification, rounds, history) in zip(unit, outcomes):
                    record(no, character, spec, verbalized, verification, rounds, history)

            queue = [(no, character, spec) for no, (character, spec) in enumerate(planned, start=1)]

            def attempt_one(unit: list[tuple[int, Any, TaskSpec]]) -> list[tuple[Verbalized, Any, int, list[dict]]]:
                """输入：单题单元。输出：单元素结果列表。
                预期：把单题路径包装成与批量路径**同形状**的单元，派发逻辑只有一份。
                """
                no, character, spec = unit[0]
                return [attempt(no, character, spec)]

            def attempt_chunk(unit: list[tuple[int, Any, TaskSpec]]) -> list[tuple[Verbalized, Any, int, list[dict]]]:
                """输入：一批 (序号, 字, TaskSpec)。输出：与单元等长、顺序一致的结果。
                预期：整批一次请求；批量路径不收 `avoid`（负例是按题给的），其余打包方式与单题同。
                """
                return _verbalize_batch_with_retries(
                    client, [spec for _no, _character, spec in unit],
                    retries=config.retries, temperature=config.temperature,
                    do_roundtrip=config.roundtrip, answer_judge=answer_judge,
                )

            if batch_size <= 1:
                units: list[list[tuple[int, Any, TaskSpec]]] = [[item] for item in queue]
                runner = attempt_one
            else:
                units = [queue[start : start + batch_size] for start in range(0, len(queue), batch_size)]
                runner = attempt_chunk

            if config.workers <= 1:
                for unit in units:
                    record_unit(unit, runner(unit))
            else:
                # 分批提交，避免一次建出 limit 个 future；单元内并发跑 L3，按序收结果写盘
                window = max(config.workers * 4, config.workers)
                with ThreadPoolExecutor(max_workers=config.workers) as pool:
                    for start in range(0, len(units), window):
                        block = units[start : start + window]
                        futures = [pool.submit(runner, unit) for unit in block]
                        for unit, future in zip(block, futures):
                            try:
                                outcomes = future.result()
                            except Exception as exc:  # noqa: BLE001 - 单元炸掉不该带走整批
                                failed = Verbalized(ok=False, error=f"措辞线程异常：{type(exc).__name__}")
                                outcomes = [
                                    (failed, verify(spec, failed), 1, []) for _no, _character, spec in unit
                                ]
                            record_unit(unit, outcomes)

    elapsed = time.perf_counter() - started_at
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["seconds_per_record"] = (
        round(elapsed / stats["ok"], 2) if stats["ok"] and not config.offline else None
    )
    stats["out_path"] = str(out_path)
    stats["reject_path"] = str(reject_path)
    stats["attempt_path"] = str(attempt_path)
    stats["total_produced"] = len(produced) + stats["ok"]

    pool_ziids = {character.ziid for character in characters}
    hit = pool_ziids & covered
    groups, codepointless = _pool_groups(characters)
    stats["coverage"] = {
        "pool": len(pool_ziids),
        "covered": len(hit),
        "ratio": round(len(hit) / len(pool_ziids), 6) if pool_ziids else 0.0,
        "remaining": len(pool_ziids) - len(hit),
        "pool_by_group": groups,
        "codepointless": codepointless,
    }
    stats["coverage_path"] = str(write_coverage(out_path, stats))
    return stats


# 回炉产物统一后缀：`<原文件名>.replayed.jsonl`
_REPLAY_SUFFIX = ".replayed.jsonl"


def _replay_left_path(rejects_path: Path) -> Path:
    """输入：rejects 文件路径。输出：回炉后**仍失败**样本的落盘路径。
    预期：先把文件名里重复的 `_REPLAY_SUFFIX` 全部剥掉再拼一次——否则拿回炉产物当输入会
          生成 `<x>.replayed.jsonl.replayed.jsonl`，回炉几次就嵌套几层（实测已出现三层）。
          输入本身就是回炉产物时，输出路径与输入相同：就地更新，永远只有一层。
    """
    path = Path(rejects_path)
    name = path.name
    while name.endswith(_REPLAY_SUFFIX):
        name = name[: -len(_REPLAY_SUFFIX)]
    return path.with_name(name + _REPLAY_SUFFIX)


def replay_rejects(
    config: PipelineConfig,
    rejects_path: Path,
    *,
    client: ChatClient | None = None,
    judge_client: ChatClient | None = None,
    answer_judge_client: ChatClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """输入：配置（决定产出路径 / 重试次数 / 温度 / 并发）+ rejects 文件 + 可注入 client + 进度回调。
    输出：回炉统计（含 `left_path`：仍失败的行另存的新文件）。

    预期：L7 飞轮——对 rejects 里每条**单条重试**（主人明确否决「小模型批量重写」，
          批量只会把质量一起拉下去），成功就追加进主产出。
          - 已存在的 (题型, 字) 直接跳过，不重复花配额；
          - 主产出**只追加不截断**（回炉绝不能把之前的成果冲掉）；
          - 老格式 rejects（没有 `spec` 载荷）如实计入 `unreplayable`，不假装回炉过；
          - 仍失败的写回炉产物文件（文件名里只保留**一层** `.replayed.jsonl`，不再层层嵌套）；
            原 rejects 文件一个字节不动——除非输入本身就是回炉产物，此时就地更新；
          - `config.judge=True` 时同样过 L4.5 裁判，规则与主流程完全一致（`judge_client` 可注入）。
          - 答案仲裁同样接上（`answer_judge_client` 可注入）：回炉与主流程共用 `verify` 与
            `roundtrip_answer`，两条路径的判据不能有两套口径。
    """
    say = progress or (lambda _msg: None)
    if config.offline:
        raise ConfigError("回炉需要重新调用 LLM 措辞，不能用 --offline / dry-run")

    out_path = Path(config.out_path)
    data_target = Path(config.data_path) if config.data_path else default_data_path()
    if out_path.resolve() == data_target.resolve():
        raise DataError(f"产出路径与输入数据是同一个文件（{out_path}）：这会覆盖原始数据。请换 --out。")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows, skipped = load_jsonl(rejects_path)
    difficulty = config.difficulty()
    produced = load_produced_keys(out_path) if config.resume else set()
    stats: dict[str, Any] = {
        "rejects_path": str(rejects_path),
        "out_path": str(out_path),
        "rows": len(rows),
        "skipped_lines": skipped,
        "unreplayable": 0,
        "skipped_existing": 0,
        "planned": 0,
        "recovered": 0,
        "still_failed": 0,
        "retried": 0,
        "first_pass": 0,
        "by_task": {},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    specs: list[TaskSpec] = []
    left: list[dict] = []
    for row in rows:
        payload = row.get("spec")
        if not isinstance(payload, Mapping):
            stats["unreplayable"] += 1
            left.append(row)
            continue
        key = (str(payload.get("task_id", "")), str(payload.get("ziid", "")))
        if config.resume and key in produced:
            stats["skipped_existing"] += 1
            continue
        specs.append(spec_from_record(payload))
    stats["planned"] = len(specs)
    say(f"回炉：待重试 {len(specs)} 条，跳过已有 {stats['skipped_existing']} 条，老格式 {stats['unreplayable']} 条")

    replay_providers: Sequence[ProviderConfig] = list(config.providers)
    if client is None:
        replay_providers = replay_providers or load_providers(role="verbalizer")
        client = ChatClient(replay_providers)
        say(f"使用 provider：{[p.name for p in replay_providers]}")
    config, capped_note = with_local_worker_cap(config, replay_providers)
    if capped_note:
        say(capped_note)

    if config.judge and judge_client is None:
        try:
            dedicated = tuple(load_providers(role="judge"))
        except ConfigError:
            dedicated = ()
        if dedicated:
            judge_client = ChatClient(list(dedicated))
            say(f"裁判使用独立 provider：{[p.name for p in dedicated]}")
        else:
            judge_client = client
            say('裁判复用措辞 provider（secrets.toml 里没配 role="judge" 的块）')

    # 答案仲裁客户端：回炉与主流程同一套判据（回炉恒为非 offline，见上面的 ConfigError）。
    answer_judge = answer_judge_client or client

    started_at = time.perf_counter()
    with OutputLock(out_path):
        mode = "a" if out_path.exists() else "w"
        if mode == "a" and ensure_trailing_newline(out_path):
            say("主产出最后一行被截断：已补一个换行隔开")
        attempt_path = out_path.with_suffix(".attempts.jsonl")
        with out_path.open(mode, encoding="utf-8") as handle, attempt_path.open(
            "a" if config.resume else "w", encoding="utf-8"
        ) as attempt_handle:

            def handle_one(no: int, spec: TaskSpec, verbalized: Verbalized, verification: Any,
                           history: list[dict]) -> bool:
                """输入：序号 + TaskSpec + L3 产出 + L4 结论 + 逐轮判定记录。
                输出：这道题最终有没有发布（bool），供调用方累计 first_pass。
                预期：每轮判定先追加进 attempts 日志——回炉救回的题也要留下首轮错在哪。
                """
                ziid = str(spec.provenance.get("seed_ziid", ""))
                chosen = _chosen_row(history)
                judgement = chosen.get("judge")
                roundtrip = chosen.get("roundtrip")
                published = _publishable(verification, judgement, roundtrip)
                for row in history:
                    attempt_handle.write(
                        json.dumps({"task_id": spec.task_id, "ziid": ziid, **row}, ensure_ascii=False) + "\n"
                    )
                if history:
                    attempt_handle.flush()
                if published:
                    handle.write(
                        json.dumps(
                            _spec_to_record(
                                spec, verbalized, verification, ziid, difficulty,
                                judgement, roundtrip,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                    stats["recovered"] += 1
                    stats["by_task"][spec.task_id] = stats["by_task"].get(spec.task_id, 0) + 1
                else:
                    stats["still_failed"] += 1
                    left.append(
                        {
                            "task_id": spec.task_id,
                            "ziid": ziid,
                            "stem": verbalized.stem,
                            "explanation": verbalized.explanation,
                            "failures": _round_reasons(
                                verification.tier, verification.failures, judgement, roundtrip
                            ),
                            "error": verbalized.error,
                            "judge": judgement,
                            "spec": _offline_record(ziid, spec, difficulty),
                        }
                    )
                say(f"[{no}/{len(specs)}] {spec.task_id} {ziid} "
                    f"{'OK' if published else 'REJECT'}")
                return published

            queue = [(no, spec) for no, spec in enumerate(specs, start=1)]
            if config.workers <= 1:
                for no, spec in queue:
                    verbalized, verification, rounds, history = _verbalize_with_retries(
                        client, spec, retries=config.retries, temperature=config.temperature,
                        judge_client=judge_client, do_roundtrip=config.roundtrip,
                        answer_judge=answer_judge,
                    )
                    stats["retried"] += 1 if rounds > 1 else 0
                    published = handle_one(no, spec, verbalized, verification, history)
                    stats["first_pass"] += 1 if rounds == 1 and published else 0
            else:
                batch = max(config.workers * 4, config.workers)
                with ThreadPoolExecutor(max_workers=config.workers) as pool:
                    for start in range(0, len(queue), batch):
                        chunk = queue[start : start + batch]
                        futures = [
                            pool.submit(
                                _verbalize_with_retries,
                                client,
                                spec,
                                retries=config.retries,
                                temperature=config.temperature,
                                judge_client=judge_client,
                                do_roundtrip=config.roundtrip,
                                answer_judge=answer_judge,
                            )
                            for no, spec in chunk
                        ]
                        for (no, spec), future in zip(chunk, futures):
                            try:
                                verbalized, verification, rounds, history = future.result()
                            except Exception as exc:  # noqa: BLE001 - 单条炸掉不该带走整批
                                verbalized = Verbalized(ok=False, error=f"措辞线程异常：{type(exc).__name__}")
                                verification = verify(spec, verbalized)
                                rounds = 1
                                history = []
                            stats["retried"] += 1 if rounds > 1 else 0
                            published = handle_one(no, spec, verbalized, verification, history)
                            stats["first_pass"] += 1 if rounds == 1 and published else 0

    left_path = _replay_left_path(rejects_path)
    with left_path.open("w", encoding="utf-8") as handle:
        for row in left:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started_at
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["elapsed_seconds"] = round(elapsed, 2)
    stats["left"] = len(left)
    stats["left_path"] = str(left_path)
    return stats


def auto_replay(
    replay: Callable[[Path], Mapping[str, Any]],
    rejects_path: Path,
    *,
    max_rounds: int,
    say: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """输入：单轮回炉函数 + 起始 rejects 文件 + 最多轮数 + 进度回调。输出：逐轮统计（去掉起止时刻）。
    预期：L7 自动回炉的**唯一收敛判据是「这一轮有没有新救回」**——
          - 起始文件不存在 → 一轮都不跑；
          - `planned == 0`（没有可回炉的样本）→ 停；
          - `recovered == 0` → 停：同一批输入 + 同一温度 + 没有状态变化，再喂一遍只会复现
            同一批失败（实测 live_r18 就是这样空转的，白烧配额）；
          - `recovered > 0` 才把残留文件当下一轮输入，最多 `max_rounds` 轮。
          残留样本一条都不丢：每一轮的残留都由 `replay` 写进 `<x>.replayed.jsonl`。
    """
    say = say or (lambda _msg: None)
    current = Path(rejects_path)
    rounds: list[dict[str, Any]] = []
    for round_no in range(1, max_rounds + 1):
        if not current.exists():
            break
        say(f"自动回炉 第 {round_no} 轮：{current}")
        one = dict(replay(current))
        rounds.append({k: v for k, v in one.items() if k not in {"started", "finished"}})
        if int(one.get("planned") or 0) == 0:
            say(f"自动回炉 第 {round_no} 轮：没有可回炉的样本，停")
            break
        if int(one.get("recovered") or 0) == 0:
            say(
                f"自动回炉 第 {round_no} 轮：一条都没救回，停"
                f"（剩余 {one.get('still_failed', 0)} 条留在 {one.get('left_path')}）"
            )
            break
        left = one.get("left_path")
        if not left:
            break
        current = Path(str(left))
    return rounds


def _reject_defense(failure: str) -> str:
    """输入：一条失败原因（形如 `防线名: 说明`）。输出：防线名。
    预期：没有冒号的旧格式整条当防线名，绝不因为格式不认识就丢样本。
    """
    head, sep, _ = str(failure).partition(":")
    return (head if sep else str(failure)).strip()


def _tally_rejects(rows: Iterable[Mapping[str, Any]]) -> dict:
    """输入：rejects 行（dict 序列）。输出：{lines, unreplayable, by_task, by_defense, by_error}。
    预期：by_task 按题型名排序；by_defense / by_error 按计数降序（同数按名字排序），
          一眼看出这一轮主要死在哪个防线上。
    """
    by_task: Counter = Counter()
    by_defense: Counter = Counter()
    by_error: Counter = Counter()
    unreplayable = 0
    lines = 0
    for row in rows:
        lines += 1
        by_task[str(row.get("task_id", "?"))] += 1
        for failure in row.get("failures") or []:
            by_defense[_reject_defense(failure)] += 1
        error = str(row.get("error") or "").strip()
        if error:
            by_error[error] += 1
        if not row.get("spec"):
            unreplayable += 1
    return {
        "lines": lines,
        "unreplayable": unreplayable,
        "by_task": dict(sorted(by_task.items())),
        "by_defense": dict(sorted(by_defense.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_error": dict(sorted(by_error.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def summarize_rejects(rejects_path: Path) -> dict:
    """输入：`.rejects.jsonl` 路径（也可直接给 `.replayed.jsonl`）。输出：只读汇总 dict。
    预期：**一个字节都不改**——rejects 只增不删，这里只把「死在哪个防线、哪个题型」数出来。
          存在同级 `<文件名>.replayed.jsonl` 时一并汇总到 `replayed` 字段：那是回炉后**仍失败**
          的样本，所以它的 by_defense 是「换过措辞、回过原因之后还过不去的硬骨头」。
    """
    path = Path(rejects_path)
    if not path.exists():
        raise DataError(f"rejects 文件不存在：{path}")
    rows, skipped = load_jsonl(path)
    report: dict[str, Any] = {"path": str(path), "skipped": skipped, **_tally_rejects(rows)}
    replayed = path.with_name(path.name + ".replayed.jsonl")
    if replayed.exists():
        left_rows, left_skipped = load_jsonl(replayed)
        report["replayed"] = {"path": str(replayed), "skipped": left_skipped, **_tally_rejects(left_rows)}
    return report


def summarize_attempts(attempts_path: Path) -> dict:
    """输入：`<out>.attempts.jsonl` 路径。输出：逐轮判定汇总 dict。
    预期：回答「首轮被拒的是什么、后来救回了多少」——按 (题型, 字) 归并每道题的各轮判定：
          首轮通过记 first_pass；首轮没过而末轮过了记 recovered；末轮还没过记 still_failed。
          「通过」用的是与发布同一个判据 `_round_passed`——开了裁判时，程序全过但裁判否决的那轮
          也算没过（否则线上没发布、这里却记成一次过），并记进 by_first_defense 的「裁判否决」桶。
          `by_first_defense` 只统计**首轮非 A**的题，直接指出是哪条闸门先把题拦下来的。
    """
    path = Path(attempts_path)
    if not path.exists():
        raise DataError(f"attempts 文件不存在：{path}")
    rows, skipped = load_jsonl(path)
    per_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("task_id", "?")), str(row.get("ziid", "?")))
        per_key.setdefault(key, []).append(row)

    first_pass = recovered = still_failed = 0
    by_first_defense: Counter = Counter()
    for rounds in per_key.values():
        ordered = sorted(rounds, key=lambda row: int(row.get("round", 0)))
        first, last = ordered[0], ordered[-1]
        if _round_passed(first.get("tier"), first.get("judge"), first.get("roundtrip")):
            first_pass += 1
            continue
        for failure in _round_reasons(
            first.get("tier"), first.get("failures"), first.get("judge"), first.get("roundtrip")
        ):
            by_first_defense[_reject_defense(failure)] += 1
        if _round_passed(last.get("tier"), last.get("judge"), last.get("roundtrip")):
            recovered += 1
        else:
            still_failed += 1
    ranked = lambda counter: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))  # noqa: E731
    return {
        "path": str(path),
        "skipped": skipped,
        "records": len(per_key),
        "first_pass": first_pass,
        "recovered": recovered,
        "still_failed": still_failed,
        "by_first_defense": ranked(by_first_defense),
    }
