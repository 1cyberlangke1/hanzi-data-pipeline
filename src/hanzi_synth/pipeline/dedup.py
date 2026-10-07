"""L6 去重与污染检测：产出里「换汤不换药」的题必须由程序揪出来，不靠人工抽检。

主人的两条硬要求决定了本模块的形态：
- **不人工抽检** —— 近似重复由 n-gram 指纹 + Jaccard 自动判定；
- **产出必须自足、假设字典一定对** —— 污染检测查的是「题干有没有把外部语料大段抄进去」，
  而不是去质疑字典。

本模块**只读产出、只写旁路文件**：原产出文件一个字节都不改。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..verify.checks import normalize_for_match

# 默认 n=4：中文四字窗口足以区分「同一个字的两种问法」与「换了字的另一种问法」
DEFAULT_NGRAM = 4
# 默认阈值 0.9：低于此值只算「相似」，不算「重复」——宁可漏判也不误杀有效多样性
DEFAULT_THRESHOLD = 0.9
# 候选上限：倒排表里公共 gram 最多的前 N 条参与精确比对，防止长尾把单条拖成 O(n)
CANDIDATE_CAP = 400
# 一条文本最多看多少个「最稀有」的 gram 去捞候选。模板套话（高频 gram）对找候选毫无区分度：
# 不设这个预算的话，「所有题共用同一模板」会让每条都去扫几万条倒排表，整体退化成 O(N²)。
GRAM_BUDGET = 24
# 高频 gram 封顶：出现次数超过 max(MIN_DF_CUTOFF, MAX_DF_RATIO * 已收录条数) 的 gram 视为套话。
# 但凑够 MIN_GRAMS 个 gram 之前不封顶——否则整份语料都是同一模板时一个候选都捞不到。
MIN_GRAMS = 4
MAX_DF_RATIO = 0.02
MIN_DF_CUTOFF = 32
# 污染窗口 n=13：命中即意味着题干与上游语料**连续照抄 13 字**，远超「改写」的合理范围
CONTAMINATION_NGRAM = 13
# 语料最多扫多少条：0 = 不限。索引只建在**题干**上，所以内存与语料规模无关，
# 扫描是流式的（逐条语料看它命中哪些题干 n-gram），100MB 语料也不会把内存吃爆
DEFAULT_MAX_TEXTS = 0
# 报告里最多列这么多条命中样本，避免报告本身变成第二个大文件
REPORT_HITS = 50
# 从 jsonl 里抽语料时优先看的字段名（含字海 meanings.mean_h 的路径）
_TEXT_KEYS = ("text", "stem", "meaning", "content", "mean_h", "explanation")
# 允许递归下钻的容器字段：只认这些，避免把 ziid / radical 之类的短字段当成语料
_CONTAINER_KEYS = ("meanings", "property", "mean")


def _shingles(text: str, size: int) -> frozenset[str]:
    """输入：文本 + n。输出：归一化后的字符级 n-gram 集合。
    预期：先 `normalize_for_match` 去掉空白与列举分隔符，所以「E3 90 80」与「E3、90、80」
          视为同一串；空文本返回空集；短于 n 的文本返回整串单元素集，
          否则「整个数据集都是短句」时所有条目都会因无指纹而被判成不同。
    """
    norm = normalize_for_match(text or "")
    if not norm:
        return frozenset()
    if len(norm) <= size:
        return frozenset({norm})
    return frozenset(norm[index : index + size] for index in range(len(norm) - size + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """输入：两个 n-gram 集合。输出：Jaccard 相似度 0..1。预期：任一为空返回 0.0（空集不算相似）。"""
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


class DedupIndex:
    """输入：n / 阈值 / 候选上限。输出：可增量加入的近似去重索引。

    预期：`add(text)` 先用 n-gram 倒排表捞出**可能有重合的候选**，只对候选算 Jaccard，
          不做 O(n²) 两两比较——完全不像的两条文本没有公共 n-gram，倒排表里根本不会相遇。
          候选按「公共 gram 数降序、下标升序」排序后截断，保证结果与哈希种子无关（可复现）。
    """

    def __init__(
        self,
        *,
        ngram: int = DEFAULT_NGRAM,
        threshold: float = DEFAULT_THRESHOLD,
        candidate_cap: int = CANDIDATE_CAP,
        gram_budget: int = GRAM_BUDGET,
    ) -> None:
        """输入：n / 阈值 / 候选上限。输出：索引实例。预期：非法参数当场报错，不带到运行期。"""
        if ngram < 1:
            raise ValueError(f"ngram 必须 >= 1，收到 {ngram}")
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold 必须在 (0, 1]，收到 {threshold}")
        if candidate_cap < 1:
            raise ValueError(f"candidate_cap 必须 >= 1，收到 {candidate_cap}")
        if gram_budget < 1:  # 0 会让捞候选的循环立刻 break，静默退化成「永不判重」
            raise ValueError(f"gram_budget 必须 >= 1，收到 {gram_budget}")
        self.ngram = ngram
        self.threshold = threshold
        self.candidate_cap = candidate_cap
        self.gram_budget = gram_budget
        self._postings: dict[str, list[int]] = {}
        self._grams: list[frozenset[str]] = []
        # 精确层：归一化全文 -> 首次出现的位置。O(1) 全覆盖，最省事的一层不能漏
        self._exact: dict[str, int] = {}

    def add(self, text: str) -> tuple[bool, float, int | None]:
        """输入：一条文本。输出：(是否与已收录的某条重复, 最高相似度, 最相似那条的下标或 None)。
        预期：**无论是否重复都收录**——否则重复簇里第二条之后就没法被后面的条目比中。
              先走精确层（归一化全文命中即 1.0），再走近似层：按 df 升序只看最稀有的
              `GRAM_BUDGET` 个 gram 捞候选，跳过被判定为套话的高频 gram，最后逐个算 Jaccard。
        """
        grams = _shingles(text, self.ngram)
        norm = normalize_for_match(text or "")
        position = len(self._grams)
        self._grams.append(grams)
        exact = self._exact.get(norm) if norm else None
        if exact is not None:
            for gram in grams:
                self._postings.setdefault(gram, []).append(position)
            return True, 1.0, exact

        df_cutoff = max(MIN_DF_CUTOFF, int(len(self._grams) * MAX_DF_RATIO))
        counts: dict[int, int] = {}
        used = 0
        for df, gram in sorted((len(self._postings.get(g, ())), g) for g in grams):
            if used >= self.gram_budget or (used >= MIN_GRAMS and df > df_cutoff):
                break
            used += 1
            for index in self._postings.get(gram, ()):
                counts[index] = counts.get(index, 0) + 1
            if len(counts) > self.candidate_cap * 4:
                break
        candidates = sorted(counts, key=lambda index: (-counts[index], index))[: self.candidate_cap]
        best = 0.0
        best_index: int | None = None
        for index in candidates:
            score = _jaccard(grams, self._grams[index])
            if score > best:
                best, best_index = score, index
        for gram in grams:
            self._postings.setdefault(gram, []).append(position)
        if norm:
            self._exact.setdefault(norm, position)
        return best >= self.threshold, best, best_index

    def __len__(self) -> int:
        """输入：无。输出：已收录条数。预期：给报告与断言用。"""
        return len(self._grams)


def load_jsonl(path: Path) -> tuple[list[dict], int]:
    """输入：jsonl 路径。输出：(解析成功的记录列表, 跳过的坏行数)。
    预期：空行与坏 JSON 都跳过并计数——报告要如实说明「有多少行没能参与」；
          文件不存在返回 ([], 0)，不抛异常（旁路工具不该因为缺文件崩掉）。
    """
    rows: list[dict] = []
    skipped = 0
    if not path.exists():
        return rows, skipped
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                skipped += 1
    return rows, skipped


def _row_ref(row: Mapping[str, Any], line_no: int) -> dict:
    """输入：一行记录与它在文件里的行号。输出：报告里引用这条记录的最小信息。
    预期：只带 line / task_id / ziid，不把整条题干塞进报告（报告要能一眼看完）。
    """
    return {"line": line_no, "task_id": row.get("task_id"), "ziid": row.get("ziid")}


def dedup_file(
    path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    ngram: int = DEFAULT_NGRAM,
    key: str = "stem",
) -> tuple[dict, list[dict]]:
    """输入：产出 jsonl 路径 + 阈值 + n + 取文本的字段名。输出：(去重报告, 保留的记录列表)。
    预期：**只读原文件**；首条出现的记录保留，其后相似度 >= threshold 的记为重复并归到同一组；
          缺 key 字段或该字段为空的记录单独计数（`empty`），既不保留也不判重，交由调用方决定。
    """
    index = DedupIndex(ngram=ngram, threshold=threshold)
    kept: list[dict] = []
    groups: dict[int, dict] = {}
    # `index.add` 返回的是**索引内部下标**（每条都进索引），而 `groups` 的键是**保留记录的下标**，
    # 有重复行之后两者就错位了；这里记下每个内部下标属于哪个组，链式重复（A≈B≈C）也能归到同一组。
    group_of: list[int] = []
    parsed = skipped = empty = duplicates = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(row, dict):
                skipped += 1
                continue
            parsed += 1
            text = row.get(key)
            if not isinstance(text, str) or not text.strip():
                empty += 1
                continue
            duplicate, similarity, match = index.add(text)
            if duplicate and match is not None:
                duplicates += 1
                group = group_of[match]
                groups[group]["dropped"].append({**_row_ref(row, line_no), "similarity": round(similarity, 4)})
                group_of.append(group)
            else:
                position = len(kept)
                kept.append(row)
                groups[position] = {"kept": _row_ref(row, line_no), "dropped": []}
                group_of.append(position)
    report = {
        "out_path": str(path),
        "key": key,
        "ngram": ngram,
        "threshold": threshold,
        "records": parsed,
        "skipped": skipped,
        "empty": empty,
        "unique": len(kept),
        "duplicates": duplicates,
        "groups": [group for group in groups.values() if group["dropped"]],
    }
    return report, kept


def _extract_text(value: Any, depth: int = 0) -> str:
    """输入：jsonl 一行的任意值。输出：其中第一个能当语料用的字符串（列表则按行拼接）。
    预期：只认 `_TEXT_KEYS` 与 `_CONTAINER_KEYS`，**不**随手递归所有字段——
          否则 chars.jsonl 的 `ziid`（第一个字符串值）会被当成释义原文。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (_extract_text(item, depth) for item in value) if part)
    if isinstance(value, Mapping) and depth < 3:
        for key in _TEXT_KEYS:
            if key in value:
                found = _extract_text(value[key], depth + 1)
                if found:
                    return found
        for key in _CONTAINER_KEYS:
            if key in value:
                found = _extract_text(value[key], depth + 1)
                if found:
                    return found
    return ""


def read_corpus(path: Path) -> list[str]:
    """输入：语料文件（.jsonl 或纯文本）。输出：文本列表。
    预期：jsonl 逐行抽文本字段，抽不到就跳过该行（**不退化为整行 JSON**，那会把
          `ziid` 之类的键名混进语料）；纯文本按行读。坏行跳过——一条脏数据不该带走整个语料。
    """
    texts: list[str] = []
    if not path.exists():
        return texts
    is_jsonl = path.suffix.lower() == ".jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if not is_jsonl:
                texts.append(line)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            found = _extract_text(row)
            if found:
                texts.append(found)
    return texts


def contamination(
    stems: Sequence[str],
    corpus_texts: Sequence[str],
    *,
    ngram: int = CONTAMINATION_NGRAM,
    max_texts: int = DEFAULT_MAX_TEXTS,
) -> dict:
    """输入：待查题干序列 + 外部语料文本序列 + n + 语料扫描上限。输出：污染报告。
    预期：**没给语料就如实报 `checked: False`**，不假装查过；
          给了语料则**只把题干的 n-gram 建成探针表**，再流式扫语料看命中——
          内存只与题干条数有关，与语料规模无关；题干里任何一条 n-gram 命中即判该条污染
          （n=13 意味着「连续照抄 >= 13 字」）。扫到 max_texts 就停并标记 `truncated: true`。
    """
    if not corpus_texts:
        return {
            "checked": False,
            "reason": "未提供外部语料，无法判断题干是否抄了上游原文",
            "ngram": ngram,
            "stems": len(stems),
        }
    probes: dict[str, list[int]] = {}
    for index, stem in enumerate(stems):
        for gram in _shingles(stem, ngram):
            probes.setdefault(gram, []).append(index)
    matched: dict[int, set[str]] = {}
    processed = 0
    truncated = False
    for text in corpus_texts:
        if max_texts and processed >= max_texts:
            truncated = True
            break
        processed += 1
        for gram in _shingles(text, ngram):
            for index in probes.get(gram, ()):
                matched.setdefault(index, set()).add(gram)
    hits = [
        {"index": index, "sample": sorted(grams)[0], "matched_ngrams": len(grams)}
        for index, grams in sorted(matched.items())
    ]
    return {
        "checked": True,
        "ngram": ngram,
        "stems": len(stems),
        "corpus_texts": len(corpus_texts),
        "processed_texts": processed,
        "truncated": truncated,
        "contaminated": len(hits),
        "ratio": round(len(hits) / len(stems), 6) if stems else 0.0,
        "hits": hits[:REPORT_HITS],
    }


def contamination_file(
    out_path: Path,
    corpus_path: Path | None,
    *,
    ngram: int = CONTAMINATION_NGRAM,
    max_texts: int = DEFAULT_MAX_TEXTS,
    key: str = "stem",
) -> dict:
    """输入：产出路径 + 语料路径（可为 None）+ n + 上限 + 取文本字段。输出：污染报告。
    预期：语料为 None 时仍返回 `checked: False` 的报告（如实说没查），不抛错。
    """
    rows, _skipped = load_jsonl(out_path)
    stems = [row[key] for row in rows if isinstance(row.get(key), str) and row[key].strip()]
    corpus = read_corpus(corpus_path) if corpus_path is not None else []
    report = contamination(stems, corpus, ngram=ngram, max_texts=max_texts)
    report["out_path"] = str(out_path)
    report["corpus_path"] = str(corpus_path) if corpus_path is not None else None
    return report


def write_report(out_path: Path, report: Mapping[str, Any], *, suffix: str = ".dedup.json") -> Path:
    """输入：产出路径 + 报告 + 旁路后缀。输出：旁路报告路径。
    预期：写 `<out><suffix>`（默认 `<out>.dedup.json`），原产出一个字节不动。
    """
    target = out_path.with_name(out_path.name + suffix)
    target.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def write_kept(out_path: Path, kept_rows: Sequence[Mapping[str, Any]]) -> Path:
    """输入：产出路径 + 去重后保留的记录。输出：新文件路径。
    预期：写 `<out>.dedup.jsonl`；**不覆盖原产出**——要替换得由主人显式决定，
          程序不替用户删数据。
    """
    target = out_path.with_name(out_path.name + ".dedup.jsonl")
    with target.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            handle.flush()
    return target
