"""chars.jsonl 加载与准入过滤。

准入规则（findings.md §9.7 定稿 + 本次收紧）：**只要解析出 >=1 个「语法完整」的 IDS 候选就进池**，
其余字段（释义 / 部首 / 读音 / 码位）一律不作准入条件。

「语法完整」= 用 ids.parse_tree 能完整消费（描述符元数够）。
实测差异：词法合法 79,254 条 → 剔除残缺 IDS（如 `㐃` 的 `⿱丨`）后 **79,042 条**。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..config import default_data_path
from .ids import IdsSyntaxError, parse_candidates, parse_tree
from .records import Character


class DataError(RuntimeError):
    """输入：数据文件问题描述。输出：可读异常。预期：调用方打印人话并退出，不抛栈给用户。"""


def _complete_candidates(candidates: tuple[str, ...]) -> tuple[str, ...]:
    """输入：词法合法的候选元组。输出：只保留能被 parse_tree 完整解析的候选。
    预期：全部残缺时返回空元组——该字据此判为「无可用 IDS」，不进池。
    """
    out: list[str] = []
    for candidate in candidates:
        try:
            parse_tree(candidate)
        except IdsSyntaxError:
            continue
        out.append(candidate)
    return tuple(out)


def _to_character(raw: dict) -> Character | None:
    """输入：一条原始记录 dict。输出：Character，或 None（没有可用 IDS 时）。
    预期：字段缺失按 None/0 兜底，不因为个别脏字段丢整条。
    """
    prop = raw.get("property") or {}
    candidates = _complete_candidates(parse_candidates(prop.get("ids")))
    if not candidates:
        return None  # 准入闸门：无可用 IDS 直接出局

    meanings = raw.get("meanings") or {}
    normalized_meanings = {
        key: tuple(str(item) for item in (value or []) if str(item).strip())
        for key, value in meanings.items()
    }
    variants = tuple(text for text in (_variant_text(item) for item in (raw.get("variants") or [])) if text)

    return Character(
        ziid=str(raw.get("ziid") or prop.get("ziid") or ""),
        zi=str(prop.get("zi") or ""),
        ids_candidates=candidates,
        stroke_total=int(prop.get("stroke_total") or 0),
        radical=(prop.get("radical") or None),
        structure=(int(prop["structure"]) if prop.get("structure") is not None else None),
        standard_form=(prop.get("standard_form") or None),
        meanings=normalized_meanings,
        variants=variants,
        refs=_refs(prop),
    )


# 只有 zh_ref 的书名在 crawler/README.md 里有明确记载（中华字海）；
# hd/tg/bf 三个字段的数据形状相同但书名无据可查，故只在 refs 里留通用键，
# **不**在题目里给它们编书名（编了就是幻觉）。
_REF_FIELDS = (("zh_ref", "zh"), ("hd_ref", "hd"), ("tg_ref", "tg"), ("bf_ref", "bf"))


def _refs(prop: dict) -> dict[str, dict[str, int]]:
    """输入：原始 property。输出：{库键: {"page": 页, "number": 序号}}。
    预期：形状不符（缺 page 或不是 int）的字段直接跳过，绝不猜。
    """
    out: dict[str, dict[str, int]] = {}
    for field_name, key in _REF_FIELDS:
        value = prop.get(field_name)
        if isinstance(value, dict) and isinstance(value.get("page"), int):
            out[key] = {"page": int(value["page"]), "number": int(value.get("number") or 0)}
    return out


def _variant_text(item) -> str:
    """输入：variants 里的一项（dict 或字符串）。输出：可直接书写的字符，取不到时返回空串。
    预期：`{"zi": "壹"}` 直接用；只有 `{"ziid": "U058F9"}` 时按码位还原成「壹」——
          这一步是确定性换算，不引入任何外部信息。
    """
    if isinstance(item, dict):
        text = str(item.get("zi") or "")
        ziid = str(item.get("ziid") or "")
    else:
        text, ziid = "", str(item)
    if text:
        return text
    if len(ziid) >= 5 and ziid[:1] == "U":
        try:
            return chr(int(ziid[1:], 16))
        except ValueError:
            return ""
    return ""


def iter_characters(
    path: Path | str | None = None,
    *,
    limit: int | None = None,
) -> Iterator[Character]:
    """输入：chars.jsonl 路径（默认 crawler/data/chars.jsonl）、条数上限。
    输出：惰性迭代进池的 Character。
    预期：文件不存在时抛 DataError；坏行报出具体行号；limit 在**过滤后**计数。
    """
    target = Path(path) if path is not None else default_data_path()
    if not target.exists():
        raise DataError(f"找不到数据文件 {target}（原始数据不入库，请先跑 crawler）")

    emitted = 0
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{target}:{line_no} 不是合法 JSON：{exc}") from exc
            record = _to_character(raw)
            if record is None or not record.ziid:
                continue
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def load_characters(path: Path | str | None = None, *, limit: int | None = None) -> list[Character]:
    """输入：同 iter_characters。输出：进池字的列表。预期：小批量验证用，全量请用迭代器。"""
    return list(iter_characters(path, limit=limit))
