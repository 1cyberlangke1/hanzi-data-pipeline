"""chars.jsonl 的类型化记录。

只承载合成需要的字段；原始 dict 不保留，避免下游随手取到「已决定不用」的字段
（两分 / 叶码一族全部禁用）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

# 释义库标签（数据自带字段）：mean_h 中华字海 / mean_z 综合 / mean_b 补遗 / mean_v 异体 /
# mean_k 康熙字典 / mean_j 集韵 / mean_g 古文字。标签只用于区分来源，不参与判分。


@dataclass(frozen=True)
class Character:
    """输入：chars.jsonl 的一条记录。输出：类型化的字记录。
    预期：`ids_candidates` 非空才代表该字进合成字库（唯一的准入条件）。
    """

    ziid: str
    zi: str
    ids_candidates: tuple[str, ...]
    stroke_total: int
    radical: str | None
    structure: int | None
    standard_form: str | None
    meanings: Mapping[str, tuple[str, ...]]
    variants: tuple[str, ...]
    refs: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @property
    def has_codepoint(self) -> bool:
        """输入：无。输出：该字是否有 Unicode 码位。
        预期：ZIID 以 U 开头 = 有码位（可直接书写）；Z/P 开头 = 无码位（须用占位符回填）。
        """
        return self.ziid.startswith("U")

    @property
    def group(self) -> str:
        """输入：无。输出：'U' / 'Z' / 'P' 三组之一，供分池统计。"""
        return self.ziid[0] if self.ziid[:1] in {"U", "Z", "P"} else "?"

    def meaning_blocks(self, labels: tuple[str, ...] = ("mean_h", "mean_z", "mean_b", "mean_v", "mean_k", "mean_j")) -> tuple[str, ...]:
        """输入：要取的释义库标签。输出：非空的释义块文本（顺序按标签顺序）。
        预期：只做拼接与去空，不做切分——切分交给 facts.text 模块。
        """
        out: list[str] = []
        for label in labels:
            out.extend(self.meanings.get(label, ()))
        return tuple(block for block in out if block and block.strip())


# 释义块是爬虫带下来的**原文形态**，里面混着排版残片与词典内部标记。下面几条把它们逐条钉死，
# 全部是确定性字符串处理——构造时先删干净，模型就抄不到，也接不上错误的成分：
# - `<br>`：分点/换行的 HTML 残片，不是释义内容；
# - 「拼音xxx」小段：只是读音标注，读音已由 reading 字段单独给出。留着它，模型会把紧随其后的
#   解释挂到读音上——实测小模型对 `拼音xie2<br>[～毒]印度的古译名。` 输出
#   「这个字的读音是 xie2，拼音对应的是印度的古译名」，整句不成话；
# - 词头引用：`～` 就是本字本身，这一小段讲的是**词**的意思，本字单用没有这层意思。删掉词头
#   后剩下的「印度的古译名」无所指，模型就会自己编一个主语接上去。两种写法都要认：
#   `[～毒]印度的古译名`（词头在前）与 `[三～]姓氏`（词头在后）。只认前者时后者会漏进题干，
#   实测 T13 因此漏出 `释义是「[三～]姓氏」` 这种词典记法（400 条产出里 44 条带这类残片）；
# - 「同【依】」这类互见条目：它会把答案那个字直接写出来（实测小模型原样抄进题干 = 泄题）。
_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
# 段首的读音标注：「拼音xie2」「读音gul」「ㄨㄩ拼音hao3」，可带 ㈠㈡ / ① 这类序号前缀。
# 「读音gul」是朝鲜吏读的注音，和「拼音」一样是词典的注音标记、不是释义，所以一并剥掉。
_LEADING_READING_RE = re.compile(r"^[^\u3400-\u9fff]*?(?:拼音|读音)\s*[A-Za-z\u00c0-\u024f]+\s*\d?\s*。?")
_HEADWORD_SEGMENT_RE = re.compile(r"[\[〔【][^\]〕】]*[～~]")
_CROSS_REF_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]?\s*同\s*[【『“][^】』”]*[】』”]\s*。?")
# 「见《直音篇》」「见汉语方言大词典」「见「偛」」这类**互见条目**：讲的是别处（另一本书 / 另一个字）
# 怎么说，不是本字单用的释义。留着它有两重害处：
#   ① 出处会被写进产出——实测 3,401 条 LLM 产出里 1,174 条带「根据康熙字典 / 出自《中华字海》」
#      这类引用（22% 的 T17），读者手上并没有那份资料；
#   ② 它是**全汉字**，形状上是一条合法义项，会直接当上答案（「这个字是什么意思？→ 见汉语方言大词典」）。
# 只认「见 + 书名号 / 括号 / 字典类词」三种形态，不碰「罕见」「意见」这类正常的「见」。
_SEE_ALSO_RE = re.compile(
    r"[①②③④⑤⑥⑦⑧⑨⑩]?\s*见\s*(?:《[^》]*》|[【『「“][^】』」”]*[】』」”]"
    r"|[^，。；、]{0,10}(?:字典|词典|字海|集韵|广韵|说文|尔雅|篇))\s*。?"
)
# 释义块里的**分点序号**（①⑴㈠ 这类）是词典排版，不是释义内容：留着它，产出就会写成
# 「含义 ①归。①归依。②转身」——读者看到的是字典的编号，不是人说的话。
# 只剥**段首**的序号，段落边界靠 join("\n") 保留（去掉序号不等于把几个义项粘成一句）。
_ENUM_PREFIX_RE = re.compile(r"^[\s\u2460-\u2473\u2474-\u2487\u3220-\u3229]+")


def clean_meaning_block(block: str) -> str:
    """输入：一个释义块的原始文本。输出：只留本字单用释义的正文（原有分点保留成换行）。
    预期：逐小段丢弃「纯读音标注 / 词头引用 / 互见条目」与空段，不改写任何释义内容；
          返回空串表示该块没有本字单用的释义，调用方据此不出题。
    """
    kept: list[str] = []
    for segment in _BREAK_RE.split(block or ""):
        text = _LEADING_READING_RE.sub("", segment)
        text = _CROSS_REF_RE.sub("", text)
        text = _SEE_ALSO_RE.sub("", text).strip()
        text = _ENUM_PREFIX_RE.sub("", text).strip()
        if not text or _HEADWORD_SEGMENT_RE.search(text):
            continue
        kept.append(text)
    return "\n".join(kept)
