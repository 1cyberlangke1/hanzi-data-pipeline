"""离线程序化渲染：题目与解析全部由程序从 TaskSpec 的事实拼出，不经任何模型。

输入：TaskSpec（构造器已经算好的事实与答案）+ 一个 variant 序号。
输出：(stem, explanation)。确定性：同一条 spec、同一个 variant，永远渲染出同一句话。
预期：模板由程序自己写，措辞正确性由模板本身保证——这里不设闸门。唯一由数据决定、
      写模板时看不出来的约束是「非选择题的答案不许出现在题干里」，所以 `render` 对
      会泄题的变体返回 None，由调用方换下一个句式。

一条题目 = 开场语 + 问法骨架 + 候选呈现（仅选择题）+ 收尾语。
**多样性主要来自问法骨架**：每种题型都写了 30 条结构不同的问法（T8 约束题 130 条；直接问、指令式、
数一数式、铺垫两句式、从属性入手式、填空式……），不是同一句话换同义词。
骨架之上再叠加 4 种引号写法（「」/『』/“”/裸字）与 4 种候选列举写法，
所以同一条事实能写出几千种不同句子。骨架里**只有 TaskSpec 的事实**，不引入任何外部信息。

引号只加在**单个汉字**上：给数字、拼音、码位加引号读着别扭，所以主语不是汉字时
自动退回裸写（见 `_quote`）。

释义类题（T13 靠释义认字 / T17 问字义）**不在这里**：它们的题干或答案必须带上字典释义原文，
而 AGENTS.md §六 规定释义类题必须走 LLM 路径（释义原文只进提示词、不进产出）。离线模板拿不到
释义，实测渲染出来的是「读作 yī、yǐn 的字，请写出它的结构拆解」——同音字有一大堆，读者无从
确定问的是哪一个；T17 的解析则是把字典原文当答案抄了出去（「㐇」的释义是「古文書所見奴婢名也」）。
所以这两型的骨架与解析一并从这里删掉，不留死代码；名单见 `LLM_ONLY_TASK_IDS`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..tasks.base import TaskSpec
from ..verify.checks import answer_leak_limit, normalize_for_match

# 开场语：**只有空串**。原先这里挂过 7 条播报前缀（「小测验：」「来一题：」「看这一题：」
# 一类），它们覆盖了 87.5% 的题干，却一个新问题都没提出——「不加前缀不也能看出是一个问题？」
# 主人问得对：前缀是**凑数用的**，不是句式。删掉后 8 倍放大没了，句式数由骨架自己挣
# （T8 因此补到 100 条骨架）。留这个空元组是为了 `variant_space` / `render` 的混合进制
# 少一处分支——维度还在，取值只有一个。
_OPENERS: tuple[str, ...] = ("",)

# 收尾语：同样只是语气，不引入新信息。
_TAILS: tuple[str, ...] = (
    "",
    "请写出答案。",
    "写出结果即可。",
    "给出你的答案。",
    "把答案写出来。",
)

# 引号写法：同一句骨架换一种引法，看着就不像同一条模板。
_QUOTES: tuple[tuple[str, str], ...] = (("「", "」"), ("『", "』"), ("“", "”"), ("", ""))

# 候选的列举写法（选择题专用）。
_LIST_STYLES: tuple[str, ...] = ("、", "，", " / ", "；")

# 选择题的候选呈现方式：候选必须**逐个原样**列出。
_OPTION_STYLES: tuple[str, ...] = (
    "候选是：{options}。",
    "选项为：{options}。",
    "可选的有：{options}。",
    "从 {options} 里挑一个。",
    "在 {options} 之中选一个。",
    "给出的候选是 {options}。",
)

# 解析的开场语与收尾语（「答案是：」「由此可知：」/「这就是答案。」）**已删除**：它们从来没生效过——
# 解析的混合进制原先接在题干进制之后，`variant` 走到那一步时已经被除到 0，六种开场与四种收尾永远
# 只取第 0 个（实测全量 739,005 条的解析 100% 是空串开场）。它们是纯语气、不添信息，且与题干收尾语
# 重复，所以处理方式不是「激活」而是删掉；解析的多样性改由每型 3~5 条**不同说法**的解析骨架承担
# （`_EXPLANATIONS`），它们在 `variant_space` 里占一条**独立的高位轴**，与题干互不占用进制位。

# 每种题型的**问法骨架**：每型 24 条，条与条之间句式结构不同，不是同义词替换。
# 骨架里只用 TaskSpec 的事实占位符：{subject} 主语 / {must} 附带信息 / {options} 候选 /
# {path} 方位 / {conditions} 条件 / {edits} 替换步骤 / {direction_word} 最多或最少。
# **非选择题的骨架里绝不出现 {answer}**——答案不许出现在题干里。
_CORES: Mapping[str, tuple[str, ...]] = {
    "T1_stroke_total": (
        "{subject}一共有多少笔画？",
        "{subject}的总笔画数是多少？",
        "数一数，{subject}有多少笔。",
        "{subject}写出来一共几画？",
        "请给出{subject}的笔画总数。",
        "{subject}是由几笔组成的？",
        "把{subject}的笔画数写出来。",
        "{subject}按笔画数算，是几画？",
        "{subject}需要写几笔？",
        "统计一下{subject}的笔画，结果是几？",
        "先看{subject}这个字，它一共几画？",
        "{subject}可以数出多少笔？",
        "在笔画这一项上，{subject}记作多少画？",
        "{subject}这个字有几画？",
        "请数出{subject}的笔画总数。",
        "{subject}用几笔能写完？",
        "{subject}这个字总共用几笔？",
        "{subject}的笔画数目是几？",
        "把{subject}拆成笔画来数，一共多少笔？",
        "{subject}的笔数是多少？",
        "{subject}有几个笔画？",
        "写{subject}这个字，一共要下多少笔？",
        "{subject}按笔顺写完，一共经过多少笔？",
        "{subject}的笔画统计结果是多少？",
        "{subject}几画？",
        "{subject}数出来是几画？",
        "{subject}的笔画，数出来是几？",
        "{subject}按笔画算，合计多少笔？",
        "请数一下{subject}的笔画数，直接给数字。",
        "{subject}总共几笔，数数看。",
    ),
    "T3_ids_forward": (
        "请拆解{subject}，写出它的结构式。",
        "{subject}该怎么拆解？写成一个结构式字符串。",
        "把{subject}拆解开，得到的结构式是什么？",
        "请写出{subject}的结构拆解结果。",
        "{subject}能拆解成什么样的结构？写出来。",
        "用部件结构式表示{subject}，拆解结果是什么？",
        "{subject}由哪些部件、以什么方式组成？拆解后写出结构式。",
        "给出{subject}的部件拆解串。",
        "拆解{subject}，把部件之间的组合关系写成结构式。",
        "{subject}拆解后的结构是什么？用结构式回答。",
        "把{subject}拆解后的组成写成结构式字符串。",
        "请给出{subject}的拆解式。",
        "{subject}拆解出的部件组合关系怎么写？",
        "写出{subject}拆解后的结构描述串。",
        "{subject}可以拆解成哪些部件？用结构式表示。",
        "{subject}拆解出的结构式应该写成什么？",
        "{subject}的部件层级怎么拆解表示？写出结构式。",
        "请把{subject}拆解到部件一级，写出结构式。",
        "{subject}拆解出的结构式字符串是什么？",
        "用结构式描述{subject}拆解后的组成。",
        "{subject}的拆解结果是哪个结构式？",
        "把{subject}拆解出的组成关系写出来。",
        "{subject}内部的部件是怎么组合的？拆解后用结构式回答。",
        "给出{subject}的部件拆解表示。",
        "{subject}怎么拆解？",
        "用结构式写一下{subject}拆解后的组成。",
        "{subject}的部件怎么组合的？拆解后给个结构式。",
        "把{subject}拆解成部件，写成结构式。",
        "{subject}的拆解式是什么？",
        "{subject}的结构拆解式怎么表示？",
    ),
    "T4_reading": (
        "{subject}怎么读？",
        "{subject}的读音是什么？",
        "请写出{subject}的拼音。",
        "{subject}念作什么？",
        "{subject}的读法是什么？",
        "这个字{subject}的读音是什么？",
        "给{subject}标个读音，怎么标？",
        "{subject}的拼音怎么写？",
        "请给出{subject}的读音。",
        "{subject}的发音是什么？",
        "{subject}读作什么？",
        "写出{subject}的读音。",
        "{subject}的读音怎么注？",
        "遇到{subject}这个字，读音该怎么念？",
        "{subject}的读音标注是？",
        "请拼出{subject}的读音。",
        "{subject}用拼音怎么写？",
        "{subject}这个字念作什么？",
        "{subject}的正确读法是什么？",
        "把{subject}的读音写出来。",
        "{subject}这个音怎么读？",
        "{subject}读起来是什么读音？",
        "{subject}念作哪个音？",
        "请写出{subject}的读音标注。",
        "{subject}怎么读才对？",
        "读{subject}这个字，读音是什么？",
        "{subject}的拼音给我。",
        "{subject}念起来是什么读音？",
        "{subject}的拼音是什么？",
        "{subject}怎么发音？",
    ),
    "T5_standard_form": (
        "{subject}的标准字形是哪一个？",
        "请写出{subject}的正体写法。",
        "{subject}的规范写法是什么？",
        "{subject}的标准写法是？",
        "{subject}对应哪个正体字？",
        "把{subject}写成规范字形，写哪个？",
        "{subject}的正字是哪个？",
        "{subject}的规范字形怎么写？",
        "给出{subject}的标准字形。",
        "{subject}的标准字形应该是哪个字？",
        "{subject}的正体是哪一个字？",
        "请给出{subject}的规范写法。",
        "{subject}对应的正体字是哪个？",
        "{subject}的标准写法是哪个字？",
        "{subject}的通用字形是哪一个？",
        "{subject}的正体字要写成哪个？",
        "{subject}该用哪个字形？",
        "请写出{subject}的标准字形。",
        "{subject}的规范写法是哪个字？",
        "把{subject}转成标准字形，结果是哪个字？",
        "{subject}的正规写法是哪个字？",
        "{subject}对应的正字是哪个？",
        "{subject}的正式写法是什么？",
        "请给出{subject}的正体字形。",
        "{subject}的正体是哪个？",
        "写成规范写法，{subject}该写哪个？",
        "{subject}的标准字形给我。",
        "{subject}的正体字是哪个？",
        "{subject}的规范字形是哪个字？",
        "把{subject}写成正体，写什么？",
    ),
    "T5_variant": (
        "{subject}的异体字是哪一个？",
        "请写出{subject}的异体写法。",
        "{subject}的别体是什么？",
        "{subject}还有哪个异体写法？",
        "写出{subject}的异体字。",
        "{subject}的异体是什么？",
        "{subject}有哪些异体写法？写出一个。",
        "{subject}的另一个异体写法是什么？",
        "请给出{subject}的一个异体字。",
        "{subject}的异体字形是哪个？",
        "{subject}写成异体是哪个字？",
        "{subject}的异体形式是什么？",
        "{subject}的异体字要写成哪个？",
        "{subject}还能写成哪个别体？",
        "把{subject}写成异体字，写哪个？",
        "{subject}的异体字形怎么写？",
        "{subject}有异体写法吗？写出一个。",
        "{subject}对应的异体字是？",
        "请写出{subject}的一个别体。",
        "{subject}的异体写法是哪个字？",
        "{subject}还能写成哪个异体字？",
        "给出{subject}的异体写法。",
        "{subject}的异体字字形是哪个？",
        "{subject}有哪些别体？写出一个。",
        "{subject}的异体给一个。",
        "{subject}的异体字有哪个？",
        "{subject}有没有别体？写一个。",
        "{subject}的异体字写法给我。",
        "把{subject}写成异体，写哪个字？",
        "{subject}的另一种异体写法是什么？",
    ),
    "T6_zh_ref": (
        "《{must}》里{subject}在第几页？",
        "《{must}》收录{subject}的页码是多少？",
        "查《{must}》，{subject}在哪一页？",
        "《{must}》把{subject}收在哪一页？",
        "翻《{must}》，{subject}的页码是几？",
        "《{must}》中{subject}出现在第几页？",
        "{subject}在《{must}》的第几页？",
        "请查《{must}》，给出{subject}的页码。",
        "《{must}》里{subject}被编在第几页？",
        "{subject}收在《{must}》的哪一页？",
        "在《{must}》中查{subject}，页码是多少？",
        "《{must}》给{subject}分配的是第几页？",
        "要查{subject}，该翻《{must}》的第几页？",
        "《{must}》里{subject}的页码是？",
        "请给出{subject}在《{must}》中的页码。",
        "{subject}在《{must}》里排在第几页？",
        "《{must}》上{subject}的页码是多少？",
        "翻开《{must}》，{subject}在第几页？",
        "《{must}》把{subject}放在第几页？",
        "查一下《{must}》里{subject}的页码。",
        "{subject}的页码在《{must}》里是几？",
        "《{must}》中{subject}的收录页码是？",
        "《{must}》中{subject}对应的页码是几？",
        "{subject}在《{must}》哪一页能查到？",
        "{must}的第几页有{subject}？",
        "在{must}中查{subject}，翻到第几页？",
        "{must}这本书里{subject}的页码是几？",
        "在{must}里查{subject}，它在第几页？",
        "{subject}在{must}里是第几页？",
        "帮我查一下{must}，{subject}的页码是多少？",
    ),
        "T8_constraint": (
        "有一个字满足这些条件：{conditions}。它是什么字？",
        "下面这些条件能定出哪个字：{conditions}？",
        "什么字符合这些条件：{conditions}？",
        "哪一个汉字满足{conditions}？",
        "根据这些条件找出符合条件的字：{conditions}。",
        "已知{conditions}，求满足条件的字。",
        "条件就是这些：{conditions}。对应的是哪个字？",
        "{conditions}——这是哪个字？",
        "要找一个字，它满足：{conditions}。是哪个字？",
        "给定这些条件：{conditions}。答案是什么字？",
        "从这些条件出发，能推出哪个字：{conditions}？",
        "同时满足{conditions}的字是哪一个？",
        "符合条件的字是什么：{conditions}。",
        "这几条线索说的是同一个字：{conditions}。它是哪个？",
        "线索是{conditions}，请说出符合的字。",
        "下面描述的汉字是哪一个：{conditions}。",
        "满足{conditions}的汉字是什么？",
        "限定条件就是这些：{conditions}。符合的是哪个字？",
        "已知{conditions}，可以确定哪个字？",
        "请根据这些条件说出满足要求的字：{conditions}。",
        "这些特征指向哪个字：{conditions}？",
        "一个字满足{conditions}，它是什么？",
        "已知{conditions}，猜出满足条件的字。",
        "哪些字同时满足{conditions}？给出答案。",
        "{conditions}，是哪个字？",
        "满足{conditions}的是哪个字？",
        "这字满足{conditions}，它是谁？",
        "符合{conditions}的是什么字？",
        "{conditions}——猜一个字。",
        "有这样一个字，{conditions}，请说出它。",
        "只知道{conditions}，能确定哪个字？",
        "把这些条件拼起来：{conditions}。得到的是哪个字？",
        "有字如此：{conditions}。是哪一个？",
        "满足{conditions}的那个字是什么？",
        "请说出同时满足{conditions}的字。",
        "如果{conditions}，那么这个字是哪一个？",
        "根据{conditions}，符合的字是哪一个？",
        "同时具备{conditions}的汉字是哪一个？",
        "请找出满足{conditions}的字。",
        "符合这些条件的字是什么：{conditions}？",
        "哪一个字同时符合{conditions}？",
        "这些条件描述的那个字是什么：{conditions}？",
        "要猜的字满足{conditions}，它是哪个？",
        "请说出符合这些条件的汉字：{conditions}。",
        "哪一个汉字同时具备{conditions}？",
        "满足{conditions}的字请写出来，是哪一个？",
        "已知{conditions}，符合条件的字是哪个？",
        "符合{conditions}的是哪个字，请回答。",
        "同时满足这些条件的字是什么：{conditions}？",
        "有这么一个字，它满足{conditions}，是哪个？",
        "请把满足{conditions}的字说出来。",
        "按这些条件来找：{conditions}。哪个字符合？",
        "哪一个字的特征是这些：{conditions}？",
        "具有这些特征的字是哪一个：{conditions}？",
        "已知条件：{conditions}。对应的字是什么？",
        "请根据这些线索找出符合条件的字：{conditions}。",
        "哪个字具备这些条件：{conditions}？",
        "符合{conditions}的字，请给出答案。",
        "有字满足{conditions}，请说出是哪一个。",
        "从这些条件能推出哪个字：{conditions}？",
        "满足{conditions}的字，你能说出来吗？",
        "哪一个字满足这些特征：{conditions}？",
        "这些条件指向哪个字：{conditions}？",
        "请挑出同时满足{conditions}的字。",
        "已知{conditions}，符合的是哪个汉字？",
        "有哪个字满足{conditions}？",
        "请说出同时符合{conditions}的汉字。",
        "同时符合{conditions}的字是什么？",
        "要定出这个字，条件就是这些：{conditions}。它是什么？",
        "请找出符合{conditions}的汉字。",
        "哪一个汉字满足这些条件：{conditions}？",
        "满足{conditions}的字，请给出。",
        "有字满足{conditions}，是哪一个？",
        "按这些条件猜字：{conditions}。答案是哪个字？",
        "请回答符合{conditions}的汉字是哪个。",
        "条件就是这些：{conditions}。能确定的是哪个字？",
        "请说出满足这个条件的字：{conditions}。",
        "同时满足{conditions}的汉字是什么？",
        "哪一个字符合这些条件：{conditions}？",
        "满足这些条件的汉字是哪个：{conditions}？",
        "有字如此，{conditions}，它是什么？",
        "请找出同时符合{conditions}的那个字。",
        "符合{conditions}的汉字是哪一个？",
        "从这些线索看：{conditions}。是哪个字？",
        "请说出这些条件对应的是什么字：{conditions}。",
        "哪个字的条件就是这些：{conditions}？",
        "满足{conditions}的那个汉字是什么？",
        "请挑出符合这些条件的字：{conditions}。",
        "有字满足这些：{conditions}。它是哪个？",
        "哪一个字同时具备这些特征：{conditions}？",
        "已知{conditions}，请说出满足的字。",
        "符合这些条件的是哪一个字：{conditions}？",
        "请说出同时满足这些条件的字：{conditions}。",
        "满足{conditions}的答案是什么字？",
        "有哪个汉字符合{conditions}？",
        "请根据这些条件找出符合的汉字：{conditions}。",
        "同时具备{conditions}的字是哪一个？",
        "满足{conditions}的字，请回答是哪个。",
        "请说出符合{conditions}的那个汉字。",
        "条件就是这些：{conditions}。符合的是哪个字？",
        "给你一个条件：{conditions}。满足它的字是哪个？",
        "有一个字，{conditions}。你猜是哪一个？",
        "按这两个特征找字：{conditions}。符合的是哪一个？",
        "若{conditions}，则这个字是哪一个？",
        "同时符合{conditions}的字是哪一个？",
        "请你根据这些条件确定一个字：{conditions}。",
        "在汉字里，满足{conditions}的字是哪一个？",
        "把这些当作条件：{conditions}。符合的字是哪一个？",
        "从字表里找出符合{conditions}的字。",
        "以这些为线索：{conditions}。符合的字是哪一个？",
        "符合这个条件的字是哪一个：{conditions}？",
        "筛选条件如下：{conditions}。剩下的是哪个字？",
        "想找一个满足{conditions}的字，是哪一个？",
        "请说出一个符合{conditions}的字。",
        "满足{conditions}的汉字，你能说出是哪一个吗？",
        "依这些条件：{conditions}。符合的是哪一个字？",
        "请找出具备{conditions}的那一个字。",
        "同时满足这两个条件的字是哪一个：{conditions}？",
        "哪一个字能同时满足{conditions}？",
        "请说出符合{conditions}的字。",
        "以这些为依据：{conditions}。符合的字是哪一个？",
        "由这些特征能找出哪个字：{conditions}？",
        "满足{conditions}的字，请说出是哪一个。",
        "请你找出符合{conditions}的那个字。",
        "条件全部对上的那一个汉字是哪个：{conditions}？",
        "根据这两个特征：{conditions}。符合的字是哪一个？",
        "请说出一个满足{conditions}的汉字。",
        "哪一个汉字能同时符合{conditions}？",
        "从这两个条件看：{conditions}。符合的是哪一个字？",
        "若满足{conditions}，这个字是哪一个？",
    ),

    # T9 的骨架**不许把「部件」直接拼在 {path} 后面**：位置标签里有一半本身就以「部件」结尾
    # （包围部件 / 被包围部件 / 右包围部件 / 左下包围部件），拼出来是「包围部件部件」——
    # 实测全量 67,551 条里 809 条是这种病句，另有 2,254 条写成「沿包围部件取㐆的部件」、
    # 2,260 条写成「上边取「㐇」的部件」、5,753 条写成没有问号的「问㐀的上边部件。」。
    # 所以这里一律用「{subject}的{path}」当名词短语（「㐆的包围部件」「㐀的第 3 个部件」），
    # 需要链式读法时用「沿{path}…下去」。
    "T9_component_chain": (
        "{subject}的{path}那个部件是哪个字？",
        "取{subject}的{path}，取出来的是哪个字？",
        "从{subject}出发，沿{path}一路找下去，最里层是哪个字？",
        "{subject}沿{path}走到底，是哪个字？",
        "把{subject}顺着{path}一层层拆下去，最后拆出哪个字？",
        "{subject}的{path}是什么字？",
        "沿{path}把{subject}拆开，取到的是哪个字？",
        "在{subject}的{path}上，是哪个字？",
        "{subject}的{path}里是哪个字？",
        "{subject}当中，{path}那个位置写的是什么字？",
        "{subject}的{path}能拆出哪个字？",
        "把{subject}的{path}拿出来，是哪个字？",
        "看{subject}的{path}，那里是哪个字？",
        "{subject}的{path}用的是哪个部件？",
        "{subject}的{path}位置上是哪个字？",
        "顺着{path}拆{subject}，得到哪个字？",
        "{subject}的{path}，拆出来是哪个字？",
        "{subject}的{path}那一块是什么字？",
        "从{subject}的{path}取部件，取到哪个字？",
        "请说出{subject}的{path}是哪个字。",
        "{subject}的{path}这个部件是哪个字？",
        "{subject}的{path}对应的是哪个字？",
        "{subject}的{path}拆解后是哪个字？",
        "{subject}的{path}写出来是哪个字？",
        "把{subject}沿{path}拆一层，是哪个字？",
        "{subject}的{path}落在哪个字上？",
        "{subject}的{path}对应的是什么字？",
        "{subject}的{path}上是哪个字？",
        "沿{path}看{subject}，看到的是哪个字？",
        "{subject}的{path}取出来是哪个字？",
    ),
    "T10_odd_one_out": (
        "选出部首与其它几个不同的那一个。",
        "哪一个字的部首和别的不同？",
        "这几个里，哪个字的部首与其它不同？",
        "选出部首有别于其它几个的字。",
        "下面几个字中，哪一个的部首是独一无二的？",
        "挑出部首跟其它几个不一样的字。",
        "哪一项的部首与其它项不同？",
        "找出部首与众不同的那个字。",
        "这些字里，哪个的部首和其它几个不属同一类？",
        "请选出部首最特殊的那一个字。",
        "从下面挑出部首不同的字。",
        "哪一个字的部首和其它几个对不上？",
        "哪一项的部首与别的不一样？",
        "找出部首与其余各项不同的字。",
        "下面哪一个字的部首最特别？",
        "请指出部首不同的那个字。",
        "这些候选中，谁的部首和别的不一样？",
        "哪个字的部首是这几个里的异类？",
        "选出部首与其它不一致的字。",
        "哪一项的部首与其它不同？",
        "找出那一个部首不同的字。",
        "这几个字里，哪一个的部首最为特殊？",
        "请选出部首异于其它各项的字。",
        "下面哪一项的部首与其它不同？",
        "哪个字的部首和别的不一样？",
        "找出部首不同的那个。",
        "部首异类的那个是哪个字？",
        "下面哪个字的部首最不寻常？",
        "谁的部首跟其它几个不同？",
        "挑出部首不一样的字。",
    ),
    "T11_stroke_extreme": (
        "哪一个字的笔画{direction_word}？",
        "请选出笔画数{direction_word}的那一个。",
        "这几个字里，哪个的笔画{direction_word}？",
        "挑出笔画{direction_word}的字。",
        "下面哪个字的笔画数{direction_word}？",
        "哪一项的笔画{direction_word}？",
        "找出笔画{direction_word}的那个字。",
        "这些字中，谁的笔画{direction_word}？",
        "请指出笔画{direction_word}的字。",
        "哪个字的笔画数在候选里{direction_word}？",
        "从下面挑出笔画{direction_word}的字。",
        "哪一项的笔画数{direction_word}？",
        "在这几个字里，笔画{direction_word}的是哪个？",
        "请选出笔画{direction_word}的一项。",
        "哪个候选的笔画{direction_word}？",
        "这几个字比一比，笔画{direction_word}的是谁？",
        "找出笔画数{direction_word}的那一项。",
        "哪一项的笔画在候选里{direction_word}？",
        "请挑出笔画{direction_word}的那一个字。",
        "这些候选里，笔画{direction_word}的是哪一个？",
        "谁的笔画{direction_word}，请指出来。",
        "从候选里选出笔画{direction_word}的字。",
        "哪一个字的笔画排在{direction_word}的一头？",
        "请给出笔画{direction_word}的那一项。",
        "哪个字笔画{direction_word}？",
        "笔画{direction_word}的是哪一个？",
        "谁的字形笔画{direction_word}？",
        "选出笔画{direction_word}的字。",
        "这几个字里笔画{direction_word}的是谁？",
        "笔画{direction_word}的那个，请指出。",
    ),
    "T12_ids_transform": (
        "从{subject}出发，{edits}，得到的是哪个字？",
        "以{subject}为起点，{edits}，最后得到什么字？",
        "{subject}照着步骤做下来：{edits}，结果是哪个字？",
        "从{subject}开始，{edits}，得到的是哪个字？",
        "把{subject}按步骤改：{edits}。改完是哪个字？",
        "{subject}经过这样的操作：{edits}，会变成什么字？",
        "先取{subject}，再{edits}，得到哪个字？",
        "对{subject}做一次改动：{edits}。结果是什么字？",
        "{subject}→{edits}→？请补上得到的字。",
        "如果{edits}，那么{subject}会变成哪个字？",
        "操作步骤是{edits}，起点是{subject}，终点是哪个字？",
        "{subject}经此一变：{edits}，成了哪个字？",
        "把{subject}这样处理：{edits}。得到的字是？",
        "起点{subject}，动作是{edits}，终点字是哪个？",
        "{subject}动一次手术：{edits}，变成哪个字？",
        "按这个变换走：{edits}。从{subject}出发会到哪个字？",
        "{subject}的部件替换后，{edits}，会变成哪个字？",
        "已知变换是{edits}，对{subject}做这个变换，结果是？",
        "从{subject}出发执行这个变换：{edits}。落点是哪个字？",
        "把{subject}的部件调整一下：{edits}，结果字是？",
        "{subject}按规则变形：{edits}，变成什么字？",
        "起点字{subject}，规则是这些：{edits}。结果字是哪个？",
        "经过这样的变换：{edits}。{subject}变成了哪个字？",
        "请对{subject}执行：{edits}，并写出得到的字。",
        "{subject}换一下：{edits}，变成什么字？",
        "{edits}，{subject}会变成哪个字？",
        "把{subject}改一下：{edits}。得到什么？",
        "{subject}→{edits}，结果是？",
        "按步骤改{subject}：{edits}。变成哪个字？",
        "{subject}经过这样的变换：{edits}。出来的是哪个字？",
    ),
    "T14_unicode_codepoint": (
        "{subject}的码位是多少？",
        "{subject}的码点是什么？",
        "写出{subject}的 Unicode 码位。",
        "{subject}的编码是多少？",
        "{subject}的 Unicode 码位是什么（十六进制或等值十进制都行）？",
        "{subject}在 Unicode 里的码位是几？",
        "请给出{subject}的码点。",
        "{subject}对应的码位是多少？",
        "查一下{subject}的 Unicode 码位。",
        "{subject}的码位编号是什么？",
        "请写出{subject}对应的码点值。",
        "{subject}的 Unicode 编号是多少？",
        "{subject}在 Unicode 码表里占哪个位置？",
        "{subject}的码点是多少（十六进制或等值十进制都行）？",
        "给出{subject}的 Unicode 码位。",
        "{subject}的码位取值是什么？",
        "请报出{subject}的码点。",
        "{subject}对应的 Unicode 编号是几？",
        "{subject}的码位是什么值？",
        "{subject}在 Unicode 表中的码位是？",
        "请写出{subject}的码位编号。",
        "{subject}的 Unicode 码点是哪个值？",
        "{subject}的 Unicode 码点具体是多少？",
        "{subject}这个字的码位是几？",
        "{subject}的码位是几？",
        "{subject}在 Unicode 里编号是多少？",
        "{subject}的码点值是哪个？",
        "{subject}对应的码点是几？",
        "{subject}的 Unicode 编号给我。",
        "查一下{subject}的码点。",
    ),
    "T15_unicode_bytes": (
        "{subject}在 UTF-8 编码下的字节序列是什么？",
        "写出{subject}的 UTF-8 字节。",
        "{subject}用 UTF-8 编码后是哪几个字节？",
        "{subject}的 UTF-8 编码结果是什么？",
        "{subject}编码成 UTF-8 是哪些字节？",
        "请给出{subject}的 UTF-8 字节序列。",
        "{subject}的 UTF-8 字节是什么？",
        "把{subject}用 UTF-8 编码，得到什么字节？",
        "{subject}的 UTF-8 编码是哪个字节串？",
        "{subject}在 UTF-8 下的字节表示是什么？",
        "请写出{subject}编码后的字节序列（UTF-8）。",
        "{subject}转成 UTF-8 字节是什么？",
        "{subject}的 UTF-8 字节具体是哪些？",
        "{subject}的 UTF-8 字节序列怎么写？",
        "{subject}按 UTF-8 编码得到哪些字节？",
        "给出{subject}的 UTF-8 编码结果。",
        "{subject}的 UTF-8 字节串是？",
        "请报出{subject}的 UTF-8 字节序列。",
        "{subject}在 UTF-8 里的字节是什么？",
        "{subject}编码为 UTF-8 后是哪几个字节？",
        "{subject}的 UTF-8 表示是什么？",
        "把{subject}编成 UTF-8，字节序列是？",
        "{subject}编码成 UTF-8 后字节串怎么写？",
        "请写出{subject}在 UTF-8 下的字节。",
        "{subject}的 UTF-8 字节给我。",
        "{subject}编成 UTF-8 是什么？",
        "{subject}的 UTF-8 编码是哪几个字节？",
        "{subject}的 UTF-8 编码结果有哪些？",
        "{subject}在 UTF-8 下是什么字节？",
        "把{subject}编成 UTF-8，字节是什么？",
    ),
    "T16_unicode_reverse": (
        "码位{subject}对应哪一个字？",
        "码点{subject}是哪个字？",
        "哪一个字对应码位{subject}？",
        "码位{subject}对应的字是什么？",
        "{subject}这个码位上是哪个字？",
        "请找出码位{subject}对应的字。",
        "码位{subject}在字表里是哪个字？",
        "哪个字的码位是{subject}？",
        "对应码位{subject}的汉字是哪一个？",
        "码位{subject}指向哪个字？",
        "请从下面挑出码位为{subject}的字。",
        "码位{subject}上的汉字是哪个？",
        "哪一个字的码点等于{subject}？",
        "码位{subject}是哪个汉字的编号？",
        "请指出码位{subject}对应的汉字。",
        "码位{subject}对应的汉字是什么？",
        "哪一个候选的码位是{subject}？",
        "码点{subject}对应的是哪个字？",
        "在下面几个字里，哪个的码位是{subject}？",
        "码位{subject}是哪一个字的编号？",
        "请选出码位{subject}对应的字。",
        "哪个字写在码位{subject}上？",
        "码位{subject}对应哪个汉字？",
        "哪一项的码点是{subject}？",
        "码位{subject}是哪个字？",
        "{subject}对应哪个字？",
        "按{subject}去找，能找到哪个字？",
        "码位{subject}——是哪个字？",
        "查码位{subject}，对应哪个字？",
        "请说出码位{subject}对应的字。",
    ),
}

# 解析核心：每条都必须**原样包含答案**。{answer_q} 是「答案加引号」的写法，
# 只有答案是单个汉字时才带引号（见 `_context`）。
_EXPLANATIONS: Mapping[str, tuple[str, ...]] = {
    "T1_stroke_total": (
        "{subject}一共 {answer} 画。",
        "{subject}的笔画数是 {answer}。",
        "数下来，{subject}是 {answer} 画。",
        "{subject}数出来是 {answer} 画。",
    ),
    "T3_ids_forward": (
        "{subject}的结构式是 {answer}。",
        "把{subject}拆解开，得到 {answer}。",
        "{subject}的拆解结果是 {answer}。",
        "{subject}可以拆成 {answer}。",
    ),
    "T4_reading": (
        "{subject}读作 {answer}{reading_note}。",
        "{subject}的读音是 {answer}{reading_note}。",
        "{subject}念 {answer}{reading_note}。",
        "{subject}的拼音写作 {answer}{reading_note}。",
    ),
    "T5_standard_form": (
        "{subject}的标准字形是{answer_q}。",
        "{subject}的正体写作{answer_q}。",
        "{subject}的规范写法是{answer_q}。",
    ),
    "T5_variant": (
        "{subject}的异体字是{answer_q}。",
        "{subject}的异体写作{answer_q}。",
        "{subject}的别体是{answer_q}。",
    ),
    "T6_zh_ref": (
        "《{must}》把{subject}收在第 {answer} 页。",
        "《{must}》里{subject}在第 {answer} 页。",
        "{subject}在《{must}》的第 {answer} 页。",
    ),
        "T8_constraint": (
        "满足这些条件的字是{answer_q}。",
        "这些条件定出的字是{answer_q}。",
        "符合条件的是{answer_q}。",
    ),
    "T9_component_chain": (
        "取{subject}的{path}，得到{answer_q}。",
        "{subject}的{path}是{answer_q}。",
        "沿着{path}找下去，那个部件是{answer_q}。",
        "沿{path}拆到底，拆出来的字是{answer_q}。",
    ),
    "T10_odd_one_out": (
        "{answer_q}的部首和其它几个不一样。",
        "其它几个的部首相同，只有{answer_q}不同。",
        "部首与其它不同的那个是{answer_q}。",
    ),
    "T11_stroke_extreme": (
        "{answer_q}的笔画数是 {answer_strokes}，在候选里{direction_word}。",
        "候选里{answer_q}是 {answer_strokes} 画，笔画{direction_word}。",
        "数下来，{answer_q}有 {answer_strokes} 画，在候选里{direction_word}。",
    ),
    "T12_ids_transform": (
        "按步骤替换后得到{answer_q}。",
        "执行完这些替换，结果是{answer_q}。",
        "一步步做下来，最后得到{answer_q}。",
    ),
    "T14_unicode_codepoint": (
        "{subject}的码位是 {answer}{codepoint_note}。",
        "{subject}的码点是 {answer}{codepoint_note}。",
        "{subject}的 Unicode 码位是 {answer}{codepoint_note}。",
    ),
    "T15_unicode_bytes": (
        "{subject}的 UTF-8 编码是 {answer}{utf8_note}。",
        "{subject}编码成 UTF-8 是 {answer}{utf8_note}。",
        "{subject}的 UTF-8 字节是 {answer}{utf8_note}。",
    ),
    "T16_unicode_reverse": (
        "码位{subject}对应的字是{answer_q}。",
        "{subject}这个码位对应{answer_q}。",
        "这个码位对应{answer_q}。",
    ),
}

_DIRECTION_WORDS = {"fewest": "最少", "most": "最多"}

# 汉字码位区间：基本区、扩展 A、兼容区、扩展 B 及以上（都是单码位汉字）。
_HANZI_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x3FFFF))


def _is_hanzi(text: str) -> bool:
    """输入：任意字符串。输出：它是不是单个汉字。
    预期：引号只加在汉字上——给数字、拼音、码位加引号读着别扭。
    """
    if len(text) != 1:
        return False
    code = ord(text)
    return any(low <= code <= high for low, high in _HANZI_RANGES)


def _quote(text: str, style: tuple[str, str]) -> str:
    """输入：待引用文本 + 引号写法。输出：引用后的文本。
    预期：**只有单个汉字才加引号**，其余原样返回（非汉字自动退回裸写）。
    """
    left, right = style
    if not left or not _is_hanzi(text):
        return text
    return f"{left}{text}{right}"


# 上下文里**允许为空**的占位符：它们是句子的补充说明而不是骨架——主答案已经给出等价写法时，
# 这半句按设计就不出现（见 `_context` 里 reading_note / utf8_note / codepoint_note 的取空条件）。
_OPTIONAL_PLACEHOLDERS: frozenset[str] = frozenset({"reading_note", "utf8_note", "codepoint_note"})


class _EmptyPlaceholder(ValueError):
    """输入：被模板引用、但值为空的占位符名。输出：异常。
    预期：只在 `render` 内部传递「这句渲染不出」，不往外抛——空值不是数据错误，
          是这条记录 / 这个变体本来就不该出题。
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _Context(dict):
    """输入：`_context` 产出的占位符表。输出：给 `str.format_map` 用的映射。
    预期：模板**实际用到**的占位符若为空串，抛 `_EmptyPlaceholder` 让 `render` 判 None——
          没被用到的空值不影响（`_OPTIONAL_PLACEHOLDERS` 里的补充说明一律允许为空）。
          这样「句子缺主语」这种垃圾在渲染这一步就被拦下，不会走到闸门后面去。
    """

    def __getitem__(self, key: str) -> str:
        value = super().__getitem__(key)
        if not str(value).strip() and key not in _OPTIONAL_PLACEHOLDERS:
            raise _EmptyPlaceholder(key)
        return value


def _context(spec: TaskSpec, quote: tuple[str, str], sep: str) -> _Context:
    """输入：TaskSpec + 引号写法 + 候选分隔符。输出：句式填充用的上下文。
    预期：每个占位符都有值，且**全部来自 spec**；某个骨架槽位取到空串时，`_Context` 会在
          模板真正用到它的那一刻抛 `_EmptyPlaceholder`，`render` 据此判 None。
    """
    facts: Mapping[str, Any] = spec.facts or {}
    readings = facts.get("reading_display") or []
    if isinstance(readings, str):
        readings = [readings]
    direction = str(facts.get("direction", ""))
    answer = str(spec.answer)
    # 下面三个注解都是「另一种等价写法」：与主答案相同时留空，
    # 免得渲染出「读作 xi3，拼音记作 xi3」这种同一个值说两遍的句子。
    reading_note = ""
    for item in readings:
        if str(item) != answer:
            reading_note = f"（也写作 {item}）"
            break
    utf8_bytes = str(facts.get("utf8_bytes", ""))
    utf8_note = f"（也写作 {utf8_bytes}）" if utf8_bytes and utf8_bytes != answer else ""
    codepoint_forms = [str(item) for item in (facts.get("codepoint_forms") or [])]
    codepoint_hex = codepoint_forms[0] if codepoint_forms else ""
    bare_hex = codepoint_hex[2:] if codepoint_hex.startswith("U+") else codepoint_hex
    # 「码位是 U+13312」是错的——13312 是十进制写法。只有答案本身不是十六进制写法时，
    # 才补一句「即 U+XXXX」把规范写法给出来。
    codepoint_note = (
        f"，即 {codepoint_hex}" if codepoint_hex and answer not in {codepoint_hex, bare_hex} else ""
    )
    return _Context({
        "subject": _quote(spec.visible_subject, quote),
        "must": "、".join(spec.must_contain),
        "options": sep.join(_quote(item, quote) for item in spec.options),
        "path": "、".join(spec.must_contain),
        "conditions": str(facts.get("conditions", "")),
        "direction_word": _DIRECTION_WORDS.get(direction, ""),
        "answer": answer,
        "answer_q": _quote(answer, quote),
        "answer_strokes": str((facts.get("strokes") or {}).get(answer, "")),
        "utf8_note": utf8_note,
        "reading_note": reading_note,
        "codepoint_note": codepoint_note,
        "edits": "；".join(str(item) for item in (facts.get("edits") or [])),
    })


# 离线模板池**不覆盖**的题型：释义类题（T13 靠释义认字 / T17 问字义）。
# 输入：无。输出：题型 ID 集合。
# 预期：这是「模板池里没有骨架」的那批题型的**唯一**来源——`has_templates` 与它必须一致，
#      `src/tests/test_render.py` 逐条钉住，防止哪天有人把骨架加回来却忘了这条纪律。
LLM_ONLY_TASK_IDS: frozenset[str] = frozenset({"T13_ids_reverse", "T17_meaning_gloss"})


def has_templates(task_id: str) -> bool:
    """输入：题型 ID。输出：离线模板池能不能渲染这一型（骨架与解析都在才算）。
    预期：调用方（`render.generate` / `render.stats`）靠它把释义类题整型排除——
          不排的话它们会被算进 `unrenderable`，看着像「渲染失败」而不是「按纪律不走这条路」。
    """
    return bool(_CORES.get(task_id)) and bool(_EXPLANATIONS.get(task_id))


def _stem_count(task_id: str, has_options: bool) -> int:
    """输入：题型 ID + 这题有没有候选。输出：**题干**的组合空间大小（不含解析写法）。
    预期：骨架 × 引号 × 候选呈现 × 候选分隔 × 开场 × 收尾；没有候选的题不乘候选那两段
          （它们根本不参与渲染，乘进去会把空间报大）。`stem_space` 与 `axes` 共用这一份算式——
          各写一遍的话「统计口径」与「渲染口径」迟早会漂（旧报告里 `space=2400` 却写着
          「不同解析 16」，正是两边各算各的留下的）。
    """
    cores = len(_CORES.get(task_id, ()))
    if not cores:
        return 0
    option_part = len(_OPTION_STYLES) * len(_LIST_STYLES) if has_options else 1
    return cores * len(_QUOTES) * option_part * len(_OPENERS) * len(_TAILS)


def stem_space(spec: TaskSpec) -> int:
    """输入：TaskSpec。输出：**题干**的组合空间大小（不含解析写法）。
    预期：单列出来是因为「答案有没有泄进题干」只由题干决定：`generate._render_one` 找可渲染的
          句式时只走这一段，不必把解析的写法也乘进去重跑一遍。
    """
    return _stem_count(spec.task_id, bool(spec.options))


@dataclass(frozen=True)
class Axes:
    """输入：无。输出：一个 variant 拆出来的各轴下标。
    预期：`render` 按它取模板，统计侧（`render.stats.census` 与数据集句式扫描）按它把「不同句式」
          拆成「不同骨架 / 不同收尾 / 不同引号写法」——两边共用同一份拆法，不各拆各的。
    """

    core: int
    quote: int
    option_style: int
    sep: int
    opener: int
    tail: int
    explanation: int


def axes(task_id: str, variant: int, *, has_options: bool) -> Axes | None:
    """输入：题型 ID + variant 序号 + 这题有没有候选。输出：各轴下标；题型没有模板时 None。
    预期：解析写法占**高位**（最慢变化），低位依次是 骨架 → 引号 → 候选呈现 → 候选分隔 →
          开场 → 收尾，与 `render` 的混合进制一致。variant 超出空间不报错——调用方本来就按
          环形取模，`%` 保证越界值落到某个合法下标上。
    """
    cores = _CORES.get(task_id, ())
    if not cores or not _EXPLANATIONS.get(task_id, ()):
        return None
    stem_count = _stem_count(task_id, has_options)
    if stem_count <= 0:
        return None
    explanation = variant // stem_count
    variant %= stem_count
    core = variant % len(cores)
    variant //= len(cores)
    quote = variant % len(_QUOTES)
    variant //= len(_QUOTES)
    option_style = 0
    sep = 0
    if has_options:
        option_style = variant % len(_OPTION_STYLES)
        variant //= len(_OPTION_STYLES)
        sep = variant % len(_LIST_STYLES)
        variant //= len(_LIST_STYLES)
    opener = variant % len(_OPENERS)
    variant //= len(_OPENERS)
    tail = variant % len(_TAILS)
    return Axes(
        core=core,
        quote=quote,
        option_style=option_style,
        sep=sep,
        opener=opener,
        tail=tail,
        explanation=explanation,
    )


def explanation_space(spec: TaskSpec) -> int:
    """输入：TaskSpec。输出：解析写法的组合数（每型解析骨架的条数）。
    预期：模板池里没有这型（释义类题）时返回 0，让 `variant_space` 也是 0。
    """
    if not _CORES.get(spec.task_id, ()):
        return 0
    return len(_EXPLANATIONS.get(spec.task_id, ()))


def variant_space(spec: TaskSpec) -> int:
    """输入：TaskSpec。输出：这道题能渲染出的 (题干, 解析) 组合总数。
    预期：题干空间 × 解析写法数。解析写法占**高位**（最慢变化），所以低位仍是题干：
          同一 (题型, 字) 换一次解析说法不会把题干一起换掉，顺着环形找句式时也只在同一段
          解析里走。这样接轴不会挪动任何一条已有的题干（`X % (n*k) % n == X % n`）。
    """
    return stem_space(spec) * explanation_space(spec)


def render(spec: TaskSpec, variant: int) -> tuple[str, str] | None:
    """输入：TaskSpec + variant 序号。输出：(stem, explanation)，无法渲染时 None。
    预期：variant 的**高位选解析写法、低位**按「骨架 × 引号 × 候选呈现 × 候选分隔 × 开场 ×
          收尾」混合进制拆开选题干，各段独立取样，所以相邻 variant 渲染出来的句子结构完全不同。
          非选择题的答案若落在题干里（泄题），返回 None 让调用方换句式；
          模板用到的占位符取到空串（例如没有字形的记录）同样返回 None——渲染不出的整条由调用方
          计入 `unrenderable` 并写进落选文件，不是静默丢弃。
    """
    cores = _CORES.get(spec.task_id, ())
    bodies = _EXPLANATIONS.get(spec.task_id, ())
    if not cores or not bodies:
        return None
    # 进制拆解只有一处（`axes`）：渲染与统计按同一份下标取模板，两边不会漂。
    picked = axes(spec.task_id, variant, has_options=bool(spec.options))
    if picked is None:
        return None
    core = cores[picked.core]
    quote = _QUOTES[picked.quote]
    option_style = _OPTION_STYLES[picked.option_style] if spec.options else ""
    sep = _LIST_STYLES[picked.sep] if spec.options else _LIST_STYLES[0]
    opener = _OPENERS[picked.opener]
    tail = _TAILS[picked.tail]

    context = _context(spec, quote, sep)
    try:
        parts = [opener + core.format_map(context)]
        if option_style:
            parts.append(option_style.format_map(context))
        if tail:
            parts.append(tail)
        stem = "".join(parts)
        # 答案在题干里出现几次算泄题：**与闸门 `no_answer_leak` 共用同一判据**
        # （`verify.checks.answer_leak_limit`），只是这里用来换句式而不是判废。T9 的答案常是
        # 「上」「一」这类字，题干里的方位词、语气词本来就可能带上它们；选择题的答案又必然
        # 出现在候选名单里，撞上模板用词（「从…里挑一个」撞答案「从」）时次数就超了。
        # 渲染完再被闸门拒，这条 (题型, 字) 就一条数据都出不来——实测 38 条被拒的记录里
        # 32 条本来只要换一条不撞答案的句式就能出（`synthesis/out/probe_leak_rescue.py`）。
        stem_norm = normalize_for_match(stem)
        if any(
            stem_norm.count(normalize_for_match(str(item))) > answer_leak_limit(spec, item)
            for item in spec.accepted
            if normalize_for_match(str(item))
        ):
            return None
        body = bodies[picked.explanation % len(bodies)]
        explanation = body.format_map(context)
    except (_EmptyPlaceholder, KeyError, IndexError):
        return None
    return stem, explanation


def render_variants(spec: TaskSpec, count: int) -> list[tuple[str, str]]:
    """输入：TaskSpec + 要渲染的变体数。输出：(stem, explanation) 列表（未去重）。
    预期：供句式多样性计数用；渲染不出来的变体（泄题那些）直接跳过。
    """
    out: list[tuple[str, str]] = []
    for variant in range(count):
        rendered = render(spec, variant)
        if rendered is not None:
            out.append(rendered)
    return out
