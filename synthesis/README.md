# 汉字合成数据管线

以 `crawler/` 抓下来的 `chars.jsonl` 为底座，合成汉字知识问答数据。

事实由程序从结构化字段构造，答案在出题那一刻就定死；LLM 只负责把这份事实说成人话，产出再逐条
校验。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

需要一份 `crawler/data/chars.jsonl`（用 `crawler/crawl_zisea.py chars` 抓，默认落在
`crawler/data/`，该目录不入库）。默认路径可用 `--data` 改。

## 配置

凭据放仓库根的 `secrets.toml`，只从文件读，不进环境变量，文件本身已在 `.gitignore` 里。

```toml
[[providers]]
name = "my-endpoint"
base_url = "https://example.com/v1"
model = "your-model"
api_key = "..."
role = "verbalizer"   # verbalizer = 措辞，judge = 裁判
rpm = 0               # 每分钟请求上限，0 = 不限速
batch = 1             # 一次请求处理几道题

[providers.extra_body]
# 关思考的字段名各家不一样，这里原样透传，写什么就发什么
reasoning_effort = "none"
```

可以写多个 `[[providers]]` 块，顺序就是优先级，前面的失败自动换后面的。任何兼容 OpenAI Chat
Completions 的端点都能接，不绑定某一家。

请求体里不设 `max_tokens`；模型吐的思考段一律剥离，不进产出。

## 命令

| 命令 | 作用 | 联网 |
|---|---|---|
| `list` | 列出已注册的 16 个题型 | 否 |
| `dry-run` | 只跑出题，不调模型 | 否 |
| `render` | 离线渲染题面与解析，不调模型 | 否 |
| `render-stats` | 遍历句式组合，统计每型能渲染出多少种不同题干 | 否 |
| `run` | 端到端合成 | 是 |
| `coverage` | 现有产出覆盖了字库里多少字 | 否 |
| `reject-stats` | 按拦截规则与题型汇总没通过的记录 | 否 |
| `attempt-stats` | 逐轮验证判定汇总 | 否 |
| `audit` | 把一批产出重过验证，组装最终数据集 | 否 |
| `dedup` | 找出高度相似的题干，只读 | 否 |
| `contamination` | 查题干有没有连续照抄外部语料 | 否 |
| `roundtrip` | 把产出交回模型，让它只凭题面答题 | 是 |
| `difficulty` | Rasch 校准，要有作答数据才做 | 否 |

```powershell
python -m hanzi_synth.cli list
python -m hanzi_synth.cli dry-run --limit 16
python -m hanzi_synth.cli render --limit 2000 --out synthesis/out/offline.jsonl
python -m hanzi_synth.cli run --limit 40 --workers 4 --out synthesis/out/batch.jsonl
python -m hanzi_synth.cli audit --in synthesis/out --out synthesis/out/dataset.jsonl
```

`run` 常用的开关：`--tasks` 选题型，`--hops` / `--constraints` / `--distractor-depth` 调难度，
`--workers` 并发数，`--retries` 失败重写次数，`--loop` 没通过的自动重跑轮数，`--judge` 开 LLM 裁判，
`--no-roundtrip` 关反向答题校验，`--exclude` 跳过已有产出，`--batch` 一次请求处理多道题。完整参数见
`--help`。

## 题型

| 题型 | 问什么 | 离线可渲染 |
|---|---|---|
| T1 `stroke_total` | 总笔画数 | 是 |
| T3 `ids_forward` | 给出 IDS 问是哪个字 | 是 |
| T4 `reading` | 读音 | 是 |
| T5 `standard_form` | 正体与异体 | 是 |
| T5 `variant` | 异体字 | 是 |
| T6 `zh_ref` | 中华字海的页码与序号 | 是 |
| T8 `constraint` | 同时满足多个条件的字 | 是 |
| T9 `component_chain` | 部件链上第 n 层的部件 | 是 |
| T10 `odd_one_out` | 候选里哪个不合群 | 是 |
| T11 `stroke_extreme` | 候选里笔画最多或最少的 | 是 |
| T12 `ids_transform` | IDS 替换一步后是哪个字 | 是 |
| T13 `ids_reverse` | 给读音和释义，问 IDS | 否，靠 LLM |
| T14 `unicode_codepoint` | Unicode 码位 | 是 |
| T15 `unicode_bytes` | UTF-8 编码字节 | 是 |
| T16 `unicode_reverse` | 码位反查是哪个字 | 是 |
| T17 `meaning_gloss` | 释义 | 否，靠 LLM |

T13 和 T17 不在离线渲染的覆盖范围内：T13 只靠读音和释义渲染出来的题，同音字太多，读者认不出
问的是哪个字；T17 的解析会把字典原文当答案抄出去。这两型只能靠 LLM 措辞。

## 产出

产出写在 `--out` 指定的文件里（默认目录 `synthesis/out/`，已 gitignore）。同目录还会出现几个
另外还会写几个文件：`<out>.rejects.jsonl` 没通过的记录、`<out>.attempts.jsonl` 逐轮校验结果、`<out>.coverage.json`
覆盖率、`<out>.lock` 锁定文件（防止两个进程同时写同一个产出）。`audit` 另外写 `<out>.dropped.jsonl` 和 `<out>.audit.json`。

## 测试

```powershell
python -m unittest discover -s src/tests -t src   # 511 个，不联网
```

## 许可

代码按 Apache License 2.0 分发，见仓库根 `LICENSE`。