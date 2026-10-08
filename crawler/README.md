# 字海网（新叶典）离线工具

把字海网的汉字数据抓到本地，离线浏览。这个目录只有工具代码，没有数据：抓下来的东西全部写在
`crawler/data/`，那个目录在 `.gitignore` 里，不入库、不随仓库分发。

## 站点已经全面改版

2026-09-01 新站上线，前端换成 Vue/Vite SPA，HTML 壳里没有内容，旧的 `/zscontent.asp?uni=`
接口全部失效，只能走 JSON API。旧版 `build_index.py`、`crawl_detail.py` 和 `chars/`、`idioms/`
产出已经删掉。

## 网络上的几个坑

- 必须用 `http://`。`https://zisea.com` 在当前网络下 TLS 握手会被干扰（curl 返回 35，
  schannel 收不到 handshake），`http://` 完全正常。
- 本地网络若开了 Clash TUN（fake-ip 模式），`zisea.com` 会解析成 `198.18.0.126`。别硬编码真实 IP，
  直接走域名。
- 站点真实 IP 备用值 `106.12.154.220`（腾讯云）。旧脚本里写死的 `106.12.68.185` 已失效。

## ZIID 编号

新站用统一的 6 位编号：一个字母加 5 位字符，前端正则是 `^[UZP][0-9A-Z]{5}$`。

| 系列 | 含义 | 条数 | 编号规则 |
|------|------|------|---------|
| `U` | Unicode 所见字符 | 103,368 | `U` + 5 位十六进制码位，如 `U04E00` 是 U+4E00「一」 |
| `Z` | 字海（中华字海）所见字形 | 23,623 | `Z` + 页码 3 位 + 序号 2 位，如 `Z43651` |
| `P` | 提案所见字形等 | 18,513 | `P` + 5 位，如 `P80004`，多数是没有 Unicode 码位的俗字 |

合计 145,504 个字符，和 `/api/stats` 的 `total_characters` 一致。

## API

| 端点 | 说明 |
|------|------|
| `GET /api/groups` | 全部系列、前缀桶与计数，枚举入口 |
| `GET /api/z-range?prefix=&page=&page_size=` | 某个前缀下的轻量字符列表，`page_size` 上限 500 |
| `GET /api/z/{ZIID}` | 单字完整详情 |
| `GET /api/unicode-range?start=&end=&page=&page_size=` | 按码位区间取轻量列表 |
| `GET /api/stats` | 全站统计，含 `daily` 增量线索 |
| `GET /api/sprite-version` | 精灵图版本号 |
| `GET /sprites/{ZIID}.png` | 字形图片，只有无 Unicode 码位的字才有，其余回 404 |

探明但没封装的端点：`/api/chengyu`（本工具不收成语）、
`/api/search/{direct,liangfen,ids,bushou,pinyin,reverse,duti,imgchars}`、`/api/radicals`、
`/api/pinyin-index`、`/api/pinyin-syllables`、`/api/liangfen-parts`、
`/api/books/{zhonghua,hanyu,tonggui,bafu}`。

错误约定：ZIID 格式错回 `400 Invalid ZIID`，格式对但没收录回 `404 Character not found`，
`page_size > 500` 回 `422`。

## 目录结构

```
crawler/
├── crawl_zisea.py          爬虫入口脚本，子命令 stats / enumerate / chars / sprites / verify / view
├── view_zisea.py           查看器入口脚本，等价于 crawl_zisea view
├── zisea/
│   ├── cli.py              子命令的参数解析
│   ├── client.py           HTTP 客户端（重试 / 限速 / UA）
│   ├── api.py              端点封装与分页迭代
│   ├── crawl.py            抓取编排（枚举 → 差异 → 并发抓取）
│   ├── sprites.py          无码位字的字形图片下载
│   ├── store.py            JSONL 追加写入 + 断点续爬
│   ├── robots.py           robots.txt 检查
│   ├── verify.py           完整性核对（站点计数 vs 本地产出）
│   └── viewer.py           离线查看器（本地网页界面）
└── tests/                  127 个用例，全部不联网
```

## 常用命令

```powershell
# 全站统计
python crawler/crawl_zisea.py stats --ignore-robots

# 只枚举 ZIID，不抓详情
python crawler/crawl_zisea.py enumerate --ignore-robots

# 增量抓全部字符详情，默认输出到 crawler/data/
python crawler/crawl_zisea.py chars --ignore-robots

# 试抓 200 条
python crawler/crawl_zisea.py chars --limit 200 --ignore-robots

# 下载无码位字的字形图片（增量）
python crawler/crawl_zisea.py sprites --ignore-robots

# 完整性核对
python crawler/crawl_zisea.py verify --ignore-robots

# 启动离线查看器 → http://127.0.0.1:8720
python crawler/view_zisea.py
```

查看器默认纯离线，本地缺精灵图时回 404；要回源取图得加 `--allow-origin-fetch`。

## 完整性核对

`verify` 拿站点的权威计数和本地产出逐项核对，只读、不写产出，一共 8 项：

| 检查 | 判定标准 |
|------|------|
| 详情条数 | `chars.jsonl` 条数 == `/api/stats.total_characters` |
| 重复 ZIID | 主键无重复 |
| 行格式与必需字段 | 无坏行、无缺 `property` / `meanings` |
| 各系列计数 | U / Z / P 逐系列 == `/api/stats.series` |
| 逐前缀计数 | 每个前缀桶（如 `U04`、`Z4`、`PA`）的本地计数 == `/api/groups` 的计数 |
| chars_index 与 chars 主键一致 | 两个文件的主键集合完全相同 |
| 有 IDS 的字数 | == `/api/stats.ids.with_ids` |
| 精灵图齐全 | 每个无码位字都有 `sprites/{ZIID}.png`，且没有多余的图 |

其中「逐前缀计数」这一项就足以证明没漏抓：`/api/groups` 是站点自己给的枚举入口，它把全站的字
切成 56 个前缀桶并给出每桶计数，只要 56 个桶的本地计数全部相等，本地集合就等于站点集合。
其余 7 项是交叉印证。

实测（2026-10）全绿：

```
[通过] 详情条数：本地 145504 条 / 站点 145504 条
[通过] 重复 ZIID：无重复
[通过] 行格式与必需字段：全部合法
[通过] 各系列计数：逐系列一致 {'U': 103368, 'Z': 23623, 'P': 18513}
[通过] 逐前缀计数（完整性核心）：56 个前缀桶全部对上
[通过] chars_index 与 chars 主键一致：两侧各 145504 个 ZIID 完全一致
[通过] 有 IDS 的字数：本地 113359 / 站点 113359
[通过] 精灵图齐全：42136 张全部就位
```

### 增量与断点续爬

1. 用 `/api/groups` 加 `/api/z-range` 轻量枚举出全站 ZIID，约 300 次请求覆盖 14.5 万条；
2. 和 `chars.jsonl` 已有的主键求差，只抓新增的 ZIID（详情请求才是重活）；
3. 和上一轮 `chars_index.jsonl` 对比，报告新增和下架的字。

中断后直接重跑就能续；进程被杀留下的半行会在启动时自动截断。「已完成集合」直接从
`chars.jsonl` 回读，不另存状态文件，免得两份状态漂移。

## 产出格式

| 文件 | 内容 |
|------|------|
| `chars.jsonl` | 每行一个字的完整详情，字段与 `/api/z/{ZIID}` 原样一致 |
| `chars_index.jsonl` | 全站轻量索引 `{ziid, prefix}`，用于增量对比 |
| `sprites/{ZIID}.png` | 无 Unicode 码位字的字形图片 |
| `viewer_index.json` | 查看器索引，可删，会自动重建 |
| `state.json` | 最近一次运行的统计和 `sprite-version` |
| `failed_ziids.txt` | 上一轮失败的 ZIID，全部成功时不存在 |

记录里不写时间戳，保持和 API 返回逐字节可比，方便做 diff。

## 单字记录长什么样

```jsonc
{
  "ziid": "U04E00",
  "goto_from": null,
  "property": {
    "ziid": "U04E00", "zi": "一", "stroke_total": 1, "radical": "一",
    "stroke_remaining": 0, "head": "一", "tail": "", "structure": 0,
    "ids": "#(H)(G),{一}#(T)(t)",     // 字形描述（IDS）
    "pinyin": "hg",                   // 叶码，站点自有的两分拼音编码，不是拼音
    "pinyin_alt": "yi", "head_py_display": "heng", "tail_py_display": "",
    "zh_ref": {"page": 1, "number": 1},   // 中华字海页码与序号
    "hd_ref": {...}, "tg_ref": {...}, "source_ref": "G0-523B,..."
  },
  "meanings": { "mean_h": ["㈠拼音yi1<br>①…"], "mean_z": [], … },
  "variants": [{"ziid": "U058F9", "zi": "壹"}],
  "prev": null, "next": "U04E59"
}
```

`property.pinyin` 是叶码（两分拼音编码），真实读音藏在 `meanings.mean_h` 的「拼音xx1」里，
查看器会把它抽出来做拼音检索。

## 测试

```powershell
# 已激活 .venv 时
python -m unittest discover -s crawler/tests -t crawler

# 没激活时，在仓库根目录跑
.\.venv\Scripts\python.exe -m unittest discover -s crawler/tests -t crawler
```

127 个用例全部不访问网络：HTTP 层用 `tests/fakes.py` 里的假 Session，查看器的 HTTP 测试绑在
`127.0.0.1:0` 随机端口上，robots 与 verify 的用例也全走假 session。实测 4.0 秒跑完。