# 字海网离线工具集

这个仓库有两个部分：

- `crawler/` —— 把字海网（新叶典）的字形数据抓到本地、离线浏览的爬虫，产出写在 `crawler/data/`。
- `synthesis/` —— 拿抓下来的 `chars.jsonl` 当底座，合成汉字问答数据的管线，代码在 `src/hanzi_synth/`。

仓库里只有代码和文档，没有任何抓取到的数据。

## 快速开始

```powershell
# 建虚拟环境（Python 3.11+）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .            # 装 src/hanzi_synth，之后可以直接 python -m hanzi_synth.cli

# 离线浏览 → http://127.0.0.1:8720，只读本地数据，不联网
python crawler/view_zisea.py

# 爬虫测试：127 个，不联网
python -m unittest discover -s crawler/tests -t crawler

# 合成管线
python -m hanzi_synth.cli list                        # 列出 16 个题型
python -m hanzi_synth.cli dry-run --limit 16          # 只出题，不联网
python -m hanzi_synth.cli run --limit 40 --workers 4  # 端到端，需要本地 secrets.toml
python -m hanzi_synth.cli audit --in synthesis/out --out synthesis/out/dataset.jsonl

# 合成管线测试：511 个，不联网
python -m unittest discover -s src/tests -t src
```

合成管线要一份本地的 `secrets.toml`，里面写 provider 的 base_url、model、api_key。凭据只从
这个文件读，不进环境变量，文件本身已在 `.gitignore` 里。格式和用法见 `synthesis/README.md`。

### 抓取

爬虫的联网子命令默认会先检查站点的 robots.txt，被禁止就中止；要抓就得自己加 `--ignore-robots`：

```powershell
python crawler/crawl_zisea.py chars --ignore-robots      # 增量抓全部字符详情
python crawler/crawl_zisea.py sprites --ignore-robots    # 无码位字的字形图片
python crawler/crawl_zisea.py verify --ignore-robots     # 核对：站点收录的字是不是都抓下来了
```

## 许可

代码按 Apache License 2.0 分发，全文见 `LICENSE`，版权归 1cyberlangke1。站点数据不在许可范围内
——它也不在这个仓库里。

各部分的细节见 `crawler/README.md`、`synthesis/README.md` 与 `NOTICE`。