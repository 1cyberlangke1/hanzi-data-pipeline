"""字海网爬虫入口脚本。

用法：python crawler/crawl_zisea.py <stats|enumerate|chars|sprites|verify|view> [选项]
联网子命令默认先查站点 robots.txt（字海网为 Disallow: /），被禁止即中止；
确认已获授权或属于个人学习研究时，加 --ignore-robots 自行放行。
把脚本所在目录加入 sys.path，这样既能当脚本跑，也能被测试以包形式导入。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zisea.cli import main  # noqa: E402 - 必须在 sys.path 调整之后导入

if __name__ == "__main__":
    raise SystemExit(main())
