"""字海网离线查看器入口脚本。

用法：python crawler/view_zisea.py [--port 8720] [--out crawler/data] [--rebuild]
免责声明：仅供学习交流，数据版权归原站所有，本工具不提供数据分发。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zisea.cli import main  # noqa: E402 - 必须在 sys.path 调整之后导入

if __name__ == "__main__":
    raise SystemExit(main(["view", *sys.argv[1:]]))
