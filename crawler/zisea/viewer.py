"""字海网离线查看器：本机只读 HTTP 服务 + 网页界面。

输入：crawler/data/chars.jsonl（由 crawl_zisea.py 产出）
输出：本机 http://127.0.0.1:8720 的浏览界面，支持检索与单字详情
预期：默认纯离线，只用标准库、不访问外网。本地缺精灵图时回 404，前端退化成占位符；
      只有显式开启 allow_origin_fetch（CLI: --allow-origin-fetch）才会回源站点代理取图 ——
      默认关闭是为了让查看器不绕过爬虫的 robots.txt 闸门。

免责声明：本工具仅供学习交流，数据版权归原站所有，不对外提供数据分发。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .client import DEFAULT_BASE_URL
from .sprites import SPRITES_DIR
from .store import load_json, write_json_atomic

INDEX_FILE = "viewer_index.json"
WEB_DIR = Path(__file__).resolve().parent / "web"

# 叶码是站点自有的两分拼音编码（property.pinyin），真实读音从释义文本里抽。
PINYIN_RE = re.compile(r"拼音([a-zA-ZüÜêÊ]+)([1-5])?")
TAG_RE = re.compile(r"<br\s*/?>|<[^>]+>")
ZIID_RE = re.compile(r"^[UZP][0-9A-Z]{5}$")


def strip_tags(text: str) -> str:
    """输入：含 <br> 等标签的释义片段。输出：替换成空格并压缩后的纯文本。"""
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", text)).strip()


def extract_pinyin(text: str) -> str:
    """输入：释义纯文本。输出：去重后的拼音（带声调数字），如 "yi1 heng2"；无则空串。"""
    seen: list[str] = []
    for match in PINYIN_RE.finditer(text):
        syllable = match.group(1).lower() + (match.group(2) or "")
        if syllable not in seen:
            seen.append(syllable)
    return " ".join(seen)


@dataclass
class Entry:
    """索引条目：检索用的最小字段集，详情按 offset 回读原始 JSONL。"""

    ziid: str
    zi: str
    ye: str
    pinyin: str
    radical: str
    stroke: int
    offset: int
    text: str

    def to_list(self) -> list:
        """输入：无。输出：紧凑数组，用于落盘索引（比对象省一半以上体积）。"""
        return [self.ziid, self.zi, self.ye, self.pinyin, self.radical, self.stroke, self.offset, self.text]

    @classmethod
    def from_list(cls, row: list) -> "Entry":
        """输入：索引数组。输出：Entry。预期：字段顺序与 to_list 一致。"""
        return cls(row[0], row[1], row[2], row[3], row[4], int(row[5]), int(row[6]), row[7])


class ViewerIndex:
    """输入：chars.jsonl 路径。输出：可检索的索引 + 按 ZIID 取详情的读句柄。
    预期：索引缺失或过期时自动重建；重建过程中爬虫可以继续写文件（只读已完整写入的行）。
    """

    def __init__(self, chars_path: str | Path) -> None:
        self.chars_path = Path(chars_path)
        self.index_path = self.chars_path.parent / INDEX_FILE
        self.sprite_dir = self.chars_path.parent / SPRITES_DIR
        self.entries: list[Entry] = []
        self.by_ziid: dict[str, Entry] = {}
        self._lock = threading.Lock()

    # ---------- 索引构建 ----------

    def ensure_index(self, rebuild: bool = False, log=print) -> None:
        """输入：是否强制重建。输出：无。预期：索引比数据源新就直接加载，否则重建。"""
        if not self.chars_path.exists():
            raise FileNotFoundError(f"找不到数据文件：{self.chars_path}（先跑 crawl_zisea.py chars）")
        if not rebuild and self.index_path.exists():
            if self.index_path.stat().st_mtime >= self.chars_path.stat().st_mtime:
                log(f"加载已有索引 {self.index_path.name} ...")
                self._load_index()
                return
        log("构建索引中（首次约十几秒）...")
        self._build_index()
        log(f"索引就绪：{len(self.entries)} 条")

    def _build_index(self) -> None:
        """输入：无。输出：无。预期：边读边记字节偏移；损坏行跳过并计数，不中断。"""
        entries: list[Entry] = []
        skipped = 0
        with self.chars_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    skipped += 1
                    continue
                entries.append(self._make_entry(record, offset))
        self._set_entries(entries)
        write_json_atomic(
            self.index_path,
            {"count": len(entries), "skipped": skipped, "entries": [e.to_list() for e in entries]},
        )

    @staticmethod
    def _make_entry(record: dict, offset: int) -> Entry:
        """输入：一条详情记录与其在文件中的字节偏移。输出：Entry。
        预期：缺失字段一律降级为空值，保证索引不因个别脏数据整体失败。
        """
        prop = record.get("property") or {}
        meanings = record.get("meanings") or {}
        chunks: list[str] = []
        for key in ("mean_h", "mean_z", "mean_b", "mean_v", "mean_k", "mean_j", "mean_g", "discuss"):
            for item in meanings.get(key) or []:
                if isinstance(item, str):
                    chunks.append(item)
        text = strip_tags(" ".join(chunks))
        return Entry(
            ziid=str(record.get("ziid") or ""),
            zi=str(prop.get("zi") or ""),
            ye=str(prop.get("pinyin") or ""),
            pinyin=extract_pinyin(text),
            radical=str(prop.get("radical") or ""),
            stroke=int(prop.get("stroke_total") or 0),
            offset=offset,
            text=text,
        )

    def _load_index(self) -> None:
        """输入：无。输出：无。预期：索引损坏时回退到重建，不让查看器起不来。"""
        data = load_json(self.index_path, default=None)
        if not data or "entries" not in data:
            self._build_index()
            return
        self._set_entries([Entry.from_list(row) for row in data["entries"]])

    def _set_entries(self, entries: list[Entry]) -> None:
        """输入：条目列表。输出：无。预期：同时维护列表与 ZIID 反查表。"""
        self.entries = entries
        self.by_ziid = {e.ziid: e for e in entries}

    # ---------- 查询 ----------

    def get_record(self, ziid: str) -> dict | None:
        """输入：ZIID。输出：完整详情 dict；不存在返回 None。预期：按偏移 seek 读单行，不载入全文件。"""
        entry = self.by_ziid.get(ziid.upper())
        if entry is None:
            return None
        with self._lock, self.chars_path.open("rb") as handle:
            handle.seek(entry.offset)
            line = handle.readline()
        try:
            return json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def search(self, query: str, limit: int = 60, offset: int = 0) -> tuple[list[Entry], int]:
        """输入：关键词、返回条数、起始偏移。输出：(命中条目, 命中总数)。
        预期：ZIID 精确 > 单字精确 > 前缀/包含匹配（ZIID、汉字、拼音、叶码、部首、释义）。
        """
        needle = query.strip()
        if not needle:
            return [], 0
        upper = needle.upper()

        if ZIID_RE.match(upper):
            entry = self.by_ziid.get(upper)
            if entry is not None:
                return [entry], 1
            # 形似 ZIID 但查不到时不能直接返回空：像拼音 "zhong1" 恰好符合
            # 「Z + 5 位字母数字」，直接返回空会让拼音检索失效，这里继续走普通检索。

        exact = [e for e in self.entries if e.zi == needle]
        if exact:
            return exact[offset : offset + limit], len(exact)

        lowered = needle.lower()
        hits = [
            e
            for e in self.entries
            if lowered in e.pinyin
            or upper in e.ziid
            or lowered == e.ye
            or needle == e.radical
            or needle in e.text
        ]
        return hits[offset : offset + limit], len(hits)

    def entry_to_light(self, entry: Entry) -> dict:
        """输入：Entry。输出：前端列表用的轻量 dict（含释义摘要）。"""
        return {
            "ziid": entry.ziid,
            "zi": entry.zi,
            "ye": entry.ye,
            "pinyin": entry.pinyin,
            "radical": entry.radical,
            "stroke": entry.stroke,
            "text": entry.text[:80],
        }

    def stats(self) -> dict:
        """输入：无。输出：条数与数据文件大小的概要。"""
        size = self.chars_path.stat().st_size if self.chars_path.exists() else 0
        return {"count": len(self.entries), "bytes": size, "indexed_at": self.index_path.stat().st_mtime if self.index_path.exists() else 0}


class ViewerHandler(BaseHTTPRequestHandler):
    """只读 HTTP 处理器。输入：GET 请求。输出：HTML / JSON / 精灵图字节流。"""

    index: ViewerIndex
    base_url: str = DEFAULT_BASE_URL
    # 默认不回源：保持查看器纯离线，也避免绕过爬虫的 robots.txt 闸门。
    allow_origin_fetch: bool = False
    server_version = "ZiseaViewer/1.0"

    def log_message(self, fmt: str, *args) -> None:
        """输入：默认日志参数。输出：无。预期：静音，避免刷屏拖慢服务。"""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        """输入：GET 路径与查询串。输出：对应响应。预期：任何异常都回 500 且带人话错误。"""
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send_html()
            elif parsed.path == "/api/meta":
                self._send_json(self.index.stats())
            elif parsed.path == "/api/search":
                self._handle_search(parse_qs(parsed.query))
            elif parsed.path.startswith("/api/char/"):
                self._handle_char(parsed.path.rsplit("/", 1)[-1])
            elif parsed.path.startswith("/api/sprite/"):
                self._handle_sprite(parsed.path.rsplit("/", 1)[-1])
            else:
                self._send_json({"error": "未知路径"}, status=404)
        except Exception as exc:  # noqa: BLE001 - 顶层兜底，避免线程里抛裸异常
            self._send_json({"error": f"服务内部错误：{exc}"}, status=500)

    def _handle_search(self, query: dict) -> None:
        """输入：查询参数 q/limit/offset。输出：{"items":[...], "total":int}。"""
        keyword = (query.get("q") or [""])[0]
        limit = min(int((query.get("limit") or ["60"])[0]), 300)
        offset = max(int((query.get("offset") or ["0"])[0]), 0)
        entries, total = self.index.search(keyword, limit=limit, offset=offset)
        self._send_json({"items": [self.index.entry_to_light(e) for e in entries], "total": total})

    def _handle_char(self, ziid: str) -> None:
        """输入：路径里的 ZIID。输出：完整详情 JSON，或 404。"""
        record = self.index.get_record(ziid)
        if record is None:
            self._send_json({"error": f"未找到 {ziid}"}, status=404)
            return
        self._send_json(record)

    def _handle_sprite(self, ziid: str) -> None:
        """输入：路径里的 ZIID。输出：站点精灵图 PNG 字节流。
        预期：优先读本地已下载的图片（离线可用）；没有才回源站点代理；
              两边都取不到时回 404，前端会退化成占位符。
        """
        local = self.index.sprite_dir / f"{ziid}.png"
        if local.exists():
            payload = local.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(payload)
            return

        if not self.allow_origin_fetch:
            self._send_json(
                {
                    "error": f"本地没有 {ziid} 的字形图"
                    "（先跑 crawl_zisea.py sprites 下载，或用 --allow-origin-fetch 回源取）"
                },
                status=404,
            )
            return

        import urllib.error
        import urllib.request

        url = f"{self.base_url}/sprites/{quote(ziid)}.png"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = response.read()
        except (urllib.error.URLError, OSError):
            self._send_json({"error": "精灵图不可用"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self) -> None:
        """输入：无。输出：web/index.html 内容。预期：文件缺失时给出明确提示。"""
        page = WEB_DIR / "index.html"
        if not page.exists():
            self._send_json({"error": f"缺少界面文件 {page}"}, status=500)
            return
        payload = page.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: dict, status: int = 200) -> None:
        """输入：可序列化对象与状态码。输出：UTF-8 JSON 响应。"""
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(
    chars_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8720,
    rebuild: bool = False,
    allow_origin_fetch: bool = False,
    log=print,
) -> None:
    """输入：数据文件、监听地址与端口、是否重建索引、是否允许回源取精灵图。
    输出：无（阻塞直到 Ctrl+C）。
    预期：只监听回环地址，不对外网暴露；端口占用时报人话并退出。
    """
    index = ViewerIndex(chars_path)
    index.ensure_index(rebuild=rebuild, log=log)

    handler = type(
        "BoundViewerHandler",
        (ViewerHandler,),
        {"index": index, "allow_origin_fetch": allow_origin_fetch},
    )
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise SystemExit(f"无法监听 {host}:{port}（端口可能被占用）：{exc}") from exc

    log(f"字海网离线查看器已启动：http://{host}:{port}")
    log("仅供学习交流，数据版权归原站所有；按 Ctrl+C 退出。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("\n已退出。")
    finally:
        httpd.server_close()
