"""查看器单元测试：索引构建、检索语义、HTTP 端点。

HTTP 部分用真实 ThreadingHTTPServer 绑 127.0.0.1:0（随机空闲端口），
只测本机回环，不出网。
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from zisea.viewer import Entry, ViewerHandler, ViewerIndex, extract_pinyin, strip_tags


def sample_record(ziid: str, zi: str, ye: str, radical: str, stroke: int, mean: str) -> dict:
    """输入：字段值。输出：一条形如 /api/z/{ZIID} 的详情记录。"""
    return {
        "ziid": ziid,
        "goto_from": None,
        "property": {
            "ziid": ziid,
            "zi": zi,
            "pinyin": ye,
            "radical": radical,
            "stroke_total": stroke,
            "structure": 1,
            "ids": "⿰女子",
            "head": "女",
            "tail": "子",
            "head_py_display": "nü",
            "tail_py_display": "zi",
            "zh_ref": {"page": 1, "number": 2},
            "source_ref": "G0-3A43",
        },
        "meanings": {"mean_h": [mean], "mean_z": [], "mean_b": [], "mean_v": [], "mean_k": [], "mean_j": [], "mean_g": [], "discuss": []},
        "variants": [{"ziid": "U058F9", "zi": "壹"}],
        "prev": None,
        "next": None,
    }


SAMPLES = [
    sample_record("U04E00", "一", "hg", "一", 1, "㈠拼音yi1<br>①数名，最小的正整数。<br>②纯；专。"),
    sample_record("U0597D", "好", "hk", "女", 6, "㈠拼音hao3<br>①优点多或使人满意的，与【坏】相对。"),
    sample_record("P80004", "", "si", "一", 4, "敦煌俗字。"),
]


class TextHelperTests(unittest.TestCase):
    def test_strip_tags_turns_br_into_space(self) -> None:
        """<br> 与其余标签都去掉，并压缩空白。"""
        self.assertEqual(strip_tags("甲<br>乙<b>丙</b>"), "甲 乙 丙")

    def test_extract_pinyin_dedupes_and_keeps_tone(self) -> None:
        """拼音去重且保留声调数字，按出现顺序拼接。"""
        self.assertEqual(extract_pinyin("拼音yi1 又拼音heng2 再拼音yi1"), "yi1 heng2")

    def test_extract_pinyin_without_tone(self) -> None:
        """无声调数字时也要能抽出来。"""
        self.assertEqual(extract_pinyin("拼音hao"), "hao")

    def test_extract_pinyin_empty(self) -> None:
        """没有拼音时返回空串。"""
        self.assertEqual(extract_pinyin("敦煌俗字。"), "")


class EntryTests(unittest.TestCase):
    def test_list_roundtrip(self) -> None:
        """Entry 与数组互转后字段完全一致。"""
        entry = Entry("U04E00", "一", "hg", "yi1", "一", 1, 4096, "数名")
        self.assertEqual(Entry.from_list(entry.to_list()), entry)


class ViewerIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "chars.jsonl"
        with self.path.open("w", encoding="utf-8") as handle:
            for record in SAMPLES:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.index = ViewerIndex(self.path)
        self.index.ensure_index(log=lambda _m: None)

    def test_builds_index_from_jsonl(self) -> None:
        """索引条数等于文件行数，且字段抽取正确。"""
        self.assertEqual(len(self.index.entries), 3)
        first = self.index.by_ziid["U04E00"]
        self.assertEqual((first.zi, first.ye, first.pinyin, first.radical, first.stroke), ("一", "hg", "yi1", "一", 1))

    def test_entry_for_sprite_only_char_has_empty_zi(self) -> None:
        """P 组无 Unicode 字形，zi 应为空串（前端据此改用精灵图）。"""
        self.assertEqual(self.index.by_ziid["P80004"].zi, "")

    def test_get_record_returns_full_detail(self) -> None:
        """按偏移回读应拿到完整原始记录。"""
        record = self.index.get_record("U0597D")
        self.assertEqual(record["property"]["ids"], "⿰女子")
        self.assertEqual(record["variants"][0]["zi"], "壹")

    def test_get_record_is_case_insensitive(self) -> None:
        """小写 ziid 也能命中。"""
        self.assertIsNotNone(self.index.get_record("u04e00"))

    def test_get_record_missing_returns_none(self) -> None:
        """不存在的 ZIID 返回 None。"""
        self.assertIsNone(self.index.get_record("U99999"))

    def test_search_by_ziid_exact(self) -> None:
        """ZIID 精确匹配优先，且只返回一条。"""
        entries, total = self.index.search("U04E00")
        self.assertEqual(total, 1)
        self.assertEqual(entries[0].ziid, "U04E00")

    def test_search_by_single_char_exact(self) -> None:
        """单字精确匹配优先于全文包含匹配。"""
        entries, total = self.index.search("好")
        self.assertEqual(total, 1)
        self.assertEqual(entries[0].ziid, "U0597D")

    def test_search_by_pinyin(self) -> None:
        """拼音可搜，且不误伤其他字。"""
        entries, _ = self.index.search("hao3")
        self.assertEqual([e.ziid for e in entries], ["U0597D"])

    def test_pinyin_that_looks_like_ziid_still_searches(self) -> None:
        """像 "zhong1" 这种「字母 + 数字」的拼音恰好符合 ZIID 格式，
        查不到该编号时必须回退到普通检索，不能直接返回空。"""
        entries, total = self.index.search("yi1")
        self.assertEqual(total, 1)
        self.assertEqual(entries[0].ziid, "U04E00")

    def test_unknown_ziid_shaped_query_returns_empty(self) -> None:
        """形似 ZIID 且确实查不到、也不匹配任何字段时，返回空而不是报错。"""
        self.assertEqual(self.index.search("Z99999"), ([], 0))

    def test_search_by_ye_code(self) -> None:
        """叶码精确匹配。"""
        entries, _ = self.index.search("hg")
        self.assertEqual([e.ziid for e in entries], ["U04E00"])

    def test_search_by_radical(self) -> None:
        """部首精确匹配。"""
        entries, _ = self.index.search("女")
        self.assertEqual([e.ziid for e in entries], ["U0597D"])

    def test_search_by_meaning_text(self) -> None:
        """释义关键词可搜。"""
        entries, _ = self.index.search("敦煌")
        self.assertEqual([e.ziid for e in entries], ["P80004"])

    def test_search_empty_query(self) -> None:
        """空关键词返回空结果，不返回全部。"""
        self.assertEqual(self.index.search("   "), ([], 0))

    def test_search_respects_limit_and_offset(self) -> None:
        """limit/offset 用于分页。"""
        entries, total = self.index.search("拼音", limit=1, offset=1)
        self.assertEqual(total, 2)
        self.assertEqual(len(entries), 1)

    def test_entry_to_light_truncates_text(self) -> None:
        """列表用轻量字段的释义截断到 80 字，避免传输过大。"""
        light = self.index.entry_to_light(self.index.by_ziid["U04E00"])
        self.assertLessEqual(len(light["text"]), 80)
        self.assertEqual(light["ziid"], "U04E00")

    def test_stats_reports_count_and_bytes(self) -> None:
        """stats 返回条数与数据文件字节数。"""
        stats = self.index.stats()
        self.assertEqual(stats["count"], 3)
        self.assertGreater(stats["bytes"], 0)

    def test_reload_uses_cached_index(self) -> None:
        """索引已存在且比数据新时直接复用，不重建。"""
        fresh = ViewerIndex(self.path)
        fresh.ensure_index(log=lambda _m: None)
        self.assertEqual(len(fresh.entries), 3)

    def test_rebuild_flag_forces_rebuild(self) -> None:
        """--rebuild 时忽略缓存索引，重新扫描。"""
        fresh = ViewerIndex(self.path)
        fresh.ensure_index(rebuild=True, log=lambda _m: None)
        self.assertEqual(len(fresh.entries), 3)

    def test_broken_line_is_skipped_not_fatal(self) -> None:
        """半行损坏时跳过该行，其余记录仍可用。"""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{半行\n")
        fresh = ViewerIndex(self.path)
        fresh.ensure_index(rebuild=True, log=lambda _m: None)
        self.assertEqual(len(fresh.entries), 3)

    def test_missing_data_file_raises_clear_error(self) -> None:
        """数据文件不存在时给出可读错误。"""
        index = ViewerIndex(Path(self._tmp.name) / "nope.jsonl")
        with self.assertRaises(FileNotFoundError):
            index.ensure_index(log=lambda _m: None)


class HttpEndpointTests(unittest.TestCase):
    """真实起一个回环 HTTP 服务，端到端验证路由与响应格式。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "chars.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in SAMPLES:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        # 放一张本地精灵图，验证离线优先走本地文件而不是回源站点
        sprite_dir = Path(cls._tmp.name) / "sprites"
        sprite_dir.mkdir(exist_ok=True)
        (sprite_dir / "P80004.png").write_bytes(b"\x89PNG-LOCAL")
        index = ViewerIndex(path)
        index.ensure_index(log=lambda _m: None)
        # base_url 指向必然连不上的端口，保证精灵图测试与「当前有没有网」无关
        handler = type("TestHandler", (ViewerHandler,), {"index": index, "base_url": "http://127.0.0.1:9"})
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls._base = f"http://127.0.0.1:{cls._server.server_address[1]}"
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()
        cls._tmp.cleanup()

    def _get(self, path: str) -> tuple[int, bytes]:
        """输入：路径。输出：(状态码, 响应体)。预期：HTTP 错误也返回而不是抛异常。"""
        try:
            with urllib.request.urlopen(self._base + path, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_index_page_served(self) -> None:
        """根路径返回界面 HTML，且带免责声明。"""
        status, body = self._get("/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("字海网", text)
        self.assertIn("仅供学习交流", text)

    def test_meta_endpoint(self) -> None:
        """meta 返回条数。"""
        status, body = self._get("/api/meta")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["count"], 3)

    def test_search_endpoint(self) -> None:
        """search 返回 items 与 total。"""
        status, body = self._get("/api/search?q=hao3")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["ziid"], "U0597D")

    def test_char_endpoint(self) -> None:
        """char 返回完整记录。"""
        status, body = self._get("/api/char/U04E00")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["property"]["zi"], "一")

    def test_char_endpoint_404(self) -> None:
        """不存在的 ZIID 返回 404 与错误信息。"""
        status, body = self._get("/api/char/U99999")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body))

    def test_unknown_path_404(self) -> None:
        """未知路径返回 404。"""
        status, _body = self._get("/nope")
        self.assertEqual(status, 404)

    def test_sprite_unavailable_returns_404(self) -> None:
        """本地没有图、又没开回源时回 404，而不是 500 或挂死（Z43651 未准备本地图）。"""
        status, body = self._get("/api/sprite/Z43651")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body))

    def test_sprite_prefers_local_file(self) -> None:
        """本地已有图片时直接返回本地字节，不依赖网络。"""
        status, body = self._get("/api/sprite/P80004")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"\x89PNG-LOCAL")


class OriginStub(BaseHTTPRequestHandler):
    """假「源站」：记录被请求的路径，回一张假 PNG。用于证明默认不回源。"""

    hits: list[str] = []

    def log_message(self, fmt: str, *args) -> None:
        """输入：默认日志参数。输出：无。预期：静音。"""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        """输入：GET 路径。输出：假 PNG。预期：把路径记进 hits 供断言。"""
        type(self).hits.append(self.path)
        payload = b"\x89PNG-ORIGIN"
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class SpriteOriginFetchTests(unittest.TestCase):
    """默认不回源：本地缺图时既不碰网络，也不假装成功；开了开关才真的去取。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "chars.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in SAMPLES:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        index = ViewerIndex(path)
        index.ensure_index(log=lambda _m: None)

        OriginStub.hits = []
        self._origin = ThreadingHTTPServer(("127.0.0.1", 0), OriginStub)
        self.addCleanup(self._origin.server_close)
        self.addCleanup(self._origin.shutdown)
        threading.Thread(target=self._origin.serve_forever, daemon=True).start()
        origin_base = f"http://127.0.0.1:{self._origin.server_address[1]}"

        self._bases: dict[str, str] = {}
        for name, allow in (("off", False), ("on", True)):
            handler = type(
                f"Handler{name}",
                (ViewerHandler,),
                {"index": index, "base_url": origin_base, "allow_origin_fetch": allow},
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._bases[name] = f"http://127.0.0.1:{server.server_address[1]}"

    def _get(self, base: str, path: str) -> tuple[int, bytes]:
        """输入：查看器基址与路径。输出：(状态码, 响应体)。预期：HTTP 错误也返回。"""
        try:
            with urllib.request.urlopen(base + path, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_default_never_touches_origin(self) -> None:
        """默认配置下本地缺图只回 404，一个源站请求都不发（不绕过 robots 闸门）。"""
        status, body = self._get(self._bases["off"], "/api/sprite/Z43651")
        self.assertEqual(status, 404)
        self.assertIn("error", json.loads(body))
        self.assertEqual(OriginStub.hits, [])

    def test_opt_in_fetches_from_origin(self) -> None:
        """显式开启 allow_origin_fetch 后才回源取图。"""
        status, body = self._get(self._bases["on"], "/api/sprite/Z43651")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"\x89PNG-ORIGIN")
        self.assertEqual(OriginStub.hits, ["/sprites/Z43651.png"])


if __name__ == "__main__":
    unittest.main()
