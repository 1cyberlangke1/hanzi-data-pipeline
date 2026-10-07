"""精灵图下载与二进制取数测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeResponse, RoutingSession, make_request_exception
from zisea.client import NotFoundError, ZiseaClient, ZiseaError
from zisea.sprites import iter_sprite_ziids, run_sprites, sprite_report_to_dict


def make_client(session) -> ZiseaClient:
    """输入：假 session。输出：退避/延迟归零的客户端。"""
    return ZiseaClient(base_url="http://example.test", session=session, backoff=0, delay=0)


def record(ziid: str, zi: str) -> dict:
    """输入：ZIID 与字形字符。输出：一条最小详情记录。预期：zi 为空表示无 Unicode 字形。"""
    return {"ziid": ziid, "property": {"ziid": ziid, "zi": zi}, "meanings": {}}


class GetBytesTests(unittest.TestCase):
    def test_returns_raw_bytes(self) -> None:
        """200 时返回原始字节，不做任何解码。"""
        session = RoutingSession({"/sprites/": FakeResponse(200, content=b"\x89PNG\r\n")})
        self.assertEqual(make_client(session).get_bytes("/sprites/Z43651.png"), b"\x89PNG\r\n")

    def test_404_raises_not_found(self) -> None:
        """404 说明该字有 Unicode 字形，站点不提供图片，归一到 NotFoundError。"""
        session = RoutingSession({"/sprites/": FakeResponse(404, content=b"")})
        with self.assertRaises(NotFoundError):
            make_client(session).get_bytes("/sprites/U04E00.png")

    def test_422_raises_bad_request(self) -> None:
        """参数错误不重试。"""
        from zisea.client import BadRequestError

        session = RoutingSession({"/sprites/": FakeResponse(422)})
        with self.assertRaises(BadRequestError):
            make_client(session).get_bytes("/sprites/x.png")

    def test_network_error_exhausts_retries(self) -> None:
        """网络一直不通时抛 ZiseaError。"""
        def handler(url, params):
            raise make_request_exception()

        with self.assertRaises(ZiseaError):
            make_client(RoutingSession({"/sprites/": handler})).get_bytes("/sprites/x.png")


class IterSpriteZiidsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "chars.jsonl"

    def _write(self, lines: list[str]) -> None:
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_yields_only_records_without_unicode_glyph(self) -> None:
        """只产出 zi 为空的 ZIID，有字形的字跳过。"""
        self._write([json.dumps(record("U04E00", "一"), ensure_ascii=False), json.dumps(record("Z43651", ""), ensure_ascii=False)])
        self.assertEqual(list(iter_sprite_ziids(self.path)), ["Z43651"])

    def test_skips_broken_lines(self) -> None:
        """脏行跳过，不阻塞后面的记录。"""
        self._write(["{坏行", json.dumps(record("P80004", ""), ensure_ascii=False)])
        self.assertEqual(list(iter_sprite_ziids(self.path)), ["P80004"])

    def test_missing_file_yields_nothing(self) -> None:
        """文件不存在时产出空迭代。"""
        self.assertEqual(list(iter_sprite_ziids(Path(self._tmp.name) / "nope.jsonl")), [])


class RunSpritesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        (self.out / "chars.jsonl").write_text(
            json.dumps(record("U04E00", "一"), ensure_ascii=False)
            + "\n"
            + json.dumps(record("Z43651", ""), ensure_ascii=False)
            + "\n"
            + json.dumps(record("P80004", ""), ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    def _client(self, handler) -> ZiseaClient:
        return make_client(RoutingSession({"/sprites/": handler}))

    def test_downloads_only_needed_sprites(self) -> None:
        """只下载无字形字，落成 {ZIID}.png。"""

        def handler(url, params):
            return FakeResponse(200, content=b"PNG:" + url.encode())

        report = run_sprites(self.out, self._client(handler), log=lambda _m: None)
        self.assertEqual(report.needed, 2)
        self.assertEqual(report.downloaded, 2)
        self.assertTrue((self.out / "sprites" / "Z43651.png").exists())
        self.assertFalse((self.out / "sprites" / "U04E00.png").exists())

    def test_second_run_skips_existing(self) -> None:
        """已存在的图片不重复下载。"""

        def handler(url, params):
            return FakeResponse(200, content=b"PNG")

        run_sprites(self.out, self._client(handler), log=lambda _m: None)
        report = run_sprites(self.out, self._client(handler), log=lambda _m: None)
        self.assertEqual(report.downloaded, 0)
        self.assertEqual(report.already_have, 2)

    def test_404_counted_as_unavailable_not_failed(self) -> None:
        """站点没有该图时记 unavailable，不算失败。"""
        report = run_sprites(self.out, self._client(lambda u, p: FakeResponse(404)), log=lambda _m: None)
        self.assertEqual(report.unavailable, 2)
        self.assertEqual(report.failed, 0)

    def test_failure_recorded_and_continues(self) -> None:
        """单张持续失败只记录，其他继续。"""

        def handler(url, params):
            # 固定让其中一个字一直失败：客户端自身会重试，重试耗尽后由 run_sprites 计数
            if "Z43651" in url:
                raise make_request_exception()
            return FakeResponse(200, content=b"PNG")

        report = run_sprites(self.out, self._client(handler), workers=1, log=lambda _m: None)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.downloaded, 1)
        self.assertEqual(len(report.failed_ids), 1)

    def test_limit_caps_downloads(self) -> None:
        """--limit 限制本轮张数。"""

        def handler(url, params):
            return FakeResponse(200, content=b"PNG")

        report = run_sprites(self.out, self._client(handler), limit=1, log=lambda _m: None)
        self.assertEqual(report.downloaded, 1)

    def test_report_dict_excludes_failed_ids(self) -> None:
        """报告序列化去掉明细。"""
        from zisea.sprites import SpriteReport

        self.assertNotIn("failed_ids", sprite_report_to_dict(SpriteReport(needed=1, failed_ids=["X"])))


if __name__ == "__main__":
    unittest.main()
