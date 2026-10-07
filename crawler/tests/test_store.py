"""JsonlStore 单元测试：追加、回读、半行修复、原子替换。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zisea.store import JsonlStore, load_json, write_json_atomic


class JsonlStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sub" / "chars.jsonl"

    def test_append_creates_parent_and_roundtrips_chinese(self) -> None:
        """追加时自动建父目录；中文按 UTF-8 原样存取，不被转义成 \\uXXXX。"""
        store = JsonlStore(self.path)
        store.append({"ziid": "U04E00", "zi": "一", "meaning": "数名，最小的正整数"})
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn("一", raw)
        self.assertNotIn("\\u4e00", raw)
        self.assertEqual(list(store.iter_records())[0]["zi"], "一")

    def test_load_keys_skips_records_without_key(self) -> None:
        """缺主键的记录不进入已完成集合，避免把脏数据当成已抓。"""
        store = JsonlStore(self.path)
        store.append({"ziid": "U04E00"})
        store.append({"zi": "无主键"})
        self.assertEqual(store.load_keys(), {"U04E00"})

    def test_count_ignores_blank_lines(self) -> None:
        """空行不计入条数。"""
        store = JsonlStore(self.path)
        store.append({"ziid": "A"})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self.assertEqual(store.count(), 1)

    def test_missing_file_is_empty(self) -> None:
        """文件不存在时 count/load_keys/iter_records 都要安全返回空。"""
        store = JsonlStore(self.path)
        self.assertEqual(store.count(), 0)
        self.assertEqual(store.load_keys(), set())
        self.assertEqual(list(store.iter_records()), [])
        self.assertEqual(store.repair_tail(), 0)

    def test_repair_tail_truncates_partial_line(self) -> None:
        """模拟进程被杀留下的半行，应被截断，且前面的完整记录保留。"""
        store = JsonlStore(self.path)
        store.append({"ziid": "U04E00"})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"ziid": "U04E01", "mean')
        removed = store.repair_tail()
        self.assertGreater(removed, 0)
        self.assertEqual(store.load_keys(), {"U04E00"})

    def test_repair_tail_on_complete_file_is_noop(self) -> None:
        """文件以换行结尾时，不应改动任何字节。"""
        store = JsonlStore(self.path)
        store.append({"ziid": "U04E00"})
        before = self.path.read_bytes()
        self.assertEqual(store.repair_tail(), 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_repair_tail_on_file_without_newline_drops_everything(self) -> None:
        """整个文件都没有换行符，说明一条完整记录都没写成，应清空。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"ziid": "U04E00"', encoding="utf-8")
        store = JsonlStore(self.path)
        self.assertGreater(store.repair_tail(), 0)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "")

    def test_replace_all_is_atomic_and_replaces_old_content(self) -> None:
        """replace_all 后旧内容不应残留，条数等于新序列长度。"""
        store = JsonlStore(self.path)
        store.append({"ziid": "OLD"})
        written = store.replace_all([{"ziid": "N1"}, {"ziid": "N2"}])
        self.assertEqual(written, 2)
        self.assertEqual(store.load_keys(), {"N1", "N2"})

    def test_replace_all_leaves_no_temp_files(self) -> None:
        """原子写完成后不应残留 .tmp 文件。"""
        store = JsonlStore(self.path)
        store.replace_all([{"ziid": "N1"}])
        leftovers = [p.name for p in self.path.parent.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

    def test_broken_line_raises_with_location(self) -> None:
        """遇到损坏行要报出「文件:行号」，方便定位。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"ziid": "ok"}\n不是JSON\n', encoding="utf-8")
        store = JsonlStore(self.path)
        with self.assertRaises(ValueError) as ctx:
            list(store.iter_records())
        self.assertIn(":2", str(ctx.exception))

    def test_custom_key_field(self) -> None:
        """成语用 zi 作主键，需支持自定义字段名。"""
        store = JsonlStore(self.path, key_field="zi")
        store.append({"zi": "一心一意"})
        self.assertEqual(store.load_keys(), {"一心一意"})


class JsonHelpersTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "state.json"

    def test_load_json_returns_default_when_missing(self) -> None:
        """文件不存在时返回默认值。"""
        self.assertEqual(load_json(self.path, default={"a": 1}), {"a": 1})

    def test_load_json_returns_default_when_corrupted(self) -> None:
        """文件损坏时返回默认值而不是抛错（state.json 损坏不该阻塞爬取）。"""
        self.path.write_text("{ 不是 json", encoding="utf-8")
        self.assertIsNone(load_json(self.path))

    def test_write_json_atomic_roundtrip_keeps_chinese(self) -> None:
        """原子写后能读回，中文不被转义。"""
        write_json_atomic(self.path, {"note": "全站统计"})
        self.assertEqual(load_json(self.path)["note"], "全站统计")
        self.assertIn("全站统计", self.path.read_text(encoding="utf-8"))

    def test_write_json_atomic_overwrites_and_leaves_no_temp(self) -> None:
        """重复写应覆盖旧内容且不留临时文件。"""
        write_json_atomic(self.path, {"v": 1})
        write_json_atomic(self.path, {"v": 2})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"v": 2})
        leftovers = [p.name for p in self.path.parent.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
