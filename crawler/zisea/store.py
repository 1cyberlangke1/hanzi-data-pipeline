"""JSONL 落盘与断点续爬支持。

设计：单文件 JSONL（一行一条记录），天然可流式追加。
「已完成集合」直接从 JSONL 回读，不另存状态文件，避免两份状态漂移。
崩溃可能留下半行，启动时用 repair_tail() 截断。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


class JsonlStore:
    """输入：JSONL 文件路径与主键字段名。输出：可追加、可回读主键的存储句柄。
    预期：append 立即 flush，保证进程被杀时已写内容不丢。
    """

    def __init__(self, path: str | os.PathLike[str], key_field: str = "ziid") -> None:
        self.path = Path(path)
        self.key_field = key_field
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 读 ----------

    def iter_records(self) -> Iterator[dict]:
        """输入：无。输出：迭代文件中每条 JSON 记录。
        预期：文件不存在时产出空迭代；损坏行直接抛出（应先用 repair_tail 修）。
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_no} 不是合法 JSON: {exc}") from exc

    def load_keys(self) -> set[str]:
        """输入：无。输出：已落盘记录的主键集合（用于断点续爬时跳过）。"""
        keys: set[str] = set()
        for record in self.iter_records():
            value = record.get(self.key_field)
            if value is not None:
                keys.add(str(value))
        return keys

    def count(self) -> int:
        """输入：无。输出：文件中的记录条数（文件不存在时为 0）。"""
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    # ---------- 写 ----------

    def append(self, record: dict) -> None:
        """输入：一条记录。输出：无。预期：追加为一行并立即落盘（ensure_ascii=False 保中文可读）。"""
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_many(self, records: Iterable[dict]) -> int:
        """输入：记录序列。输出：实际写入条数。预期：一次打开批量写，减少 fsync 次数。"""
        written = 0
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            os.fsync(handle.fileno())
        return written

    def replace_all(self, records: Iterable[dict]) -> int:
        """输入：完整记录序列。输出：写入条数。
        预期：先写临时文件再原子替换，中途崩溃不会破坏旧文件。
        """
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        written = 0
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return written

    def repair_tail(self) -> int:
        """输入：无。输出：被截断的字节数（0 表示文件完好或不存在）。
        预期：只砍掉最后一个换行符之后的残留内容，用于修复崩溃产生的半行。
        """
        if not self.path.exists():
            return 0
        size = self.path.stat().st_size
        if size == 0:
            return 0
        with self.path.open("rb") as handle:
            handle.seek(max(0, size - 1))
            if handle.read(1) == b"\n":
                return 0
        with self.path.open("r+b") as handle:
            handle.seek(0)
            data = handle.read()
            cut = data.rfind(b"\n")
            keep = cut + 1 if cut >= 0 else 0
            removed = len(data) - keep
            handle.truncate(keep)
        return removed


def load_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """输入：JSON 文件路径与默认值。输出：解析结果；文件不存在或损坏时返回默认值。"""
    file_path = Path(path)
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json_atomic(path: str | os.PathLike[str], data: Any) -> None:
    """输入：目标路径与可序列化对象。输出：无。预期：临时文件 + os.replace，写入过程原子。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(file_path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, file_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
