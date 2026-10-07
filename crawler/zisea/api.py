"""字海网（新叶典）API 端点封装。

端点清单来自站点 SPA 的 JS 包（/assets/index-*.js）里的客户端对象，实测可用：
  GET /api/groups?           全部系列/前缀桶与计数（枚举入口）
  GET /api/z-range?prefix=&page=&page_size=   某前缀下的轻量字符列表
  GET /api/z/{ZIID}          单字完整详情
  GET /api/stats             全站统计 + daily 增量线索
  GET /api/sprite-version    精灵图版本号
"""

from __future__ import annotations

from typing import Iterator

from .client import NotFoundError, ZiseaClient

# 服务端 FastAPI 校验上限：page_size > 500 会返回 422。
MAX_PAGE_SIZE = 500


class ZiseaApi:
    """输入：一个 ZiseaClient。输出：各端点解析后的结构化数据。
    预期：只做端点语义包装（分页、404 归一化），不做字段裁剪。
    """

    def __init__(self, client: ZiseaClient) -> None:
        self._client = client

    # ---------- 统计类 ----------

    def groups(self) -> dict:
        """输入：无。输出：{"series":[{series,label,note,depth,count,groups:[{prefix,count}]}]}。"""
        return self._client.get_json("/api/groups")

    def stats(self) -> dict:
        """输入：无。输出：全站统计 dict，含 total_characters / series / daily 数组。"""
        return self._client.get_json("/api/stats")

    def sprite_version(self) -> str:
        """输入：无。输出：精灵图版本号字符串（如 "8894233ed869"），取不到时返回空串。"""
        data = self._client.get_json("/api/sprite-version")
        return str(data.get("version", "")) if isinstance(data, dict) else ""

    # ---------- 字符枚举 ----------

    def iter_prefixes(self) -> Iterator[tuple[str, str, int]]:
        """输入：无。输出：迭代 (系列字母, 前缀, 该前缀收录数)。
        预期：顺序与 /api/groups 返回一致（U -> Z -> P）。
        """
        for series in self.groups().get("series", []):
            letter = series.get("series", "")
            for group in series.get("groups", []):
                yield letter, group.get("prefix", ""), int(group.get("count", 0))

    def z_range(self, prefix: str, page: int = 1, page_size: int = MAX_PAGE_SIZE) -> dict:
        """输入：前缀、页码（从 1 开始）、每页条数（<=500）。
        输出：{"items":[...轻量记录...], "total":int, "page":int, "page_size":int}。"""
        return self._client.get_json(
            "/api/z-range", {"prefix": prefix, "page": page, "page_size": page_size}
        )

    def iter_z_range(self, prefix: str, page_size: int = MAX_PAGE_SIZE) -> Iterator[dict]:
        """输入：前缀、每页条数。输出：迭代该前缀下全部轻量记录。
        预期：按 total 收敛；若服务端 total 异常导致空页，立即停止，避免死循环。
        """
        page = 1
        seen = 0
        total: int | None = None
        while True:
            data = self.z_range(prefix, page=page, page_size=page_size)
            items = data.get("items") or []
            if total is None:
                total = int(data.get("total", 0))
            if not items:
                return
            for item in items:
                yield item
            seen += len(items)
            if total is not None and seen >= total:
                return
            page += 1

    def iter_ziids(self, page_size: int = MAX_PAGE_SIZE) -> Iterator[tuple[str, str]]:
        """输入：每页条数。输出：迭代 (ZIID, 前缀)，覆盖 U/Z/P 全部系列。
        预期：这是全量枚举的权威入口，比 prev/next 链更可靠。
        """
        for _letter, prefix, _count in self.iter_prefixes():
            for item in self.iter_z_range(prefix, page_size=page_size):
                ziid = item.get("ziid")
                if ziid:
                    yield ziid, prefix

    # ---------- 单字详情 ----------

    def character(self, ziid: str) -> dict | None:
        """输入：ZIID（如 "U04E00"）。输出：完整详情 dict；该 ZIID 未收录时返回 None。
        预期：只吞 404，其余错误照常抛出（400 说明 ZIID 格式写错了，是 bug）。
        """
        try:
            return self._client.get_json(f"/api/z/{ziid}")
        except NotFoundError:
            return None
