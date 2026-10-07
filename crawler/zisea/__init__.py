"""字海网（新叶典）爬虫包。

对外主要入口：ZiseaClient / ZiseaApi / run_chars / run_sprites / run_verify / main。

"""

from .api import MAX_PAGE_SIZE, ZiseaApi
from .client import DEFAULT_BASE_URL, BadRequestError, NotFoundError, ZiseaClient, ZiseaError
from .robots import RobotsDecision, check_robots
from .store import JsonlStore, load_json, write_json_atomic
from .verify import VerifyReport, run_verify

__all__ = [
    "MAX_PAGE_SIZE",
    "ZiseaApi",
    "ZiseaClient",
    "ZiseaError",
    "NotFoundError",
    "BadRequestError",
    "JsonlStore",
    "load_json",
    "write_json_atomic",
    "DEFAULT_BASE_URL",
    "RobotsDecision",
    "check_robots",
    "VerifyReport",
    "run_verify",
]
