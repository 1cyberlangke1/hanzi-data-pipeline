"""本机配置读取的单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hanzi_synth.config import ConfigError, ProviderConfig, default_data_path, load_providers

TOML_OK = """
[[providers]]
name = "edgefn"
base_url = "https://api.example.com/v1/"
model = "m1"
api_key = "sk-secret"
role = "verbalizer"
rpm = 12
batch = 2

[[providers]]
name = "broken"

[[providers]]
name = "judge"
base_url = "https://judge.example.com/v1"
model = "m2"
api_key = "sk-2"
role = "judge"
"""


class LoadProvidersTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "secrets.toml"

    def test_parses_and_normalizes(self) -> None:
        """base_url 去掉尾斜杠，数值字段转成 int，缺省字段有默认值。"""
        self.path.write_text(TOML_OK, encoding="utf-8")
        providers = load_providers(self.path)
        self.assertEqual([p.name for p in providers], ["edgefn", "judge"])
        self.assertEqual(providers[0].base_url, "https://api.example.com/v1")
        self.assertEqual(providers[0].rpm, 12)
        self.assertEqual(providers[0].batch, 2)
        self.assertEqual(providers[1].rpm, 0)

    def test_role_filter(self) -> None:
        """role 过滤只留下匹配的 provider。"""
        self.path.write_text(TOML_OK, encoding="utf-8")
        self.assertEqual([p.name for p in load_providers(self.path, role="judge")], ["judge"])

    def test_missing_file_raises_config_error(self) -> None:
        """凭据文件缺失时报人话，而不是抛 FileNotFoundError。"""
        with self.assertRaises(ConfigError):
            load_providers(Path(self._tmp.name) / "nope.toml")

    def test_no_matching_role_raises(self) -> None:
        """role 过滤后为空同样报错，避免调用方拿到空列表。"""
        self.path.write_text(TOML_OK, encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_providers(self.path, role="nonexistent")

    def test_bad_toml_raises_config_error(self) -> None:
        """TOML 语法错误包成 ConfigError。"""
        self.path.write_text("[[providers]\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_providers(self.path)

    def test_repr_masks_key(self) -> None:
        """repr 里密钥必须被遮掉，防止打进日志。"""
        provider = ProviderConfig(name="p", base_url="http://x", model="m", api_key="sk-secret")
        self.assertNotIn("sk-secret", repr(provider))
        self.assertIn("***", repr(provider))

    def test_default_data_path_points_at_crawler(self) -> None:
        """默认数据路径指向 crawler 产出。"""
        self.assertEqual(default_data_path().name, "chars.jsonl")
        self.assertEqual(default_data_path().parent.name, "data")


if __name__ == "__main__":
    unittest.main()
