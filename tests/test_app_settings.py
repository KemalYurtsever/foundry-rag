import tempfile
import unittest
from pathlib import Path

from foundry_rag.app_settings import AppSettings, load_app_settings, save_app_settings
from foundry_rag.theme import theme_palette


class AppSettingsTests(unittest.TestCase):
    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            expected = AppSettings(
                retrieval_preset="Custom",
                top_k="4",
                neighbors=1,
                answer_mode="Flexible",
                theme="Light",
            )

            save_app_settings(path, expected)

            self.assertEqual(load_app_settings(path), expected)

    def test_corrupt_settings_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.json"
            path.write_text("not json", encoding="utf-8")

            self.assertEqual(load_app_settings(path), AppSettings())

    def test_both_palettes_have_matching_keys(self):
        self.assertEqual(set(theme_palette("Dark")), set(theme_palette("Light")))


if __name__ == "__main__":
    unittest.main()
