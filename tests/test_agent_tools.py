from __future__ import annotations

import unittest

from src.agent.tools import fetch_hkbu_updates, fetch_live_page


class AgentToolsTests(unittest.TestCase):
    def test_fetch_live_page_rejects_non_http_url(self) -> None:
        with self.assertRaises(ValueError):
            fetch_live_page("ftp://example.com")

    def test_fetch_hkbu_updates_calls_underlying_fetcher(self) -> None:
        import src.agent.tools as tools_module

        calls: list[str] = []
        original = tools_module.fetch_live_page

        def fake_fetch(url: str, **_kwargs: object) -> str:
            calls.append(url)
            return f"snapshot:{url}"

        try:
            tools_module.fetch_live_page = fake_fetch
            result = fetch_hkbu_updates(
                timetable_url="https://example.com/timetable",
                news_url="https://example.com/news",
            )
        finally:
            tools_module.fetch_live_page = original

        self.assertEqual(calls, ["https://example.com/timetable", "https://example.com/news"])
        self.assertEqual(result["timetable"], "snapshot:https://example.com/timetable")
        self.assertEqual(result["news"], "snapshot:https://example.com/news")


if __name__ == "__main__":
    unittest.main()