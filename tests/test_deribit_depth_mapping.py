from __future__ import annotations

import unittest

from src import collect_deribit_snapshot as collector


class DeribitDepthMappingTests(unittest.TestCase):
    def test_top_15_requests_supported_depth_20(self) -> None:
        self.assertEqual(collector.api_order_book_depth(15), 20)

    def test_supported_depth_is_left_unchanged(self) -> None:
        self.assertEqual(collector.api_order_book_depth(20), 20)


if __name__ == "__main__":
    unittest.main()
