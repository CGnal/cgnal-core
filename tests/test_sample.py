"""Sample test module."""
import unittest


class TestTrue(unittest.TestCase):
    """Sample test class."""

    def test_true(self):
        """Sample test method."""
        self.assertEqual(1, 1)


if __name__ == "__main__":
    unittest.main()
