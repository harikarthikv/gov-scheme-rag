"""Headless UI tests using Streamlit's real script runner."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DemoRAG:
    def query(self, question):
        return {"answer": "The subsidy is 50,000 rupees [1].",
                "sources": [{"scheme_name": "Example", "filename": "example.pdf", "page": 1}],
                "similarity_scores": [0.015]}


class StreamlitTests(unittest.TestCase):
    def test_startup_and_chat_and_clear(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"),
                                default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Government scheme PDF assistant")
        app.session_state["rag"] = DemoRAG()
        app.run()
        app.chat_input[0].set_value("What is the subsidy?").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertIn("[1]", app.markdown[-1].value)
        self.assertTrue(any("[1] Example" in caption.value for caption in app.caption))
        app.button[0].click().run()
        self.assertEqual(app.session_state["chat_history"], [])


if __name__ == "__main__":
    unittest.main()
