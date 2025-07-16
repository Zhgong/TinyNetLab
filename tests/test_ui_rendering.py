import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_attention_demo_runs():
    at = AppTest.from_file(os.path.join(ROOT, "attention_demo.py"))
    at.run()
    assert len(at.exception) == 0


def test_moons_streamlit_runs():
    at = AppTest.from_file(os.path.join(ROOT, "moons_streamlit.py"))
    at.run()
    assert len(at.exception) == 0


def test_tinynet_runs():
    at = AppTest.from_file(os.path.join(ROOT, "tinynet.py"))
    at.run()
    assert len(at.exception) == 0


def test_app_runs():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"))
    at.run()
    assert len(at.exception) == 0
