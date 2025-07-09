import os
import sys
import types

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# Provide a minimal stub for streamlit to satisfy imports
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))

from i18n import TRANSLATIONS


def test_translation_keys_match():
    assert set(TRANSLATIONS["en"]) == set(TRANSLATIONS["zh"])
