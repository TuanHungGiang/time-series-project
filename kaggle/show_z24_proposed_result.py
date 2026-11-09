
"""Render the compact Z24 proposed-method report in Kaggle."""
from pathlib import Path

REPORT = Path("/kaggle/working/z24_modal_idk_niw/Z24_PROPOSED_REPORT.md")
if not REPORT.is_file(): raise FileNotFoundError(REPORT)
try:
    from IPython.display import Markdown, display
    display(Markdown(REPORT.read_text(encoding="utf-8")))
except ImportError:
    print(REPORT.read_text(encoding="utf-8"))

