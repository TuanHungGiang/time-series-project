"""Display the compact final preprocessing report in a Kaggle notebook."""
from pathlib import Path

REPORT = Path("/kaggle/working/z24_centroid_final/FINAL_PREPROCESSING_REPORT.md")
if not REPORT.is_file():
    raise FileNotFoundError(f"Run z24_centroid_final_kaggle.py first: {REPORT}")
try:
    from IPython.display import Markdown, display
    display(Markdown(REPORT.read_text(encoding="utf-8")))
except ImportError:
    print(REPORT.read_text(encoding="utf-8"))

