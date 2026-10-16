"""
Run this from your project folder (where app.py and ocr_cache/ live):

    python debug_page.py <page number>

It imports reading_order() and paragraph_separators() DIRECTLY from your
app.py (not a copy - so it always reflects whatever is actually in your
app.py right now), runs them on that page's real cached OCR data, and
prints:
  1. the raw lines as PaddleOCR detected them
  2. the order/columns after reading_order()
  3. the separator ('wrap' or 'break') chosen between every pair of lines
  4. the paragraphs that would be written to book.txt for this page

If this still shows the bug, the problem is visible right here, in this
output - paste it back and it'll show exactly which decision is wrong,
instead of guessing from a screenshot.
"""
import glob
import importlib.util
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "ocr_cache")
APP_PATH = os.path.join(BASE_DIR, "app.py")


def load_app():
    spec = importlib.util.spec_from_file_location("app", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def find_cache(page_number):
    matches = []
    for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        stem = os.path.splitext(os.path.basename(path))[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits == str(page_number):
            matches.append(path)
    return matches


def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_page.py <page number>")
        return
    page = sys.argv[1]
    matches = find_cache(page)
    if not matches:
        print(f"No ocr_cache/*.json found matching page number {page}")
        print("Files available:", os.listdir(CACHE_DIR)[:20])
        return

    app = load_app()

    for path in matches:
        print(f"=== {path} ===")
        with open(path, encoding="utf-8") as f:
            lines = json.load(f)
        print(f"{len(lines)} raw lines (as PaddleOCR / cache has them):")
        for i, line in enumerate(lines):
            import numpy as np
            p = np.asarray(line["poly"], dtype=float)
            x0, x1 = p[:, 0].min(), p[:, 0].max()
            y0, y1 = p[:, 1].min(), p[:, 1].max()
            print(f"  [{i:2d}] x:{x0:7.1f}-{x1:7.1f}  y:{y0:7.1f}-{y1:7.1f}  {line['text']!r}")

        ordered = app.reading_order(lines)
        print("\nAFTER reading_order():")
        for line in ordered:
            print(f"  {line['text']!r}")

        k = 1.0
        seps = app.paragraph_separators(ordered, k)
        print("\nSeparator decisions (wrap=' ' joins into next line, break='\\n' ends it):")
        for line, sep in zip(ordered, seps):
            label = "WRAP " if sep == " " else "BREAK"
            print(f"  {label}  {line['text']!r}")

        print("\nReconstructed book.txt paragraphs for this page:")
        paragraph = []
        for line, sep in zip(ordered, seps):
            text = app.fix(line["text"])
            paragraph.append(text)
            if sep == "\n":
                print(f"  >>> {' '.join(paragraph)}")
                paragraph = []
        if paragraph:
            print(f"  >>> {' '.join(paragraph)}")
        print()


if __name__ == "__main__":
    main()
