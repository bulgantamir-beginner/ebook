"""
Find likely OCR mistakes without reading the whole book.

Reads every cached page (ocr_cache/*.json, made by make_searchable_pdf.py),
counts how often each word occurs, and prints them rarest-first. Real words
(names, terms) tend to repeat many times across 100 pages; one-off garbled
strings usually stick out near the top of this list.

Run AFTER make_searchable_pdf.py has OCR'd at least a few pages (it reads
the cache, so it works even mid-run and needs no PaddleOCR itself).

Usage:
    python word_report.py            # print the report
    python word_report.py 3          # only show words seen <= 3 times
"""
import glob
import json
import os
import re
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "ocr_cache")

# Same corrections your PDF build applies, so already-fixed words don't
# clutter the report.
sys.path.insert(0, BASE_DIR)
from app import load_corrections, CORRECTIONS_PATH  # noqa: E402

RULES = load_corrections(CORRECTIONS_PATH)


def fix(text):
    for pattern, right in RULES:
        text = pattern.sub(right, text)
    return text


def main():
    max_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    counts = Counter()
    locations = {}  # word -> first "page: line text" seen, for context
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No cached pages in {CACHE_DIR} - run "
                          f"make_searchable_pdf.py first (even partially).")

    for path in files:
        page = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            lines = json.load(f)
        for line in lines:
            text = fix(line["text"])
            for word in re.findall(r"\S+", text):
                counts[word] += 1
                locations.setdefault(word, f"page {page}: {text}")

    rare = sorted((w for w in counts if counts[w] <= max_count),
                  key=lambda w: (counts[w], w))

    print(f"{len(files)} pages scanned, {len(counts)} distinct words.\n"
          f"{len(rare)} words appear {max_count} time(s) or fewer "
          f"(most likely to contain OCR mistakes):\n")
    for word in rare:
        print(f"  {counts[word]:>2}x  {word!r:30s}  e.g. {locations[word]}")

    print(f"\nTip: increase the cutoff for a longer list, e.g. "
          f"'python word_report.py 8'. Add real fixes to corrections.txt, "
          f"then re-run make_searchable_pdf.py to rebuild the PDF text "
          f"layer (no re-OCR needed).")


if __name__ == "__main__":
    main()