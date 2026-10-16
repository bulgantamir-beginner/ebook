"""
Book page photos  ->  ONE searchable PDF (text is selectable / Ctrl+F works).

Every recognized word is written into the PDF as invisible text, placed at the
word's own position, size and rotation on the page image, so highlighting a
word in a PDF viewer highlights that word and nothing else.

Folder layout:
    app.py
    fonts/DejaVuSans.ttf
    pages/1.jpg, 2.jpg, ...        (any of .jpg .jpeg .png)

Run:  python app.py
Output: book.pdf, book.txt.  OCR results are cached in ocr_cache/, so if you
stop the script it resumes, and tweaking placement/corrections is instant.
"""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import glob
import json
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import getAscentDescent, stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ------------------------------------------------------------------ settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "pages")
OUTPUT_PDF = os.path.join(BASE_DIR, "book.pdf")
OUTPUT_TXT = os.path.join(BASE_DIR, "book.txt")
CACHE_DIR = os.path.join(BASE_DIR, "ocr_cache")
FONT_PATH = os.path.join(BASE_DIR, "fonts", "DejaVuSans.ttf")

LIMIT = None          # e.g. 1 to test on the first page only; None = all pages
ONLY_PAGES = None     # e.g. [14, 15] to test just those two pages, anywhere
                      # in the book, without touching the ones before them.
                      # Matches the number in each file's name (14.jpg -> 14).
                      # Takes priority over LIMIT when both are set.
PAGE_WIDTH_PT = 480   # PDF page width in points (height follows the photo)
MIN_SCORE = 0.5        # drop lines the recognizer is unsure about (also helps
                       # ignore faint text showing through from the other
                       # side). 0.7 turned out to also throw away real,
                       # correctly-read paragraphs on pages with smaller or
                       # denser print (whole-paragraph "skips" the recognizer
                       # itself wasn't to blame for) - 0.5 keeps the
                       # ghost-text filtering while dropping far less real
                       # text. main() prints how many lines each page loses
                       # to this so you can see if it's still too strict.
VISIBLE_TEXT = False  # True = draw the OCR text in RED to check alignment

SKIP_BLANK_PAGES = True  # detect near-blank pages and don't OCR them at all
BLANK_INK_RATIO = 0.002  # a page counts as "blank" when fewer than this
                         # fraction of pixels are dark enough to be ink.
                         # Real ghosting/show-through covers a page so
                         # lightly that this stays far below a printed page,
                         # which typically runs 0.02-0.1+.

CORRECTIONS_PATH = os.path.join(BASE_DIR, "corrections.txt")
FONT = "DejaVuSans"


def load_corrections(path):
    """Read corrections.txt into compiled regex rules.

    Each non-empty, non-'#' line is one rule:

        wrong -> right          whole-word match, case-sensitive
        /regex/ -> right        raw regex, applied wherever it matches

    Whole-word rules only replace `wrong` when it isn't glued to other
    letters, so a rule can't corrupt part of a longer, otherwise-correct
    word. Regex rules (the /.../ form) have no such guard - useful for
    things like matching every capitalization of ХХК at once - so keep
    those patterns specific. Blank lines and lines starting with # are
    ignored.
    """
    rules = []
    if not os.path.exists(path):
        return rules
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                print(f"[corrections.txt:{lineno}] skipped, no '->' : {raw!r}")
                continue
            left, right = line.split("->", 1)
            left, right = left.strip(), right.strip()
            if len(left) >= 2 and left.startswith("/") and left.endswith("/"):
                pattern = left[1:-1]
            else:
                pattern = r"(?<!\w)" + re.escape(left) + r"(?!\w)"
            try:
                rules.append((re.compile(pattern), right))
            except re.error as e:
                print(f"[corrections.txt:{lineno}] bad regex, skipped: {e}")
    return rules


RULES = load_corrections(CORRECTIONS_PATH)


# --------------------------------------------------------------- page images
def natural_key(path):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def page_number(path):
    m = re.search(r"\d+", os.path.basename(path))
    return int(m.group()) if m else None


def list_pages():
    files = []
    for ext in ("jpg", "jpeg", "png"):
        files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext}"))
        files += glob.glob(os.path.join(INPUT_DIR, f"*.{ext.upper()}"))
    files = sorted(set(files), key=natural_key)   # 2.jpg before 10.jpg
    if ONLY_PAGES:
        wanted = set(ONLY_PAGES)
        picked = [f for f in files if page_number(f) in wanted]
        found = {page_number(f) for f in picked}
        missing = wanted - found
        if missing:
            print(f"[note] ONLY_PAGES asked for {sorted(missing)}, "
                  f"no matching file found in {INPUT_DIR}")
        return picked
    return files[:LIMIT] if LIMIT else files


def upright_image(path):
    """(path, width, height) of an upright version of the photo. Phone photos
    can carry an EXIF rotation flag that PDFs ignore, so bake it in."""
    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) in (0, 1):
            return path, im.width, im.height
        fixed = ImageOps.exif_transpose(im).convert("RGB")
        folder = os.path.join(CACHE_DIR, "_upright")
        os.makedirs(folder, exist_ok=True)
        out = os.path.join(folder, os.path.basename(path))
        fixed.save(out, quality=95)
        return out, fixed.width, fixed.height


def is_blank_page(path, ink_ratio_threshold=BLANK_INK_RATIO):
    """True if the page has ~no real print on it. Ghost text bleeding
    through from the other side of thin paper is much fainter and much
    thinner-stroked than the darkest ink actually printed on the visible
    side, so an adaptive per-page dark threshold catches real printed pages
    (even lightly-exposed photos of them) while still calling a page with
    only show-through 'blank'."""
    gray = np.array(Image.open(path).convert("L"))
    # cut the outer margin: edge shadows/staple holes/binding gutter can
    # look like ink and would otherwise throw this off on a truly blank page
    my, mx = int(gray.shape[0] * 0.04), int(gray.shape[1] * 0.04)
    gray = gray[my: gray.shape[0] - my, mx: gray.shape[1] - mx]

    paper_tone = float(np.percentile(gray, 96))         # the page's own "white"
    dark_cut = max(0.0, min(255.0, paper_tone - 40.0))   # meaningfully darker than that
    ink_ratio = float(np.mean(gray < dark_cut))
    return ink_ratio < ink_ratio_threshold, ink_ratio


# ---------------------------------------------------------------------- OCR
def make_ocr():
    from paddleocr import PaddleOCR
    kwargs = dict(
        use_doc_orientation_classify=False,   # pages are already cropped
        use_doc_unwarping=False,              # and flat: don't let Paddle warp them
        use_textline_orientation=True,        # upside-down / sideways labels
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
        enable_mkldnn=False,
        cpu_threads=8,
    )
    try:
        return PaddleOCR(return_word_box=True, **kwargs)   # per-word positions
    except Exception as e:
        print(f"[note] word boxes unavailable ({e}); words will be positioned "
              f"proportionally inside each line instead.")
        return PaddleOCR(**kwargs)


def _word_info(text, words, regions, i):
    """Per-word polygons for line i, or None if missing / inconsistent."""
    try:
        w, r = words[i], regions[i]
        if not w or len(w) != len(r):
            return None
        if "".join(w).replace(" ", "") != text.replace(" ", ""):
            return None
        out = []
        for wt, poly in zip(w, r):
            p = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(p) < 2 or not np.isfinite(p).all():
                return None
            out.append({"text": wt, "poly": p.tolist()})
        return out
    except Exception:
        return None


# Some pages carry a faint repeating stamp/watermark (e.g. an ISO-style
# "PLAN -> DO -> CHECK -> ACT" cycle) tiled across the page. PaddleOCR reads
# it like real text - sometimes as its own line, sometimes glued onto a
# real word's line (e.g. "ХЭРЭГЖҮҮЛЭХDO"). Add more of the stamp's own
# words here if a different page's watermark uses different text.
WATERMARK_WORDS = ["PDCA", "CHECK", "PLAN", "ACT", "DO"]
WATERMARK_RE = re.compile(
    "|".join(re.escape(w) for w in sorted(WATERMARK_WORDS, key=len, reverse=True)),
    re.IGNORECASE,
)


def strip_watermark(text):
    """Remove watermark words wherever they appear - standalone or glued
    onto a real word - and collapse the whitespace left behind."""
    cleaned = WATERMARK_RE.sub("", text)
    # Overlapping copies of the stamp (e.g. "DO" printed over "CHECK") can
    # leave a short, STANDALONE Latin fragment behind, like the "CH" left
    # over from "CHCHECK" once "CHECK" is removed. \b doesn't distinguish
    # Cyrillic from Latin, so this only matches a fragment set off by
    # whitespace/punctuation on both sides - never one still glued to a
    # real Cyrillic word - keeping this conservative.
    cleaned = re.sub(r"\b[A-Za-z]{1,2}\b", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def normalize_result(res):
    """One PaddleOCR result -> plain list of lines (JSON-safe)."""
    texts = list(res.get("rec_texts", []))
    polys = res.get("rec_polys")
    if polys is None or len(polys) == 0:
        polys = res.get("dt_polys", [])
    scores = res.get("rec_scores")
    angles = res.get("textline_orientation_angles")
    words, regions = res.get("text_word"), res.get("text_word_region")

    lines, dropped = [], []
    for i, (text, poly) in enumerate(zip(texts, polys)):
        if not text.strip():
            continue
        text = strip_watermark(text)
        if not text:
            continue   # the whole line was just watermark noise
        if scores is not None and i < len(scores) and float(scores[i]) < MIN_SCORE:
            dropped.append((text, float(scores[i])))
            continue
        flipped = (angles is not None and len(angles) == len(texts)
                   and int(angles[i]) == 180)
        lines.append({
            "text": text,
            "poly": np.asarray(poly, dtype=float).reshape(-1, 2).tolist(),
            "flipped": bool(flipped),
            "words": _word_info(text, words, regions, i) if words is not None
                     and regions is not None else None,
        })
    if dropped:
        lines_out = sorted(dropped, key=lambda x: x[1])[:5]
        print(f"    [low score, dropped {len(dropped)} line(s)] " +
              " | ".join(f"{t!r} ({s:.2f})" for t, s in lines_out))
    return lines


def ocr_page(ocr, path):
    rgb = np.array(Image.open(path).convert("RGB"))
    lines = []
    for res in ocr.predict(rgb[:, :, ::-1].copy()):   # PaddleOCR expects BGR
        lines += normalize_result(res)
    return lines


# ------------------------------------------------------------------ geometry
def line_frame(poly, flipped):
    """Reading direction d and 'up' u (unit vectors, image coords, y down),
    plus the line's extent along d and across it.

    Mirrors how PaddleOCR reads a box: a box at least 1.5x taller than wide is
    read top-to-bottom; a 'flipped' (180 deg) box is read the other way."""
    p = np.asarray(poly, dtype=float)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p.astype(np.float32)))
    # name the corners by where they really are on the page
    s, dif = p.sum(axis=1), np.diff(p, axis=1).ravel()
    tl, br = p[np.argmin(s)], p[np.argmax(s)]
    tr, bl = p[np.argmin(dif)], p[np.argmax(dif)]
    p = np.array([tl, tr, br, bl])
    width = max(np.linalg.norm(tl - tr), np.linalg.norm(bl - br), 1e-6)
    height = max(np.linalg.norm(tl - bl), np.linalg.norm(tr - br), 1e-6)
    if height / width >= 1.5:
        d = (bl - tl) + (br - tr)          # vertical line: top -> bottom
    else:
        d = (tr - tl) + (br - bl)          # horizontal line: left -> right
    d = d / np.linalg.norm(d)
    if flipped:
        d = -d
    u = np.array([d[1], -d[0]])            # d rotated to point "above the text"
    t, s = p @ d, p @ u
    return d, u, t.min(), t.max(), s.min(), s.max()


def split_words(line, d, t_min, t_max):
    """[(word, t0, t1)] along the reading direction, never overlapping."""
    text = fix(line["text"])
    out = []

    # exact per-word boxes from PaddleOCR (horizontal, unflipped lines only)
    if line.get("words") and not line["flipped"] and abs(d[0]) > 0.7:
        ranges, ok = [], True
        for w in line["words"]:
            t = np.asarray(w["poly"], dtype=float) @ d
            ranges.append((fix(w["text"]), float(t.min()), float(t.max())))
        tol = 0.05 * (t_max - t_min) + 2
        prev = t_min - tol
        for word, t0, t1 in ranges:
            if t0 < t_min - tol or t1 > t_max + tol or t0 < prev - tol:
                ok = False            # boxes don't fit this line -> distrust
                break
            prev = t1
        if ok:
            last = t_min
            for word, t0, t1 in ranges:
                t0 = max(t0, last)    # guarantee: no word overlaps the previous
                t1 = max(t1, t0 + 0.5)
                out.append((word, t0, t1))
                last = t1
            return out

    # fallback: share the line's length between words by glyph width
    total = stringWidth(text, FONT, 10)
    if total <= 0:
        return out
    scale = (t_max - t_min) / total
    for m in re.finditer(r"\S+", text):
        a = stringWidth(text[:m.start()], FONT, 10)
        b = stringWidth(text[:m.end()], FONT, 10)
        out.append((m.group(), t_min + a * scale, t_min + b * scale))
    return out


def fix(text):
    # Re-applying strip_watermark here (not just in normalize_result) means
    # pages whose OCR result was cached BEFORE the watermark filter existed
    # still get cleaned, every run, with no need to delete ocr_cache/ and
    # re-OCR anything - old cached JSON files only ever hold raw recognizer
    # output, and this is the one place all of it passes through on its way
    # into the PDF/book.txt.
    text = strip_watermark(text)
    for pattern, right in RULES:
        text = pattern.sub(right, text)
    return text


# ------------------------------------------------------------------ PDF text
def force_render_mode(tx, mode):
    """Set the PDF text-render mode (Tr), unconditionally writing the
    operator into the content stream.

    ReportLab's own tx.setTextRenderMode() only emits 'N Tr' when N differs
    from that particular text-object instance's assumed starting mode
    (always 0, freshly per object) - not from the real PDF graphics state,
    which is what a viewer actually uses and which does carry over from the
    previous text object's ET. Since every line here begins a brand-new
    text object, that mismatch means a call meant to make text visible can
    be silently dropped whenever the previous object last set mode 3,
    leaving the text invisible despite the code appearing to ask for mode
    0. Writing the operator directly avoids relying on that assumption."""
    tx._code.append(f"{mode} Tr")
    tx._textRenderMode = mode


def draw_run(c, text, t0, t1, d, u, s_base, fs, k, page_h_pt):
    """Invisible text `text` that starts at along-line position t0 and is
    stretched to end at t1. k = image pixels -> PDF points."""
    natural = stringWidth(text, FONT, fs)
    width_pt = (t1 - t0) * k
    if natural <= 0 or width_pt <= 0:
        return
    px, py = t0 * d + s_base * u                       # image coords
    ex, ey = px * k, page_h_pt - py * k                # PDF coords (y up)
    tx = c.beginText()
    tx.setTextTransform(d[0], -d[1], u[0], -u[1], ex, ey)   # rotate + move
    tx.setFont(FONT, fs)
    tx.setHorizScale(max(1.0, 100.0 * width_pt / natural))
    if VISIBLE_TEXT:
        force_render_mode(tx, 0)
        tx.setFillColorRGB(1, 0, 0)
    else:
        force_render_mode(tx, 3)                        # invisible, selectable
    tx.textOut(text)
    c.drawText(tx)


def draw_separator(c, sep, end_x, end_y, d, u, fs, k, page_h_pt):
    """An invisible run containing just `sep` (a space or a newline),
    placed right after the line's last word rather than at the page
    origin. PDF text extractors partly infer line breaks from how far a
    text run's position jumps from the previous one; a marker teleported
    to (0, 0) creates a large jump of its own and gets its intended
    character overridden by the extractor's own break heuristic. Keeping
    it in place, right where the line actually ends, lets our character
    (space for a wrap, newline for a real line end) actually take effect."""
    ex, ey = end_x * k, page_h_pt - end_y * k
    tx = c.beginText()
    tx.setTextTransform(d[0], -d[1], u[0], -u[1], ex, ey)
    tx.setFont(FONT, fs)
    force_render_mode(tx, 3)
    tx.textOut(sep)
    c.drawText(tx)


def draw_line(c, line, k, page_h_pt, sep):
    d, u, t_min, t_max, s_min, s_max = line_frame(line["poly"], line["flipped"])
    thickness_pt = (s_max - s_min) * k
    if thickness_pt < 1 or t_max - t_min < 1:
        return [], 0.0
    # Font size chosen so the font's ascent+descent box == the detected box
    # height: viewer selection rectangles then match the printed line.
    ascent, descent = getAscentDescent(FONT, 1.0)
    fs = thickness_pt / (ascent - descent)
    s_base = s_min + (-descent * fs) / k               # baseline above box bottom

    words = split_words(line, d, t_min, t_max)
    for i, (word, t0, t1) in enumerate(words):
        draw_run(c, word, t0, t1, d, u, s_base, fs, k, page_h_pt)
        if i + 1 < len(words) and words[i + 1][1] - t1 > 1e-3:
            # the real gap between two words, as a space, so copy/paste keeps
            # them apart without any glyph poking into a neighbouring word
            draw_run(c, " ", t1, words[i + 1][1], d, u, s_base, fs, k, page_h_pt)

    end_t = words[-1][2] if words else t_max
    end_xy = end_t * d + s_base * u
    draw_separator(c, sep, end_xy[0], end_xy[1], d, u, fs, k, page_h_pt)
    return [w for w, _, _ in words], (t_max - t_min)


# A line whose own width is at least this fraction of the page's widest
# line is treated as a paragraph wrapping into the next line, so it joins
# to what follows with a space, not a line break, when the text is copied.
# A shorter line - the true end of a paragraph, a heading, a label - still
# gets a real line break. This mirrors how word processors decide where a
# paragraph actually ends: by whether a line runs out to the margin.
WRAP_WIDTH_RATIO = 0.85


def _row_overlap(a, b):
    """1-D overlap fraction of two (lo, hi) ranges, relative to the
    narrower one - 0 if they don't overlap at all."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


# A narrow side box (a pull-quote, a callout, a table cell) sitting next to
# the main text column gets detected by PaddleOCR as its own separate lines,
# but its OWN ordering of all the lines on the page reads roughly row by
# row down the whole page - so a quote box's line 1, the paragraph's line
# 1, the quote box's line 2, the paragraph's line 2, ... come out
# interleaved, never with the quote box's own lines next to each other.
# paragraph_separators() only ever looks at two *list-adjacent* lines, so
# with that interleaving it can't tell "this is the same block continuing"
# from "this is a different column that merely lines up at this height."
#
# A first attempt grouped lines into left-to-right bands by x-overlap alone,
# globally across the whole page. That broke on a real page: a diagram with
# many small scattered labels sits above the quote box and paragraph, and
# with no notion of vertical position at all, SOME label's x-range always
# ends up touching both the quote box's column and the paragraph's column
# (transitively, through other labels), chaining everything - the entire
# page, diagram included - into one giant "band" and leaving the original
# row-by-row scramble untouched. Tested directly against that page's real
# OCR coordinates, confirming this failure before the fix below.
def _merged_intervals(ranges, tol=0.0):
    """Merge sorted (a, b) ranges that touch, overlap, or are within `tol`
    of each other."""
    ranges = sorted(ranges)
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1] + tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _split_axis(indices, boxes, axis):
    """Group `indices` by which gap-separated interval their box falls into
    along `axis` (0 = x, 1 = y). None if there's no real gap on this axis -
    every box's range touches the next, so there's nothing to cut here.

    On the y-axis, ordinary row-to-row spacing within a multi-line block
    can itself look like a 'gap' once only SOME of several side-by-side
    columns still have content left (a short column ends while a taller
    one continues) - tested directly against a real page: once two short
    columns ran out, the remaining rows of the two taller ones were only
    1-13px apart, while every genuine section break on that same page
    (between the diagram, the table, the heading, the body text) was
    100-700px. A flat pixel tolerance wouldn't generalize across photos
    taken at different resolutions/zoom, so the tolerance instead scales
    with this group's own median line height - comfortably larger than
    ordinary row spacing, nowhere near a real section gap. The x-axis
    doesn't get this: a real column gutter and incidental word spacing
    are much closer in scale, so softening that test risks re-merging
    genuinely separate columns (the bug fixed earlier)."""
    ranges = [(boxes[i][axis * 2], boxes[i][axis * 2 + 1]) for i in indices]
    tol = 0.0
    if axis == 1:
        heights = [boxes[i][3] - boxes[i][2] for i in indices]
        heights.sort()
        median_h = heights[len(heights) // 2] if heights else 0.0
        tol = 0.5 * median_h
    merged = _merged_intervals(ranges, tol)
    if len(merged) < 2:
        return None
    groups = [[] for _ in merged]
    for i in indices:
        v0, v1 = boxes[i][axis * 2], boxes[i][axis * 2 + 1]
        for gi, (m0, m1) in enumerate(merged):
            if v0 >= m0 - 1e-6 and v1 <= m1 + 1e-6:
                groups[gi].append(i)
                break
    groups = [g for g in groups if g]
    return groups if len(groups) >= 2 else None


def _xy_cut(indices, boxes):
    """Classic document-layout XY-cut: recursively split a set of lines by
    whichever axis gives the COARSER split (fewer resulting groups) -
    vertical (stacked blocks: a diagram, then a heading, then a body
    section, top to bottom) or horizontal (side-by-side blocks: a quote
    box and the paragraph next to it, left then right, each internally
    still top to bottom). Unlike a single global x-overlap pass, every cut
    here is scoped to just the indices being split at that step, so a gap
    that only exists locally (a quote box's gutter) can't be bridged by
    unrelated lines elsewhere on the page.

    Trying vertical first, unconditionally, breaks on an actual table: a
    row of side-by-side cells is itself a 'gap' on the vertical axis (a
    small y-gap before the next row), so a y-first rule finds and takes
    that cut before ever considering that the real structure is columns,
    and ends up splitting the table row by row - reading column A's first
    cell, then column B's, then back to column A, forever interleaved
    instead of each column coming out as one continuous block. Comparing
    both axes and taking whichever leaves FEWER groups favors the coarser,
    more structural division - a handful of stacked sections, or a
    handful of side-by-side columns - over a fine chop into every single
    row, which is what a table's rows look like under a y-only test. Ties
    go to vertical, since plain stacking (section after section down a
    normal page) is the more common case and should win when a page
    doesn't look like a table at all."""
    if len(indices) <= 1:
        return list(indices)
    gy = _split_axis(indices, boxes, axis=1)
    gx = _split_axis(indices, boxes, axis=0)
    if gy and gx:
        groups = gy if len(gy) <= len(gx) else gx
    else:
        groups = gy or gx
    if groups:
        out = []
        for g in groups:
            out.extend(_xy_cut(g, boxes))
        return out
    return sorted(indices, key=lambda i: boxes[i][2])   # base case: top to bottom


def reading_order(lines):
    """Reorder `lines` into real reading order via XY-cut, instead of
    PaddleOCR's own row-by-row-across-the-whole-page order."""
    if len(lines) <= 1:
        return lines
    boxes = []
    for line in lines:
        p = np.asarray(line["poly"], dtype=float)
        boxes.append((p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()))
    order = _xy_cut(list(range(len(lines))), boxes)
    return [lines[i] for i in order]


def paragraph_separators(lines, k):
    """One separator (' ' or '\\n') per line in `lines`, deciding for each
    whether it continues into the next line (wrap) or ends there.

    A wrap-join is only ever allowed between two lines whose x-ranges
    actually OVERLAP - i.e. the same reading column, by the same test
    reading_order() uses to tell columns apart in the first place. Width
    alone isn't enough to gate this: once lines are in correct column-by-
    column order (after reading_order()), the END of a short column sits
    right next to the START of the next column in the list, and that next
    column's first line can easily be wide enough, relative to the page's
    widest line, to look like a 'wrap' by width - even though it's a
    completely different cell/column and the two were never at the same
    row either. Requiring real x-overlap rules that out directly, instead
    of trying to special-case every way two unrelated lines can end up
    list-adjacent."""
    widths = []
    boxes = []   # (x_min, x_max, y_min, y_max) in raw image pixels
    for line in lines:
        d, u, t_min, t_max, s_min, s_max = line_frame(line["poly"], line["flipped"])
        widths.append((t_max - t_min) * k)
        p = np.asarray(line["poly"], dtype=float)
        boxes.append((p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()))
    max_w = max(widths) if widths else 0.0
    seps = []
    for i, w in enumerate(widths):
        is_last = (i + 1 == len(widths))
        wraps = False
        if not is_last and max_w > 0 and w >= WRAP_WIDTH_RATIO * max_w:
            x0, x1, _, _ = boxes[i]
            nx0, nx1, _, _ = boxes[i + 1]
            same_column = min(x1, nx1) > max(x0, nx0)   # real overlap
            wraps = same_column
        seps.append(" " if wraps else "\n")
    return seps


def build_pdf(pages, all_lines):
    pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))
    c = canvas.Canvas(OUTPUT_PDF)
    text_out = []
    for n, (path, lines) in enumerate(zip(pages, all_lines), 1):
        lines = reading_order(lines)
        img_path, w, h = upright_image(path)
        k = PAGE_WIDTH_PT / w
        page_h = h * k
        c.setPageSize((PAGE_WIDTH_PT, page_h))
        c.drawImage(img_path, 0, 0, width=PAGE_WIDTH_PT, height=page_h)  # JPEGs embed untouched
        text_out.append(f"--- Page {n}: {os.path.basename(path)} ---")

        seps = paragraph_separators(lines, k)
        paragraph = []   # words accumulated across wrapped lines, not yet flushed
        for line, sep in zip(lines, seps):
            words, _ = draw_line(c, line, k, page_h, sep)
            if words:
                paragraph.extend(words)
            if sep == "\n" and paragraph:
                text_out.append(" ".join(paragraph))
                paragraph = []
        if paragraph:
            text_out.append(" ".join(paragraph))

        text_out.append("")
        c.showPage()
    c.save()
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(text_out))


# ---------------------------------------------------------------------- main
def main():
    pages = list_pages()
    if not pages:
        raise SystemExit(f"No images found in {INPUT_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"{len(pages)} pages")

    ocr, all_lines = None, []
    for n, path in enumerate(pages, 1):
        cache = os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(path))[0] + ".json")
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as f:
                lines = json.load(f)
            print(f"[{n}/{len(pages)}] {os.path.basename(path)}: cached")
        else:
            img_path = upright_image(path)[0]
            if SKIP_BLANK_PAGES:
                blank, ink_ratio = is_blank_page(img_path)
                if blank:
                    lines = []
                    with open(cache, "w", encoding="utf-8") as f:
                        json.dump(lines, f, ensure_ascii=False)
                    print(f"[{n}/{len(pages)}] {os.path.basename(path)}: "
                          f"blank (ink ratio {ink_ratio:.4f}), OCR skipped")
                    all_lines.append(lines)
                    continue
            if ocr is None:
                ocr = make_ocr()
            t = time.time()
            lines = ocr_page(ocr, img_path)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump(lines, f, ensure_ascii=False)
            print(f"[{n}/{len(pages)}] {os.path.basename(path)}: "
                  f"{len(lines)} lines, {time.time() - t:.0f}s")
        all_lines.append(lines)

    build_pdf(pages, all_lines)
    print(f"\nDone: {OUTPUT_PDF}\n      {OUTPUT_TXT}")


if __name__ == "__main__":
    main()