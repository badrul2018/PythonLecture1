"""Zero-shot Urdu (Nastaliq) + Arabic OCR with Qwen2.5-VL-7B-Instruct.

One module holds the whole pipeline so it can be tested on a CPU and pasted into Colab:

    page (300 DPI) -> deskew -> column / zone split (projection profile) -> tall-crop split
    -> Qwen2.5-VL reads each crop as Markdown (batched, greedy, per-token confidence)
    -> loop / truncation retries -> [?] marks for low-confidence words
    -> Urdu-only letter normalisation (vocalised Arabic is never touched)
    -> review queue -> per-page cache -> Markdown + searchable PDF (+ bookmarks, optional
       English gloss of the hawashi, optional webhook alerts)

Carried over from the earlier Qaari/Qari notebook: deskew, loop detection, token-probability
[?] flags, per-page cache with resume, conservative letter normalisation (the review screen
lives in the notebook).
The GPU parts import torch / transformers lazily, so everything else runs without them.
"""
import codecs
import json
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

try:
    import pymupdf
except ImportError:  # PyMuPDF < 1.24
    import fitz as pymupdf


# ============================================================================ settings
@dataclass
class OCRConfig:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantize: str = "auto"            # auto | none | 4bit   (auto = 4-bit only on GPUs under 20 GB)
    dpi: int = 300
    max_pixels: int = 2048 * 28 * 28  # largest image the vision encoder sees (~1.6 MP)
    min_scale: float = 0.6            # crops taller than this allows are cut at blank rows first
    columns: str = "auto"             # auto | 1 | 2
    max_gutter_frac: float = 0.08     # wider central gaps are poetry, not a column gutter
    max_new_tokens: int = 3072
    batch_size: int = 2
    loop_retry_penalty: float = 1.1
    conf_flag: float = 0.4            # a word whose weakest token has p < this gets [?] (0 = off)
    flag_mark: str = "[?]"
    normalize_unclear: str = "keep"   # lines that are neither clearly Urdu nor Arabic: keep | urdu
    review_min_yield: float = 0.15    # output letters / letters expected from the ink
    review_flag_ratio: float = 0.08   # share of words marked [?]
    page_markers: bool = True
    bookmarks: str = "headings"       # headings | pages | off
    translate_hawashi: bool = False
    webhook_url: str = ""
    webhook_every: int = 25           # pages between progress pings (0 = start/end/errors only)


SYSTEM_PROMPT = (
    "You are an expert transcriber of historical Urdu Nastaliq and classical Arabic printed books "
    "and lithographs. Transcribe the text in the image exactly as written, in natural reading order "
    "(right to left, top to bottom), as clean Markdown.\n"
    "Rules:\n"
    "- Output only the transcription: no commentary, no translation, no English.\n"
    "- Use # or ## only for real headings that are set apart on the page.\n"
    "- Preserve every Arabic vowel mark (harakat/tashkeel) exactly as printed and never add marks "
    "that are not printed. Urdu Nastaliq is normally unvocalised: leave it so.\n"
    "- Use Urdu letters for Urdu text (ک ی ہ ھ ے) and Arabic letters for Arabic text.\n"
    "- Poetry: write each hemistich on its own line, right hemistich first, and leave a blank line "
    "between couplets.\n"
    "- Skip the running header and the page number.\n"
    "- Marginal notes (hawashi) and footnotes: transcribe them after the main text under a line "
    "'### Hawashi'. If there are none, do not write that heading.\n"
    "- If a word cannot be read, write [?] in its place."
)
USER_PROMPT = "Transcribe this page image."


# ============================================================================ layout
@dataclass
class Region:
    kind: str                         # full | right | left
    bbox: tuple                       # x0, y0, x1, y1 in deskewed page pixels
    est_letters: int = 0


@dataclass
class PageLayout:
    gray: np.ndarray                  # deskewed page
    ink: np.ndarray                   # cleaned text mask
    angle: float
    ch: float                         # typical glyph height (px)
    text_box: tuple
    regions: list
    gutter: object                    # (g0, g1) or None
    notes: list


def binarize(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bs = max(31, (min(gray.shape) // 25) | 1)
    ad = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs, 12)
    return (ad > 0) & (g < min(otsu_t + 25, 235))


def estimate_skew(ink, max_deg=3.0):
    h, w = ink.shape
    s = 1000.0 / max(h, w)
    small = cv2.resize(ink.astype(np.uint8) * 255, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    c = (small.shape[1] / 2.0, small.shape[0] / 2.0)

    def score(a):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(small, M, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.sum(np.diff(r.sum(1).astype(np.float64)) ** 2))

    best = max(np.arange(-max_deg, max_deg + 1e-9, 0.25), key=score)
    return float(max(np.arange(best - 0.25, best + 0.25 + 1e-9, 0.05), key=score))


def rotate_gray(gray, angle):
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _weighted_median(values, weights):
    if len(values) == 0:
        return 0.0
    o = np.argsort(values)
    v, wt = values[o], weights[o].astype(np.float64)
    cw = np.cumsum(wt)
    return float(v[np.searchsorted(cw, cw[-1] / 2.0)])


def _runs(mask):
    """(start, end) of every run of True values."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def find_gutter(ink, box, ch, max_gutter_frac=0.08, rules_x=()):
    """Dynamic projection-profile gutter search in the middle 30-70 % of the text box.
    Returns ((g0, g1), note) or (None, note). Works per page, so drift between scans is harmless."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    if w < 20 * ch:
        return None, "too narrow for two columns"
    a, b = x0 + int(0.3 * w), x0 + int(0.7 * w)
    for rx in rules_x:                                     # a printed column rule is the gutter
        if a <= rx <= b:
            return (int(rx - 0.5 * ch), int(rx + 0.5 * ch)), f"column rule at x={int(rx)}"
    prof = ink[y0:y1, x0:x1].sum(0).astype(np.float64)
    k = max(3, int(ch / 2)) | 1
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    ref = np.percentile(prof, 75)
    if ref <= 0:
        return None, "no ink"
    low = prof < 0.12 * ref
    best = None
    for s, e in _runs(low):
        s, e = s + x0, e + x0
        if s >= a and e <= b and (best is None or e - s > best[1] - best[0]):
            best = (int(s), int(e))
    if best is None or best[1] - best[0] < 0.5 * ch:
        return None, "no gutter"
    if best[1] - best[0] > max_gutter_frac * w:
        return None, f"central gap {best[1] - best[0]}px is too wide for a gutter (poetry?)"
    left = prof[: best[0] - x0].mean() if best[0] > x0 else 0
    right = prof[best[1] - x0:].mean() if best[1] < x1 else 0
    if min(left, right) < 0.3 * ref:
        return None, "one side of the gap is nearly empty"
    return best, f"gutter at x={best[0]}-{best[1]}"


def _zones(ink, box, gutter, ch):
    """Top-to-bottom zones: 'full' where text crosses the gutter (titles), 'split' elsewhere."""
    x0, y0, x1, y1 = box
    g0, g1 = gutter
    row_ink = ink[y0:y1, x0:x1].any(1)
    crossing = ink[y0:y1, g0:g1].any(1)
    # close small vertical gaps so a text line with its dots is one band
    gap = max(1, int(0.3 * ch))
    filled = row_ink.copy()
    for s, e in _runs(~row_ink):
        if e - s <= gap and s > 0 and e < len(row_ink):
            filled[s:e] = True
    bands = []
    for s, e in _runs(filled):
        full = crossing[s:e].sum() >= max(2, 0.2 * (e - s))
        kind = "full" if full else "split"
        if bands and bands[-1][0] == kind:
            bands[-1][2] = e + y0
        else:
            bands.append([kind, s + y0, e + y0])
    # a split zone must hold at least one full line of text
    out = []
    for kind, s, e in bands:
        if kind == "split" and e - s < 1.2 * ch and out:
            out[-1][2] = e
            continue
        if out and out[-1][0] == kind:
            out[-1][2] = e
        else:
            out.append([kind, s, e])
    return [tuple(z) for z in out]


def _split_tall(ink, bbox, max_h, ch):
    """Cut a region into pieces no taller than max_h, preferring blank rows between lines."""
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    if h <= max_h:
        return [bbox]
    k = int(np.ceil(h / max_h))
    rows = ink[y0:y1, x0:x1].sum(1).astype(np.float64)
    rows = np.convolve(rows, np.ones(3) / 3, mode="same")
    cuts, last = [], 0
    for i in range(1, k):
        ideal = int(h * i / k)
        lo, hi = max(last + int(ch), ideal - int(0.25 * h / k)), min(h - int(ch), ideal + int(0.25 * h / k))
        if hi <= lo:
            continue
        c = lo + int(np.argmin(rows[lo:hi] + np.abs(np.arange(lo, hi) - ideal) * 1e-3))
        cuts.append(c)
        last = c
    ys = [0] + cuts + [h]
    return [(x0, y0 + a, x1, y0 + b) for a, b in zip(ys[:-1], ys[1:]) if b - a > 0.5 * ch]


def analyze_page(gray, cfg=None):
    """Deskew and cut a page into reading-order regions for the vision model."""
    cfg = cfg or OCRConfig()
    notes = []
    ink0 = binarize(gray)
    angle = estimate_skew(ink0)
    if abs(angle) >= 0.05:
        gray = rotate_gray(gray, angle)
        ink0 = binarize(gray)
        notes.append(f"deskewed {angle:+.2f}°")
    H, W = ink0.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ink0.astype(np.uint8), connectivity=8)
    xs, ys, ws, hs, ar = (stats[1:, i] for i in range(5))
    ok = (hs >= 4) & (ar >= 12) & (hs < 0.08 * H) & (ws < 0.5 * W)
    ch = float(max(_weighted_median(hs[ok].astype(np.float64), ar[ok]) if ok.any() else 20.0, 8.0))

    keep = np.zeros(n, bool)
    keep[1:] = ar >= max(4, int((0.06 * ch) ** 2))                       # speckle
    keep[1:] &= ~((ws > 0.85 * W) | (hs > 0.85 * H))                       # frame, scan shadow
    edge = (xs <= 1) | (ys <= 1) | (xs + ws >= W - 1) | (ys + hs >= H - 1)
    keep[1:] &= ~(edge & (ar > 50 * ch * ch))                              # dark scan borders
    vrule = (hs > 0.25 * H) & (ws < 0.6 * ch)
    hrule = (ws > 8 * ch) & (hs < 0.35 * ch)
    rules_x = [float(cents[i + 1, 0]) for i in np.nonzero(vrule & keep[1:])[0]]
    keep[1:] &= ~(vrule | hrule)
    ink = keep[labels]

    if not ink.any():
        return PageLayout(gray, ink, angle, ch, (0, 0, W, H), [], None, notes + ["blank page"])
    rr, cc = np.nonzero(ink.any(1))[0], np.nonzero(ink.any(0))[0]
    pad = int(0.6 * ch)
    box = (max(0, int(cc[0]) - pad), max(0, int(rr[0]) - pad), min(W, int(cc[-1]) + pad + 1),
           min(H, int(rr[-1]) + pad + 1))

    gutter = None
    if str(cfg.columns) != "1":
        gutter, why = find_gutter(ink, box, ch, cfg.max_gutter_frac if str(cfg.columns) == "auto" else 1.0,
                                  rules_x)
        notes.append(why)
    zones = [("full", box[1], box[3])]
    if gutter is not None:
        zones = _zones(ink, box, gutter, ch)
        if not any(k == "split" and e - s >= 3 * ch for k, s, e in zones):
            notes.append("gutter too short, reading as one column")
            gutter, zones = None, [("full", box[1], box[3])]

    raw = []
    for kind, s, e in zones:
        s, e = max(0, s - pad), min(H, e + pad)
        if kind == "full":
            raw.append(Region("full", (box[0], s, box[2], e)))
        else:
            mid = (gutter[0] + gutter[1]) // 2
            raw += [Region("right", (mid, s, box[2], e)), Region("left", (box[0], s, mid, e))]

    # keep crops sharp: a crop is only shrunk to min_scale before it is cut at blank rows
    max_h_for = lambda w: int(cfg.max_pixels / max(1, w) / (cfg.min_scale ** 2))
    main = ok & keep[1:] & (hs >= 0.5 * ch) & (hs <= 4 * ch)
    mx, my = cents[1:, 0][main], cents[1:, 1][main]
    regions = []
    for r in raw:
        x0, y0, x1, y1 = r.bbox
        if not ink[y0:y1, x0:x1].any():
            continue
        for bb in _split_tall(ink, r.bbox, max_h_for(x1 - x0), ch):
            inside = (mx >= bb[0]) & (mx < bb[2]) & (my >= bb[1]) & (my < bb[3])
            regions.append(Region(r.kind, tuple(int(v) for v in bb), int(2 * inside.sum())))
    if gutter is not None:
        notes.append(f"{sum(1 for k, *_ in zones if k == 'split')} two-column zone(s)")
    return PageLayout(gray, ink, angle, ch, box, regions, gutter, notes)


def crop_image(layout, region, max_pixels):
    """Region of the deskewed page as RGB, never larger than max_pixels."""
    x0, y0, x1, y1 = region.bbox
    im = Image.fromarray(layout.gray[y0:y1, x0:x1]).convert("RGB")
    w, h = im.size
    if w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
    return im


def draw_regions(layout, width=900):
    """Page thumbnail with numbered crops in reading order (green = full width, blue/orange = columns)."""
    img = cv2.cvtColor(layout.gray, cv2.COLOR_GRAY2BGR)
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    col = {"full": (74, 163, 22), "right": (235, 99, 37), "left": (0, 140, 255)}
    for i, r in enumerate(layout.regions):
        x0, y0, x1, y1 = (int(v * s) for v in r.bbox)
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), col[r.kind], 2)
        cv2.putText(img, str(i + 1), (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[r.kind], 2, cv2.LINE_AA)
    if layout.gutter is not None:
        for g in layout.gutter:
            cv2.line(img, (int(g * s), 0), (int(g * s), img.shape[0]), (180, 180, 180), 1)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def render_pdf_page(doc, page_no, dpi):
    """page_no is 1-based. Returns (grayscale array, points per pixel)."""
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    return gray, page.rect.width / pix.width


# ============================================================================ script features
URDU_ONLY = set("\u0679\u0688\u0691\u06BA\u06BE\u06C1\u06C2\u06D2\u06D3")   # ٹ ڈ ڑ ں ھ ہ ۂ ے ۓ
PERSO_URDU = set("\u067E\u0686\u0698\u06AF")                                 # پ چ ژ گ
HARAKAT_RE = re.compile("[\u064B-\u0652\u0670]")
AR_LETTER = "\u0620-\u064A\u066E-\u06D3\u06FA-\u06FF"
AR_MARK = "\u064B-\u065F\u0670\u06D6-\u06ED"
LETTER_RE = re.compile(f"[{AR_LETTER}]")
LATIN_RE = re.compile("[A-Za-z]")
URDU_WORDS = set("کے کی کا ہے ہیں میں اور سے کو نے یہ وہ بھی تھا تھی تھے پر گیا ہو کہ جو".split())
ARABIC_WORDS = set("في من على إلى عن التي الذي الله قال هذا ذلك كان إن أن لا ما هو".split())
URDU_MAP = str.maketrans({"\u0643": "\u06A9",   # ك -> ک
                          "\u064A": "\u06CC",   # ي -> ی
                          "\u0649": "\u06CC",   # ى -> ی
                          "\u0647": "\u06C1"})  # ه -> ہ
URDU_DIGITS = str.maketrans({chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)})   # ٠١٢ -> ۰۱۲


def n_letters(t):
    return len(LETTER_RE.findall(t))


def harakat_rate(t):
    n = n_letters(t)
    return len(HARAKAT_RE.findall(t)) / n if n else 0.0


def line_script(line):
    """'urdu', 'arabic' or 'unclear' for one line of model output."""
    words = re.findall(f"[{AR_LETTER}{AR_MARK}]+", line)
    if not words:
        return "unclear"
    # judge on the unvocalised words: a vocalised Arabic quotation inside an Urdu sentence
    # must not turn the whole line into Arabic
    plain = [w for w in words if len(HARAKAT_RE.findall(w)) < 2]
    urdu_letters = sum(c in URDU_ONLY or c in PERSO_URDU for w in plain for c in w)
    bare = [re.sub(f"[{AR_MARK}]", "", w) for w in plain]
    urdu_hits = sum(1 for w in bare if w.translate(URDU_MAP) in URDU_WORDS)
    if not urdu_letters and not urdu_hits and (len(plain) <= 0.4 * len(words) or harakat_rate(line) >= 0.15):
        return "arabic"
    arabic_hits = sum(1 for w in bare if w in ARABIC_WORDS or (w.startswith("ال") and len(w) > 3))
    if urdu_letters or urdu_hits >= 1:
        return "urdu" if urdu_letters + 2 * urdu_hits >= arabic_hits else "arabic"
    if arabic_hits or "\u0629" in line:
        return "arabic"
    return "unclear"


def _urdu_word(w):
    """Normalise one word of an Urdu line. Words carrying Arabic vocalisation are left untouched."""
    if len(HARAKAT_RE.findall(w)) >= 2 or re.search("[\u064B-\u064D\u06D6-\u06ED]", w):
        return w
    w = w.translate(URDU_MAP)
    w = re.sub(f"\u0640+(?![{AR_MARK}])", "", w)                     # tatweel (keep ـٰ)
    return unicodedata.normalize("NFC", w)                            # ہ + ٔ -> ۂ , ے + ٔ -> ۓ


def _urdu_punct(t):
    ctx = f"(?<=[{AR_LETTER}{AR_MARK}\\]])"
    t = re.sub(ctx + r"[ \t]*,", "\u060C", t)
    t = re.sub(ctx + r"[ \t]*;", "\u061B", t)
    t = re.sub(ctx + r"[ \t]*\?(?!\])", "\u061F", t)
    t = re.sub(ctx + r"[ \t]*\.(?=\s|$)", "\u06D4", t)
    return t


def normalize_text(text, unclear="keep"):
    """Line-by-line: Urdu lines get Urdu letter forms and punctuation, Arabic lines (dense
    tashkeel, Arabic function words, ta marbuta) are returned byte for byte."""
    out = []
    for line in text.split("\n"):
        s = line_script(line)
        if s == "urdu" or (s == "unclear" and unclear == "urdu" and n_letters(line)):
            line = re.sub(f"[{AR_LETTER}{AR_MARK}\u0640]+", lambda m: _urdu_word(m.group(0)), line)
            line = _urdu_punct(line.translate(URDU_DIGITS))
        out.append(line)
    return "\n".join(out)


# ============================================================================ output checks
_LOOP_RE = re.compile(r"(.{4,120}?)(?:\s*\1){3,}", re.S)


def find_loop(text):
    """(start, end, unit) of the first chunk repeated 4+ times in a row, else None."""
    m = _LOOP_RE.search(text)
    if not m or len(m.group(0)) < 40:
        return None
    return m.start(), m.end(), m.group(1)


def trim_loop(text):
    hit = find_loop(text)
    if not hit:
        return text, False
    s, e, unit = hit
    return text[:s] + unit + " [?…]" + text[e:], True


def strip_wrappers(text):
    """Remove a ```markdown fence the model sometimes wraps its answer in."""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", t, re.S)
    return (m.group(1) if m else t).strip()


HAWASHI_RE = re.compile(r"^\s*#{1,6}\s*\**\s*(hawashi|hawāshī|marginalia|footnotes|حواشی|حاشیہ|الحواشي|حاشية)\b.*$",
                        re.I | re.M)


def split_hawashi(md):
    """(main text, hawashi text) from the model's Markdown."""
    m = HAWASHI_RE.search(md)
    if not m:
        return md.strip(), ""
    return md[:m.start()].strip(), HAWASHI_RE.sub("", md[m.end():]).strip()


# ============================================================================ confidence marks
def _bytes_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DEC = _bytes_decoder()


def token_char_spans(tokenizer, ids, text):
    """Character span each generated token produced in `text` (byte-level BPE), or None."""
    toks = tokenizer.convert_ids_to_tokens([int(i) for i in ids])
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    tb = []
    for i, t in zip(ids, toks):
        if int(i) in special or t is None:
            tb.append(b"")
            continue
        try:
            tb.append(bytes(_BYTE_DEC[c] for c in t))
        except KeyError:
            return None
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    char_end_byte, pos, out = [], 0, []
    for b in b"".join(tb):
        pos += 1
        s = dec.decode(bytes([b]))
        char_end_byte += [pos] * len(s)
        out.append(s)
    s = dec.decode(b"", final=True)
    char_end_byte += [pos] * len(s)
    out.append(s)
    if "".join(out) != text:
        return None
    spans, b0, ci = [], 0, 0
    for t in tb:
        b1 = b0 + len(t)
        c0 = ci
        while ci < len(char_end_byte) and char_end_byte[ci] <= b1:
            ci += 1
        spans.append((c0, ci))
        b0 = b1
    return spans


def flag_uncertain(text, spans, logprobs, threshold, mark="[?]"):
    """Append `mark` to every word whose least-confident token has probability < threshold.
    Returns (text, n_flagged, n_words)."""
    words = [m for m in re.finditer(r"\S+", text) if LETTER_RE.search(m.group(0)) or re.search(r"\w", m.group(0))]
    if spans is None or threshold <= 0:
        return text, 0, len(words)
    char_p = np.ones(len(text))
    for (c0, c1), lp in zip(spans, logprobs):
        p = float(np.exp(lp))
        if c1 > c0:
            char_p[c0:c1] = np.minimum(char_p[c0:c1], p)
        elif c0 < len(text):
            char_p[c0] = min(char_p[c0], p)
    out, last, n = [], 0, 0
    for m in words:
        w0, w1 = m.span()
        if m.group(0) == mark or float(char_p[w0:w1].min()) >= threshold:
            continue
        core_end = w1
        while core_end > w0 and unicodedata.category(text[core_end - 1])[0] == "P":
            core_end -= 1
        if core_end == w0:
            core_end = w1
        out += [text[last:core_end], mark]
        last = core_end
        n += 1
    out.append(text[last:])
    return "".join(out), n, len(words)


# ============================================================================ model
def _make_confidence_recorder():
    """Logits processor that keeps log p of the greedy token at every step (one float per step,
    instead of output_scores' full 150k-vocabulary tensors)."""
    import torch
    from transformers import LogitsProcessor

    class ConfidenceRecorder(LogitsProcessor):
        def __init__(self):
            self.steps = []

        def __call__(self, input_ids, scores):
            self.steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.cpu())
            return scores

    return ConfidenceRecorder()


class QwenVLEngine:
    """Qwen2.5-VL wrapper: batched greedy generation with per-token log-probabilities."""

    def __init__(self, model, processor, cfg, log=print):
        self.model, self.processor, self.cfg, self.log = model, processor, cfg, log
        self.tok = processor.tokenizer
        self.tok.padding_side = "left"
        self.batch_size = max(1, int(cfg.batch_size))
        stop = {self.tok.eos_token_id, self.tok.pad_token_id}
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0:
                stop.add(tid)
        self.stop_ids = {s for s in stop if s is not None}

    @classmethod
    def load(cls, cfg, log=print):
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as ModelClass

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU. In Colab: Runtime ▸ Change runtime type ▸ L4 or A100, then re-run.")
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9
        bf16 = torch.cuda.is_bf16_supported()
        dt = torch.bfloat16 if bf16 else torch.float16
        quant = cfg.quantize if cfg.quantize != "auto" else ("4bit" if vram < 20 else "none")
        kw = {"device_map": "cuda:0", "attn_implementation": "sdpa",
              ("dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"): dt}
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=dt)
        log(f"GPU: {props.name} ({vram:.0f} GB) · {'bfloat16' if bf16 else 'float16'} · "
            f"{'4-bit' if quant == '4bit' else 'full precision'} · transformers {transformers.__version__}")
        if quant == "4bit":
            log("  ⚠ 4-bit weights read Nastaliq noticeably worse. Use an L4 or A100 for real runs.")
        model = ModelClass.from_pretrained(cfg.model_id, **kw).eval()
        processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=64 * 28 * 28, max_pixels=cfg.max_pixels)
        log(f"Model: {cfg.model_id} · GPU memory in use {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        return cls(model, processor, cfg, log)

    def _chat(self, user_text, image=True, system=SYSTEM_PROMPT):
        content = ([{"type": "image"}] if image else []) + [{"type": "text", "text": user_text}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _generate(self, texts, images, max_new_tokens, repetition_penalty):
        import torch
        from transformers import LogitsProcessorList
        kw = {"text": texts, "padding": True, "return_tensors": "pt"}
        if images:
            kw["images"] = images
        inputs = self.processor(**kw).to(self.model.device)
        rec = _make_confidence_recorder()
        with torch.inference_mode():
            seqs = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False, num_beams=1,
                                       temperature=None, top_p=None, top_k=None,
                                       repetition_penalty=repetition_penalty, pad_token_id=self.tok.pad_token_id,
                                       logits_processor=LogitsProcessorList([rec]))
        n_in = inputs["input_ids"].shape[1]
        lps = torch.stack(rec.steps, 1).numpy() if rec.steps else np.zeros((len(texts), 0))
        res = []
        for i in range(len(texts)):
            gen = seqs[i, n_in:].tolist()
            end, finished = len(gen), False
            for j, t in enumerate(gen):
                if t in self.stop_ids:
                    end, finished = j, True
                    break
            ids, lp = gen[:end], np.where(np.isfinite(lps[i, :end]), lps[i, :end], -20.0)
            text = self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            res.append({"text": text, "ids": ids, "lp": lp, "finished": finished,
                        "mean_lp": float(lp.mean()) if len(lp) else 0.0})
        return res

    def ocr(self, images, budgets, repetition_penalty=1.0, prompt=USER_PROMPT):
        """Read many crops, batched (the batch shrinks on out-of-memory)."""
        import torch
        order = sorted(range(len(images)), key=lambda i: budgets[i])
        results, bs, i = [None] * len(images), self.batch_size, 0
        chat = self._chat(prompt)
        while i < len(order):
            idx = order[i:i + bs]
            try:
                out = self._generate([chat] * len(idx), [images[k] for k in idx],
                                     max(budgets[k] for k in idx), repetition_penalty)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                self.log(f"    (GPU memory full – batch size reduced to {bs})")
                continue
            for k, r in zip(idx, out):
                results[k] = r
            i += len(idx)
        return results

    def chat_text(self, prompt, system, max_new_tokens=1024):
        return self._generate([self._chat(prompt, image=False, system=system)], None, max_new_tokens, 1.0)[0]["text"]


# ============================================================================ page pipeline
def _budget(region, cfg):
    return int(min(cfg.max_new_tokens, max(384, 2 * region.est_letters + 192)))


def _halves(layout, region):
    x0, y0, x1, y1 = region.bbox
    if y1 - y0 < 4 * layout.ch:
        return None
    parts = _split_tall(layout.ink, region.bbox, int(0.55 * (y1 - y0)), layout.ch)
    if len(parts) < 2:
        return None
    return [Region(region.kind, p, max(1, region.est_letters // len(parts))) for p in parts]


def _postprocess(engine, r, cfg):
    raw = r["text"]
    trimmed, looped = trim_loop(raw)
    spans = None if looped else token_char_spans(engine.tok, r["ids"], raw)
    flagged, nf, nw = flag_uncertain(raw, spans, r["lp"], cfg.conf_flag, cfg.flag_mark)
    if looped:
        flagged = trimmed
    body, haw = split_hawashi(strip_wrappers(flagged))
    return {"body": normalize_text(body, cfg.normalize_unclear), "hawashi": normalize_text(haw, cfg.normalize_unclear),
            "raw": raw, "n_flagged": nf, "n_words": nw, "finished": bool(r["finished"]),
            "mean_p": round(float(np.exp(r["mean_lp"])), 3)}


def assess(region_recs, cfg):
    """Reasons a page goes to the review queue (empty list = looks fine)."""
    reasons = []
    n_words = sum(d["n_words"] for d in region_recs)
    n_flag = sum(d["n_flagged"] for d in region_recs)
    for i, d in enumerate(region_recs, 1):
        text = d["body"] + "\n" + d["hawashi"]
        letters = n_letters(text)
        for x in d["issues"]:
            reasons.append(f"crop {i}: {x}")
        if d["est_letters"] >= 40:
            y = letters / d["est_letters"]
            d["yield"] = round(y, 2)
            if y < cfg.review_min_yield:
                reasons.append(f"crop {i}: only {y:.0%} of the expected text came back")
        lat = len(LATIN_RE.findall(text))
        if lat > 20 and lat > 0.3 * (lat + letters):
            reasons.append(f"crop {i}: mostly Latin text (model commentary?)")
    if n_words and n_flag / n_words > cfg.review_flag_ratio:
        reasons.append(f"{n_flag}/{n_words} words marked {cfg.flag_mark}")
    return reasons


def process_page(engine, doc, page_no, cfg, log=print):
    t0 = time.time()
    gray, pt_per_px = render_pdf_page(doc, page_no, cfg.dpi)
    lay = analyze_page(gray, cfg)
    to_pt = lambda bb: [round(v * pt_per_px, 2) for v in bb]
    rec = {"page": page_no, "angle": round(lay.angle, 2), "layout_notes": lay.notes, "text_box": to_pt(lay.text_box),
           "regions": [], "corrected": None}
    if not lay.regions:
        rec.update(n_words=0, n_flagged=0, review={"needed": False, "reasons": []}, seconds=round(time.time() - t0, 1))
        return rec, lay
    imgs = [crop_image(lay, r, cfg.max_pixels) for r in lay.regions]
    first = engine.ocr(imgs, [_budget(r, cfg) for r in lay.regions])

    region_recs = []
    for region, img, r in zip(lay.regions, imgs, first):
        parts, issues, retried = [r], [], None
        if find_loop(r["text"]):
            r2 = engine.ocr([img], [_budget(region, cfg)], repetition_penalty=cfg.loop_retry_penalty)[0]
            retried = "loop"
            if find_loop(r2["text"]):
                issues.append("repetition loop")
            else:
                parts = [r2]
        elif not r["finished"]:
            retried = "length"
            halves = _halves(lay, region)
            if halves:
                parts = engine.ocr([crop_image(lay, h, cfg.max_pixels) for h in halves], [cfg.max_new_tokens] * len(halves))
            else:
                parts = engine.ocr([img], [cfg.max_new_tokens])
            if not all(p["finished"] for p in parts):
                issues.append("output cut off")
        pp = [_postprocess(engine, p, cfg) for p in parts]
        body = "\n\n".join(p["body"] for p in pp if p["body"])
        if not body.strip() and region.est_letters >= 40:
            issues.append("no text returned")
        region_recs.append({"kind": region.kind, "bbox": to_pt(region.bbox), "est_letters": region.est_letters,
                            "body": body, "hawashi": "\n".join(p["hawashi"] for p in pp if p["hawashi"]),
                            "raw": "\n\n".join(p["raw"] for p in pp), "n_flagged": sum(p["n_flagged"] for p in pp),
                            "n_words": sum(p["n_words"] for p in pp), "mean_p": min(p["mean_p"] for p in pp),
                            "finished": all(p["finished"] for p in pp), "issues": issues, "retried": retried})
    reasons = assess(region_recs, cfg)
    rec.update(regions=region_recs, n_words=sum(d["n_words"] for d in region_recs),
               n_flagged=sum(d["n_flagged"] for d in region_recs),
               review={"needed": bool(reasons), "reasons": reasons}, seconds=round(time.time() - t0, 1))
    return rec, lay


def page_markdown(rec):
    """Final Markdown of one page: the reviewer's text if saved, else main text + hawashi."""
    if rec.get("corrected") is not None:
        return rec["corrected"].strip()
    body = "\n\n".join(d["body"] for d in rec["regions"] if d["body"].strip())
    haw = "\n".join(d["hawashi"] for d in rec["regions"] if d["hawashi"].strip())
    return (body + ("\n\n### Hawashi\n\n" + haw if haw else "")).strip()


# ============================================================================ cache
def cache_path(chap_dir, page_no):
    return os.path.join(chap_dir, "_cache", f"page_{page_no:04d}.json")


def save_record(chap_dir, rec):
    p = cache_path(chap_dir, rec["page"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(p + ".tmp", p)


def load_record(chap_dir, page_no):
    p = cache_path(chap_dir, page_no)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_page_markdown(chap_dir, rec):
    os.makedirs(os.path.join(chap_dir, "pages"), exist_ok=True)
    with open(os.path.join(chap_dir, "pages", f"page_{rec['page']:04d}.md"), "w", encoding="utf-8") as f:
        f.write(page_markdown(rec) + "\n")


def write_review_queue(chap_dir, recs):
    """needs_review.txt: one line per flagged page that has not been corrected yet."""
    lines = [f"page {r['page']}\t" + "; ".join(r["review"]["reasons"]) for r in recs
             if r and r["review"]["needed"] and r.get("corrected") is None]
    with open(os.path.join(chap_dir, "needs_review.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


# ============================================================================ alerts
def notify(url, message, log=print):
    """POST a short message to a Discord / Slack / generic webhook. Never raises."""
    if not url:
        return False
    import urllib.request
    try:
        req = urllib.request.Request(url, data=json.dumps({"content": message, "text": message}).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "ocr-notebook"})
        urllib.request.urlopen(req, timeout=10).read()
        return True
    except Exception as e:
        log(f"  (webhook failed: {type(e).__name__})")
        return False


# ============================================================================ run
def run_pages(engine, pdf_path, pages, chap_dir, cfg, redo=(), show=None, log=print):
    """OCR every page not yet cached (plus `redo`). Each page is saved as soon as it is done."""
    doc = pymupdf.open(pdf_path)
    todo = [p for p in pages if p in set(redo) or load_record(chap_dir, p) is None]
    if len(pages) - len(todo):
        log(f"{len(pages) - len(todo)} page(s) already done (cached) – resuming.")
    os.makedirs(chap_dir, exist_ok=True)
    with open(os.path.join(chap_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict(cfg), f, indent=1)
    notify(cfg.webhook_url, f"OCR started: {os.path.basename(chap_dir)}, {len(todo)} page(s) to do.", log)
    times, n_review, p = [], 0, None
    try:
        for k, p in enumerate(todo):
            if not 1 <= p <= len(doc):
                log(f"page {p} is outside the PDF (1–{len(doc)}), skipped")
                continue
            rec, lay = process_page(engine, doc, p, cfg, log=log)
            save_record(chap_dir, rec)
            write_page_markdown(chap_dir, rec)
            times.append(rec["seconds"])
            n_review += rec["review"]["needed"]
            left = (len(todo) - k - 1) * float(np.mean(times[-5:]))
            eta = f"{k + 1}/{len(todo)} done" + (f", ~{int(left // 60)} min left" if left else "")
            if show:
                show(rec, lay, eta)
            log(f"page {p}: {rec['seconds']}s, {len(rec['regions'])} crop(s), {rec['n_flagged']} [?] · {eta}"
                + (" · REVIEW: " + "; ".join(rec["review"]["reasons"][:2]) if rec["review"]["needed"] else ""))
            if cfg.webhook_every and (k + 1) % cfg.webhook_every == 0 and k + 1 < len(todo):
                notify(cfg.webhook_url, f"OCR {os.path.basename(chap_dir)}: {eta}, {n_review} page(s) flagged.", log)
    except Exception as e:
        notify(cfg.webhook_url, f"OCR {os.path.basename(chap_dir)} STOPPED at page {p}: {type(e).__name__}: {e}. "
                                f"Re-run the cell to resume.", log)
        raise
    finally:
        write_review_queue(chap_dir, [load_record(chap_dir, q) for q in pages])
    notify(cfg.webhook_url, f"OCR finished: {os.path.basename(chap_dir)}, {len(times)} page(s), "
                            f"{n_review} flagged for review.", log)
    return todo


# ============================================================================ searchable PDF
AMIRI_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf"
FONT_CANDIDATES = ("/usr/share/fonts/truetype/fonts-hosny-amiri/Amiri-Regular.ttf",
                   "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf",
                   "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf")


def text_layer_font(cache_dir, log=print):
    """Path to an Arabic-script TTF whose cmap maps only base code points. PDF readers turn glyphs back
    into text through that map, so presentation forms (U+FB50–U+FEFF) must go or search breaks."""
    out = os.path.join(cache_dir, "textlayer_font.ttf")
    if os.path.exists(out):
        return out
    os.makedirs(cache_dir, exist_ok=True)
    src = next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
    if src is None:
        import urllib.request
        src = os.path.join(cache_dir, "Amiri-Regular.ttf")
        urllib.request.urlretrieve(AMIRI_URL, src)
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(src)
        for t in f["cmap"].tables:
            t.cmap = {k: v for k, v in t.cmap.items() if not (0xFB50 <= k <= 0xFDFF or 0xFE70 <= k <= 0xFEFF)}
        f.save(out)
    except ImportError:
        log("  ⚠ fontTools missing: search in the PDF may not match every word (pip install fonttools).")
        return src
    return out


def _char_class(c):
    o = ord(c)
    if c.isdigit() or (c.isascii() and c.isalpha()) or 0xC0 <= o <= 0x24F:
        return "L"
    if 0x0590 <= o <= 0x08FF or 0xFB1D <= o <= 0xFEFF:
        return "R"
    return "N"


def bidi_runs(line):
    """Split a right-to-left line into (is_rtl, text) runs in logical order. Digits (also \u06F1\u06F2\u06F3) and
    Latin are left-to-right; spaces and punctuation join their neighbours, or the RTL side if mixed."""
    k = [_char_class(c) for c in line]
    i = 0
    while i < len(k):
        if k[i] != "N":
            i += 1
            continue
        j = i
        while j < len(k) and k[j] == "N":
            j += 1
        prev = next((k[a] for a in range(i - 1, -1, -1) if k[a] != "N"), "R")
        nxt = k[j] if j < len(k) else "R"
        k[i:j] = [prev if prev == nxt else "R"] * (j - i)
        i = j
    out = []
    for c, t in zip(line, k):
        if out and out[-1][0] == (t == "R"):
            out[-1][1] += c
        else:
            out.append([t == "R", c])
    return [(r, s) for r, s in out]


def plain_lines(md, mark="[?]"):
    """Markdown -> plain text lines for the hidden layer (no #, **, [?] or hawashi heading)."""
    out = []
    for ln in md.split("\n"):
        if HAWASHI_RE.match(ln):
            continue
        ln = re.sub(r"^\s*#{1,6}\s*", "", ln)
        ln = re.sub(r"^\s*(?:[-*+]|>)\s+", "", ln)
        ln = re.sub(r"(\*\*|__|`)", "", ln)
        ln = ln.replace(mark, "").replace("[?…]", "")
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln and not re.fullmatch(r"[-*_=\s]+", ln):
            out.append(ln)
    return out


def write_rtl_lines(writer, font, lines, rect, max_fs=14.0):
    """Lay `lines` evenly down `rect`, right-aligned, each shrunk to fit the width."""
    if not lines:
        return
    x0, y0, x1, y1 = rect
    pitch = (y1 - y0) / len(lines)
    fs0 = max(2.0, min(max_fs, 0.8 * pitch))
    for i, ln in enumerate(lines):
        width = font.text_length(ln, fs0)
        fs = fs0 if width <= (x1 - x0) else max(1.0, fs0 * (x1 - x0) / width)
        x, y = x1, y0 + pitch * i + 0.5 * pitch + 0.35 * fs
        for rtl, t in bidi_runs(ln):
            w = font.text_length(t, fs)
            # a closing RLM keeps a final vowel mark on its letter, and the runs on one line,
            # when a reader extracts the text
            writer.append((x - w, y), t + "\u200F" if rtl else t, font=font, fontsize=fs, right_to_left=rtl)
            x -= w


def _toc_entries(recs):
    toc = []
    for idx, r in enumerate(recs, 1):
        for ln in page_markdown(r).split("\n"):
            m = re.match(r"^(#{1,3})\s+(.+?)\s*#*\s*$", ln)
            if m and not HAWASHI_RE.match(ln):
                title = re.sub(r"(\*\*|__|\[\?…?\])", "", m.group(2)).strip()
                if title:
                    toc.append([len(m.group(1)), title[:120], idx])
    level = 0
    for e in toc:                          # PDF outlines may only go one level deeper at a time
        e[0] = min(e[0], level + 1)
        level = e[0]
    return toc


def build_searchable_pdf(pdf_path, recs, out_path, font_path, cfg, title="", appendix=None):
    """Original scans with an invisible, selectable text layer (render mode 3) and bookmarks."""
    src = pymupdf.open(pdf_path)
    out = pymupdf.open()
    font = pymupdf.Font(fontfile=font_path)
    for r in recs:
        page = src[r["page"] - 1]
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.show_pdf_page(new.rect, src, r["page"] - 1)
        tw = pymupdf.TextWriter(new.rect)
        if r.get("corrected") is not None or not r["regions"]:
            write_rtl_lines(tw, font, plain_lines(page_markdown(r), cfg.flag_mark), r["text_box"])
        else:
            for d in r["regions"]:
                write_rtl_lines(tw, font, plain_lines(d["body"] + "\n" + d["hawashi"], cfg.flag_mark), d["bbox"])
        tw.write_text(new, render_mode=3)
    if appendix:
        _append_text_pages(out, appendix)
    if cfg.bookmarks != "off":
        toc = _toc_entries(recs) if cfg.bookmarks == "headings" else []
        if not toc:
            toc = [[1, f"p. {r['page']}", i] for i, r in enumerate(recs, 1)]
        if appendix:
            toc.append([1, "Appendix: hawashi (machine translation)", len(recs) + 1])
        out.set_toc(toc)
    out.set_metadata({"title": title, "creator": "Zero-shot Urdu/Arabic OCR (Qwen2.5-VL)",
                      "producer": f"PyMuPDF {pymupdf.VersionBind}"})
    out.save(out_path, garbage=3, deflate=True)
    return out_path


def _append_text_pages(doc, text, width=595, height=842, margin=56):
    """Plain visible pages (Latin script) at the end of the PDF."""
    chunks, cur = [], []
    for ln in text.split("\n"):
        cur.append(ln)
        if len(cur) >= 55:
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    for c in chunks:
        p = doc.new_page(width=width, height=height)
        p.insert_textbox(pymupdf.Rect(margin, margin, width - margin, height - margin), "\n".join(c),
                         fontsize=9.5, fontname="helv")


# ============================================================================ hawashi gloss
TRANSLATE_SYSTEM = ("You translate classical Arabic and Urdu marginal notes into plain, accurate English. "
                    "Keep Qur'an and hadith references as they are. Output only the translation.")


def translate_hawashi(engine, recs, log=print):
    """English gloss of every page's hawashi (text-only pass of the same model). Returns Markdown."""
    parts = []
    for r in recs:
        haw = "\n".join(d["hawashi"] for d in r["regions"] if d["hawashi"].strip())
        if r.get("corrected") is not None:
            haw = split_hawashi(r["corrected"])[1]
        haw = haw.replace("[?]", "").strip()
        if not haw:
            continue
        en = engine.chat_text(haw, TRANSLATE_SYSTEM, max_new_tokens=min(2048, 64 + 3 * len(haw)))
        parts.append(f"## p. {r['page']}\n\n{en.strip()}")
        log(f"  hawashi p. {r['page']} translated")
    if not parts:
        return ""
    return ("# Hawashi — English gloss\n\n_Machine translation by the OCR model. Check against the original "
            "before citing._\n\n" + "\n\n".join(parts) + "\n")


# ============================================================================ export
def export_chapter(pdf_path, chap_dir, chap_name, pages, cfg, engine=None, log=print):
    recs = [load_record(chap_dir, p) for p in pages]
    missing = [p for p, r in zip(pages, recs) if r is None]
    if missing:
        raise RuntimeError(f"Pages not processed yet: {missing[:20]} – run the OCR cell first.")
    md = []
    for r in recs:
        write_page_markdown(chap_dir, r)
        md.append((f"[p. {r['page']}]\n\n" if cfg.page_markers else "") + page_markdown(r))
    md_path = os.path.join(chap_dir, f"{chap_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(md).strip() + "\n")
    with open(os.path.join(chap_dir, f"{chap_name}_raw.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(f"[p. {r['page']}]\n" + "\n\n".join(f"<crop {i} {d['kind']}>\n{d['raw']}"
                                                              for i, d in enumerate(r["regions"], 1))
                            for r in recs) + "\n")
    appendix = ""
    if cfg.translate_hawashi:
        if engine is None:
            log("  ⚠ TRANSLATE_HAWASHI needs the model loaded (step 4); skipped.")
        else:
            appendix = translate_hawashi(engine, recs, log)
            if appendix:
                with open(os.path.join(chap_dir, f"{chap_name}_Hawashi_English.md"), "w", encoding="utf-8") as f:
                    f.write(appendix)
    pdf_out = os.path.join(chap_dir, f"Searchable_{chap_name}.pdf")
    font = text_layer_font(os.path.join(chap_dir, "_cache"), log)
    build_searchable_pdf(pdf_path, recs, pdf_out, font, cfg, title=chap_name, appendix=appendix)
    n_review = write_review_queue(chap_dir, recs)
    flagged = sum(r["n_flagged"] for r in recs)
    log(f"Saved {md_path}\nSaved {pdf_out}\n({len(recs)} pages, {flagged} words marked {cfg.flag_mark}, "
        f"{n_review} flagged page(s) not reviewed yet – see needs_review.txt)")
    return md_path, pdf_out


def config_dict(cfg):
    d = asdict(cfg)
    d.pop("webhook_url", None)
    return d
